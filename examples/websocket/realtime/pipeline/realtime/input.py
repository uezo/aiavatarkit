"""PCM conversion and input observations without connection or task ownership."""

import audioop
from collections import OrderedDict
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
import inspect
import logging
import math
from typing import Any, Awaitable, Callable

from aiavatar.sts.models import STSRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputSilenceEvent:
    """Local acoustic silence while a Realtime input item is still uncommitted.

    ``request`` is an identity snapshot, not a recognized utterance. After any
    await, check ``is_current()`` before performing a side effect. The hook task
    survives resumed voice and is canceled on commit, replacement, text invocation,
    or disconnect. Callbacks must propagate cancellation. The event expires when
    its callback task finishes; cancellation cannot retract audio already sent.

    ``input_text`` is a snapshot of the latest transcription received for this
    input item (possibly partial or empty). ``current_input_text`` exposes later
    updates, and ``current_silence_duration`` exposes additional received silence.
    ``detected_at`` uses the event loop's monotonic clock, for an application
    deadline that includes time spent waiting for the callback to start.
    These observations do not commit input, wait for ASR, or generate a response.
    """

    session_id: str
    input_item_id: str
    silence_duration: float
    request: STSRequest
    _is_current: Callable[[], bool] = field(repr=False, compare=False)
    input_text: str = ""
    _get_input_text: Callable[[], str] | None = field(default=None, repr=False, compare=False)
    _get_silence_duration: Callable[[], float] | None = field(default=None, repr=False, compare=False)
    detected_at: float | None = None

    def is_current(self) -> bool:
        return self._is_current()

    @property
    def current_input_text(self) -> str:
        """Latest transcript for this input, or empty after invalidation."""
        if not self.is_current():
            return ""
        return self._get_input_text() if self._get_input_text else self.input_text

    @property
    def current_silence_duration(self) -> float:
        """Received consecutive silence in seconds; zero after invalidation.

        Missing audio packets do not advance this value. Unlike
        ``silence_duration``, this is a live observation, not a snapshot.
        Resumed voice resets it to zero without invalidating the event.
        """
        if not self.is_current():
            return 0.0
        return self._get_silence_duration() if self._get_silence_duration else self.silence_duration


@dataclass(frozen=True)
class RealtimeInputEvent:
    """Ordered observations of an admitted audio input, independent of UI captions.

    ``type`` is ``started``, ``transcript``, or ``ended``. Transcript text is
    cumulative, including empty final corrections. Ending stops eligibility;
    a late final transcript can still follow for a retained input. Only the
    latest 32 admitted item IDs are retained. Request is an identity snapshot.
    Handlers must be synchronous, fast, and perform no blocking work or I/O.
    """

    type: str
    session_id: str
    input_item_id: str
    request: STSRequest
    text: str | None = None


@dataclass
class InputAudio:
    """Keep PCM16 sample alignment and resampler history for one input stream."""

    sample_rate: int
    pending_audio: bytes = b""
    resample_state: Any = None

    def complete_samples(self, samples: bytes) -> bytes:
        audio = self.pending_audio + samples
        complete = len(audio) - len(audio) % 2
        self.pending_audio = audio[complete:]
        return audio[:complete]

    def resample(self, audio: bytes) -> bytes:
        if audio and self.sample_rate != 24000:
            audio, self.resample_state = audioop.ratecv(
                audio, 2, 1, self.sample_rate, 24000, self.resample_state,
            )
        return audio


@dataclass
class _InputObservation:
    request: STSRequest
    completed: bool = False


@dataclass
class InputState:
    """Current captions and bounded provenance of admitted provider input items."""

    item_id: str | None = None
    transcript: str = ""
    caption: str | None = None
    transcription_completed: bool = False
    ignored: bool = False
    observations: OrderedDict[str, _InputObservation] = field(default_factory=OrderedDict)
    observed_input_id: str | None = None

    def start(self, item_id: str | None) -> bool:
        """Reset caption state when a distinct nonempty provider item starts."""
        if not item_id or item_id == self.item_id:
            return False
        self.item_id = item_id
        self.transcript = ""
        self.caption = None
        self.transcription_completed = False
        self.ignored = False
        return True

    def is_ended(self, item_id: str | None) -> bool:
        """Whether a retained item's boundary must not reopen its admission."""
        return item_id in self.observations and item_id != self.observed_input_id

    def _event(self, type: str, item_id: str, text: str | None = None) -> RealtimeInputEvent:
        request = self.observations[item_id].request
        return RealtimeInputEvent(
            type=type, session_id=request.session_id, input_item_id=item_id,
            request=deepcopy(request), text=text,
        )

    def admit(self, request: STSRequest) -> Iterator[RealtimeInputEvent]:
        """Yield admission events in order, advancing state between dispatches."""
        item_id = self.item_id
        if not item_id or item_id in self.observations:
            return
        ended = self.end()
        if ended is not None:
            yield ended
        self.observations[item_id] = _InputObservation(STSRequest(
            session_id=request.session_id, user_id=request.user_id, context_id=request.context_id,
            transaction_id=request.transaction_id, channel=request.channel, metadata=deepcopy(request.metadata),
        ))
        while len(self.observations) > 32:
            self.observations.popitem(last=False)
        self.observed_input_id = item_id
        yield self._event("started", item_id)
        if self.caption is not None:
            event = self.transcript_event(item_id, self.caption, completed=self.transcription_completed)
            if event is not None:
                yield event
        elif self.transcription_completed:
            self.observations[item_id].completed = True

    def transcript_event(
        self, item_id: str | None, text: str, *, completed: bool = False,
    ) -> RealtimeInputEvent | None:
        """Observe a retained item, allowing one final correction after ending."""
        observation = self.observations.get(item_id)
        if observation is None or observation.completed:
            return None
        observation.completed = completed
        return self._event("transcript", item_id, text)

    def end(self) -> RealtimeInputEvent | None:
        item_id = self.observed_input_id
        if item_id is None:
            return None
        self.observed_input_id = None
        return self._event("ended", item_id)

    def update_transcript(
        self, item_id: str | None, text: str = "", *, completed: bool = False, failed: bool = False,
    ) -> bool:
        """Apply ASR only to the current eligible caption; report whether applied."""
        if item_id != self.item_id or self.ignored or self.transcription_completed:
            return False
        if failed:
            self.transcription_completed = True
            self.transcript = ""
            self.caption = None
        elif completed:
            self.transcription_completed = True
            self.transcript = ""
            self.caption = text
        else:
            self.transcript += text
            self.caption = self.transcript
        return True


@dataclass
class SilenceState:
    """Acoustic evidence and validity generations; callback tasks live elsewhere."""

    buffer: bytearray = field(default_factory=bytearray)
    voiced_samples: int = 0
    silent_samples: int = 0
    active: bool = False
    started: bool = False
    fired: bool = False
    generation: int = 0
    input_generation: int = 0

    def invalidate(self, *, end_input: bool = False, reset_audio: bool = True):
        self.generation += 1
        if end_input:
            self.active = False
            self.input_generation += 1
            if reset_audio:
                self.buffer.clear()
                self.voiced_samples = self.silent_samples = 0

    def reset_item(self):
        self.started = self.fired = False

    def is_pause(self, sample_rate: int, duration: float, min_speech: float) -> bool:
        return (
            not self.fired
            and self.voiced_samples > 0
            and self.voiced_samples >= math.ceil(min_speech * sample_rate)
            and self.silent_samples >= math.ceil(duration * sample_rate)
        )

    def observe(self, audio: bytes, sample_rate: int, threshold: float, duration: float) -> Iterator[None]:
        """Advance PCM frames, yielding each point where a pause can trigger.

        Consumers must evaluate a yielded pause before requesting the next frame:
        resumed voice later in the same audio packet can rearm the detector.
        """
        frame_samples = round(sample_rate * 0.02)
        frame_bytes = frame_samples * 2
        self.buffer.extend(audio)
        complete = len(self.buffer) // frame_bytes * frame_bytes
        for offset in range(0, complete, frame_bytes):
            frame = self.buffer[offset:offset + frame_bytes]
            if audioop.rms(frame, 2) / 32768 >= threshold:
                if not self.active and self.silent_samples >= math.ceil(duration * sample_rate):
                    self.voiced_samples = 0
                self.fired = False
                self.silent_samples = 0
                self.voiced_samples += frame_samples
            else:
                self.silent_samples += frame_samples
                yield None
        del self.buffer[:complete]
        # A delayed provider speech_started can arm already-observed silence.
        yield None


class InputCallbacks:
    """Register and dispatch application observations without owning their tasks."""

    def __init__(self):
        self.event_handlers: list[Callable[[RealtimeInputEvent], None]] = []
        self.silence_handlers: list[Callable[[InputSilenceEvent], Awaitable[None]]] = []

    def register_event(self, handler: Callable[[RealtimeInputEvent], None]):
        if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, "__call__", None)):
            raise TypeError("Realtime input observers must be synchronous")
        self.event_handlers.append(handler)
        return handler

    def register_silence(self, handler: Callable[[InputSilenceEvent], Awaitable[None]]):
        self.silence_handlers.append(handler)
        return handler

    def notify(self, event: RealtimeInputEvent | None):
        if event is None:
            return
        for handler in tuple(self.event_handlers):
            try:
                result = handler(replace(event, request=deepcopy(event.request)))
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("Realtime input observers must not return awaitables")
            except Exception:
                # Application exceptions can contain private input or secrets.
                logger.warning("Realtime input event callback failed")

    async def notify_silence(self, event: InputSilenceEvent):
        for handler in tuple(self.silence_handlers):
            if not event.is_current():
                return
            try:
                await handler(event)
            except Exception:
                # Application exceptions may include private input or secrets.
                logger.warning("Realtime input silence callback failed")
