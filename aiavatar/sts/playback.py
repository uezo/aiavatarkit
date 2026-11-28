"""Record what one connection delivered and played, as reported by its adapter."""

from dataclasses import dataclass, field
import logging
from sys import float_info
from time import monotonic
from typing import Optional

from .models import STSResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Interruption:
    """Interruption facts scoped to the conversation of the interrupted response."""

    interrupted_response_context_id: Optional[str]
    interrupted_response_transaction_id: str
    spoken_text: Optional[str] = None
    answered_through_transaction_id: Optional[str] = None


@dataclass
class _Response:
    context_id: Optional[str]
    audio_texts: list[str] = field(default_factory=list)
    discarded_audio_texts: list[str] = field(default_factory=list)
    answered: bool = False
    finished: bool = False


@dataclass
class _Playback:
    playback_id: str
    transaction_id: str
    text: str
    duration_seconds: float
    started_at: float
    preceding_text: str
    ended_at: Optional[float] = None
    completed: bool = False
    is_final_chunk: bool = False

    @property
    def natural_end(self) -> float:
        return self.started_at + self.duration_seconds

    def spoken_text(self, now: float) -> str:
        at = self.ended_at if self.ended_at is not None else now
        if self.completed or at >= self.natural_end:
            return self.preceding_text + self.text
        progress = max(0.0, at - self.started_at) / self.duration_seconds
        return self.preceding_text + self.text[:int(len(self.text) * progress)]


class PlaybackLedger:
    """Track delivery and playback of recent responses on one connection."""

    def __init__(self, *, debug: bool = False):
        self.debug = debug
        self.max_responses = 8
        self._responses: dict[str, _Response] = {}
        self._last_answered: Optional[tuple[Optional[str], str]] = None
        self._playback: Optional[_Playback] = None
        self._last_playback: Optional[_Playback] = None
        self._playback_ids: set[str] = set()

    def mark_answered(self, transaction_id: Optional[str]):
        if transaction_id and (response := self._responses.get(transaction_id)) and not response.answered:
            response.answered = True
            self._last_answered = (response.context_id, transaction_id)

    def observe(self, response: STSResponse):
        transaction_id = response.transaction_id
        if not transaction_id:
            return
        if response.type == "start":
            self._responses[transaction_id] = _Response(context_id=response.context_id)
            while len(self._responses) > max(0, self.max_responses):
                self._responses.pop(next(iter(self._responses)))
        elif previous := self._responses.get(transaction_id):
            if response.type == "chunk" and response.audio_data and response.voice_text:
                previous.audio_texts.append(response.voice_text)
            elif response.type in ("final", "error", "canceled", "cancelled"):
                previous.finished = True

    def handle_playback_event(
        self, /, *, event=None, playback_id=None, transaction_id=None,
        text=None, duration_seconds=None, completed=False, **_metadata,
    ):
        if event == "start":
            self._start_playback(playback_id, transaction_id, text, duration_seconds)
            return
        if event == "discarded":
            if isinstance(transaction_id, str) and isinstance(text, str):
                response = self._responses.get(transaction_id)
                if response is not None and text in response.audio_texts:
                    response.audio_texts.remove(text)
                    response.discarded_audio_texts.append(text)
                    if self.debug:
                        logger.info("Voice playback: event=discarded, transaction=%s, text=%r", transaction_id, text)
            return

        playback = self._playback
        if playback is None or playback.playback_id != playback_id:
            return
        if (event == "end" and playback.ended_at is None
                and (transaction_id is None or transaction_id == playback.transaction_id)):
            playback.ended_at = monotonic()
            playback.completed = completed is True
            if self.debug:
                logger.info("Voice playback: event=end, transaction=%s, playback_id=%s, completed=%s",
                            playback.transaction_id, playback_id, playback.completed)
        elif event == "final" and transaction_id == playback.transaction_id:
            playback.is_final_chunk = True

    def take_discarded_audio(self, context_id: str) -> list[str]:
        texts = []
        for response in self._responses.values():
            if response.context_id == context_id:
                texts.extend(response.discarded_audio_texts)
                response.discarded_audio_texts.clear()
        return texts

    def _start_playback(self, playback_id, transaction_id, text, duration_seconds):
        if (
            not isinstance(playback_id, str) or not playback_id.strip()
            or not isinstance(transaction_id, str) or not transaction_id.strip()
            or not isinstance(text, str) or not text.strip()
            or type(duration_seconds) not in (int, float) or not 0 < duration_seconds <= float_info.max
        ):
            return

        now = monotonic()
        previous = self._playback
        previous_text = previous.spoken_text(now) if previous is not None else ""
        same_transaction = previous is not None and previous.transaction_id == transaction_id
        if same_transaction and playback_id in self._playback_ids:
            return
        response = self._responses.get(transaction_id)
        if response is None:
            return
        try:
            chunk_index = response.audio_texts.index(text)
        except ValueError:
            return
        # The client may have skipped a queued clip. Only accumulate clips that actually played.
        del response.audio_texts[:chunk_index + 1]

        # Audio is serial. A valid new start bounds the previous clip when its end report is missing.
        if previous is not None:
            if previous.ended_at is None:
                previous.ended_at = now
            if previous_text != previous.preceding_text:
                self._last_playback = previous
        if not same_transaction:
            self._playback_ids.clear()
        self._playback_ids.add(playback_id)
        self._playback = _Playback(
            playback_id, transaction_id, text, duration_seconds, now, previous_text if same_transaction else "",
        )
        if self.debug:
            logger.info("Voice playback: event=start, transaction=%s, playback_id=%s", transaction_id, playback_id)

    def snapshot(self, active_transaction_id: Optional[str]) -> Optional[Interruption]:
        now = monotonic()
        active_response = self._responses.get(active_transaction_id) if active_transaction_id else None
        playback = self._playback
        if playback is None or playback.spoken_text(now) == playback.preceding_text:
            playback = self._last_playback
        if playback is not None and not (playback.is_final_chunk and playback.completed):
            response = self._responses.get(playback.transaction_id)
            if response is not None and (active_response is None or response.context_id == active_response.context_id):
                return Interruption(
                    interrupted_response_context_id=response.context_id,
                    interrupted_response_transaction_id=playback.transaction_id,
                    spoken_text=playback.spoken_text(now),
                )

        if active_transaction_id and active_response is not None:
            if active_response.answered or active_response.finished:
                return None
            answered_through = self._last_answered
            return Interruption(
                interrupted_response_context_id=active_response.context_id,
                interrupted_response_transaction_id=active_transaction_id,
                answered_through_transaction_id=(
                    answered_through[1]
                    if answered_through and answered_through[0] == active_response.context_id else None
                ),
            )
        return None
