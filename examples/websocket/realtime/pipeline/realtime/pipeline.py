"""Realtime session orchestration, task ownership, delivery, and recording."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
import os
import re
from typing import Awaitable, Callable
from uuid import uuid4

from aiavatar.sts.llm.context_manager import ContextManager
from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.performance_recorder import PerformanceRecorder
from ..streaming import StreamingSTSPipeline
from aiavatar.sts.tts import SpeechSynthesizer
from .connection import ProviderError, RealtimeConnection, make_session_config
from .history import HistoryLoadError, HistoryStore, HistoryWriter
from .metrics import TurnMetrics
from .input import InputCallbacks, InputAudio, InputState, InputSilenceEvent, RealtimeInputEvent, SilenceState
from .output import ResponseOutput, SpeechChunk, VoiceTextFilter, voice_text
from .turn import Turn

logger = logging.getLogger(__name__)
_ANY_TURN = object()


@dataclass(kw_only=True)
class _TurnRuntime(Turn):
    """Execution resources owned by the pipeline for a logical turn."""

    output: ResponseOutput
    queue: asyncio.Queue
    worker: asyncio.Task | None = None
    responses: asyncio.Queue | None = None
    metrics: TurnMetrics = field(default_factory=TurnMetrics)
    voice_timing_filter: VoiceTextFilter | None = None


@dataclass
class _Session:
    request: STSRequest
    ready: asyncio.Future
    audio: InputAudio
    restore_context: bool = False
    connection: RealtimeConnection | None = None
    task: asyncio.Task | None = None
    finalizer: asyncio.Task | None = None
    workers: set = field(default_factory=set)
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn: _TurnRuntime | None = None
    started: bool = False
    closing: bool = False
    input: InputState = field(default_factory=InputState)
    silence: SilenceState = field(default_factory=SilenceState)
    silence_task: asyncio.Task | None = None
    transcription_enabled: bool = False
    failure: str | None = None
    cleanup_started: bool = False
    history: HistoryWriter | None = None


class OpenAIRealtimePipeline(StreamingSTSPipeline):
    """Accept mono PCM16 and synthesize Realtime's text responses by sentence.

    Coordinates input admission, response turns, interruption, and client
    delivery. Connection protocol, history workers, and measurements have
    independent owners. Injected TTS, context managers, and recorders remain
    caller-owned.

    Input is resampled to 24 kHz; output uses the injected TTS rather than native
    Realtime audio. With no injected context manager, ``context_db_path=None``
    disables local history storage and restoration.
    """

    def __init__(
        self,
        *,
        tts: SpeechSynthesizer,
        openai_api_key: str = None,
        model: str = "gpt-realtime-2.1",
        instructions: str = None,
        input_sample_rate: int = 16000,
        turn_detection: dict = None,
        transcription_model: str = None,
        language: str = None,
        sentence_end: str = "。！？!?\n",
        voice_text_tags: str | list[str] | None = None,
        tts_queue_size: int = 16,
        startup_timeout: float = 15.0,
        close_timeout: float = 5.0,
        performance_recorder: PerformanceRecorder = None,
        context_manager: ContextManager = None,
        context_db_path: str | None = "aiavatar.db",
        context_transcription_timeout: float = 10.0,
        restore_context: bool = True,
        context_history_limit: int = 100,
        input_silence_duration: float | None = None,
        input_silence_threshold: float = 0.02,
        input_silence_min_speech_duration: float = 0.2,
    ):
        super().__init__()
        if tts is None:
            raise ValueError("tts is required; native Realtime audio is not used")
        if type(input_sample_rate) is not int or not 8000 <= input_sample_rate <= 192000:
            raise ValueError("input_sample_rate must be an integer from 8000 to 192000 (mono PCM16)")
        if not sentence_end:
            raise ValueError("sentence_end must not be empty")
        voice_text_tags = [voice_text_tags] if isinstance(voice_text_tags, str) else list(voice_text_tags or [])
        if any(not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z_]\w*", tag) for tag in voice_text_tags):
            raise ValueError("voice_text_tags must contain XML tag names, without angle brackets")
        if type(tts_queue_size) is not int or tts_queue_size < 1:
            raise ValueError("tts_queue_size must be a positive integer")
        if startup_timeout <= 0 or close_timeout <= 0:
            raise ValueError("Connection timeouts must be positive")
        if not 0 < context_transcription_timeout < float("inf"):
            raise ValueError("context_transcription_timeout must be finite and positive")
        if type(restore_context) is not bool:
            raise ValueError("restore_context must be a boolean")
        if type(context_history_limit) is not int or context_history_limit < 1:
            raise ValueError("context_history_limit must be a positive integer")
        for name, value, lower, upper, allow_zero in (
            ("input_silence_duration", input_silence_duration, 0, math.inf, False),
            ("input_silence_threshold", input_silence_threshold, 0, 1, False),
            ("input_silence_min_speech_duration", input_silence_min_speech_duration, 0, math.inf, True),
        ):
            if name == "input_silence_duration" and value is None:
                continue
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or value >= upper or value < lower or (not allow_zero and value == lower)):
                raise ValueError(f"{name} is outside its supported range")
        self.turn_detection = deepcopy(turn_detection) if turn_detection is not None else {"type": "server_vad"}
        if self.turn_detection.get("type") not in ("server_vad", "semantic_vad"):
            raise ValueError("turn_detection must use server_vad or semantic_vad")
        self.turn_detection.update(create_response=False, interrupt_response=False)
        self.tts = tts
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.instructions = instructions
        self.input_sample_rate = input_sample_rate
        self.vad.sample_rate = input_sample_rate
        self.transcription_model = transcription_model
        self.language = language
        self.sentence_end = sentence_end
        self.voice_text_tags = voice_text_tags
        self.tts_queue_size = tts_queue_size
        self.startup_timeout = startup_timeout
        self.close_timeout = close_timeout
        self.performance_recorder = performance_recorder
        self._history_store = HistoryStore(context_manager, context_transcription_timeout)
        self.context_db_path = context_db_path
        self.restore_context = restore_context
        self.context_history_limit = context_history_limit
        self.input_silence_duration = input_silence_duration
        self.input_silence_threshold = input_silence_threshold
        self.input_silence_min_speech_duration = input_silence_min_speech_duration
        self._input_callbacks = InputCallbacks()
        self._sessions: dict[str, _Session] = {}

    @property
    def context_manager(self):
        return self._history_store.manager

    @context_manager.setter
    def context_manager(self, manager):
        self._history_store.manager = manager

    @property
    def context_transcription_timeout(self):
        return self._history_store.transcription_timeout

    @context_transcription_timeout.setter
    def context_transcription_timeout(self, timeout):
        self._history_store.transcription_timeout = timeout

    def on_input_silence(self, handler: Callable[[InputSilenceEvent], Awaitable[None]]):
        """Register an async callback for local silence before Realtime commits."""
        return self._input_callbacks.register_silence(handler)

    def on_input_event(self, handler: Callable[[RealtimeInputEvent], None]):
        """Register a fast synchronous input observer; asynchronous work is caller-owned."""
        return self._input_callbacks.register_event(handler)

    def _admit_input_event(self, session: _Session):
        for event in session.input.admit(session.request):
            self._input_callbacks.notify(event)

    def _transcript_input_event(self, session: _Session, item_id: str, text: str, *, completed: bool = False):
        self._input_callbacks.notify(session.input.transcript_event(item_id, text, completed=completed))

    def _end_input_event(self, session: _Session):
        self._input_callbacks.notify(session.input.end())

    def _cancel_input_silence(self, session: _Session, *, end_input: bool = False, reset_audio: bool = True):
        session.silence.invalidate(end_input=end_input, reset_audio=reset_audio)
        if session.silence_task is not None and session.silence_task is not asyncio.current_task():
            session.silence_task.cancel()
        if end_input:
            self._end_input_event(session)

    def _input_silence_current(self, session: _Session, generation: int) -> bool:
        state = session.silence
        gate = self._prepared_sessions.get(session.request.session_id)
        return (
            self.input_silence_duration is not None
            and self._sessions.get(session.request.session_id) is session
            and session.started and not session.closing
            and (gate is None or gate.is_set())
            and state.active and state.generation == generation
            and bool(session.input.item_id) and not session.input.ignored
            and not self._blocks_input(session)
            and (session.turn is None or session.turn.finished)
        )

    def _trigger_input_silence(self, session: _Session):
        state = session.silence
        if (not self._input_silence_current(session, state.generation)
                or not state.is_pause(session.audio.sample_rate, self.input_silence_duration,
                                      self.input_silence_min_speech_duration)):
            return
        # Consume this pause even when a previous callback is still running or
        # draining cancellation. Finishing that task must not replay this pause.
        state.fired = True
        if session.silence_task is not None and not session.silence_task.done():
            return
        generation = state.generation
        event = InputSilenceEvent(
            session_id=session.request.session_id, input_item_id=session.input.item_id,
            silence_duration=state.silent_samples / session.audio.sample_rate,
            request=deepcopy(session.request),
            _is_current=lambda: (session.silence_task is task and not task.done()
                                 and self._input_silence_current(session, generation)),
            input_text=session.input.caption or "",
            _get_input_text=lambda: session.input.caption or "",
            _get_silence_duration=lambda: state.silent_samples / session.audio.sample_rate,
            detected_at=asyncio.get_running_loop().time(),
        )
        task = session.silence_task = asyncio.create_task(self._input_callbacks.notify_silence(event))
        session.workers.add(task)

        def finished(completed):
            session.workers.discard(completed)
            if session.silence_task is completed:
                session.silence_task = None

        task.add_done_callback(finished)

    def _observe_input_silence(self, session: _Session, audio: bytes):
        if self.input_silence_duration is None or not self._input_callbacks.silence_handlers:
            return
        # Evaluate each observed pause before later frames can resume voice.
        for _ in session.silence.observe(audio, session.audio.sample_rate,
                                        self.input_silence_threshold, self.input_silence_duration):
            self._trigger_input_silence(session)

    def _begin_context(self, session: _Session, turn: _TurnRuntime, *, text_input: bool):
        if session.history is None:
            return
        context = session.history.turns.begin(
            text_input=text_input, request_text=turn.request.text, input_item_id=turn.input_item_id,
            transcription_enabled=session.transcription_enabled, current_input_item_id=session.input.item_id,
            transcription_completed=session.input.transcription_completed, input_caption=session.input.caption,
        )
        if session.history.submit(context):
            turn.context = context

    @staticmethod
    def _complete_context(turn: _TurnRuntime | None, *, assistant_text: str | None = None):
        if turn is not None and turn.context is not None:
            turn.context.complete_response(asyncio.get_running_loop().time(), assistant_text)

    def _record_performance(self, turn: _TurnRuntime | None, status: str, *, reason: str = None, provider_status: str = None):
        if turn is not None:
            turn.metrics.finish(
                self.performance_recorder, turn.request, turn.transaction_id, turn.output.text,
                status, reason=reason, provider_status=provider_status,
            )

    async def start_session(self, request: STSRequest):
        if not self.openai_api_key:
            raise ValueError("openai_api_key or OPENAI_API_KEY is required")
        if not request.session_id:
            raise ValueError("session_id is required")
        if request.session_id in self._sessions:
            raise ValueError("Realtime session is already active")
        rate = (request.metadata or {}).get("input_sample_rate", self.input_sample_rate)
        if type(rate) is not int or not 8000 <= rate <= 192000:
            raise ValueError("Microphone input_sample_rate must be an integer from 8000 to 192000")
        restore_context = self.restore_context and bool(request.context_id)
        if not request.context_id:
            request.context_id = str(uuid4())
        session = _Session(
            request=deepcopy(request), ready=asyncio.get_running_loop().create_future(),
            audio=InputAudio(rate),
            restore_context=restore_context,
        )
        self._remember_session(request)
        self._sessions[request.session_id] = session
        session.task = asyncio.create_task(self._run_session(session))
        try:
            await asyncio.wait_for(asyncio.shield(session.ready), self.startup_timeout)
        except BaseException:
            session.ready.cancel()
            await self.finalize(request.session_id)
            raise

    async def configure_session(self, session_id: str, metadata: dict):
        session = self._sessions.get(session_id)
        if session is None or session.closing:
            raise RuntimeError("Realtime session is not ready")
        if "barge_in_enabled" in metadata:
            self.vad.set_session_data(session_id, "barge_in_enabled", bool(metadata["barge_in_enabled"]))

    async def invoke(self, request: STSRequest):
        """Invoke text on an active session, yielding its response exactly once."""
        session = self._sessions.get(request.session_id)
        gate = self._prepared_sessions.get(request.session_id)
        if session is None or session.closing or not session.started or (gate is not None and not gate.is_set()):
            error = "Realtime invoke requires a started and activated session"
        elif not isinstance(request.text, str) or not request.text.strip():
            error = "Realtime invoke requires non-empty text"
        elif request.audio_data or request.files:
            error = "Realtime invoke supports text only; audio_data and files are not supported"
        elif request.wait_in_queue:
            error = "Realtime invoke does not support wait_in_queue"
        elif request.system_prompt_params is not None:
            error = "Realtime invoke does not support per-request system_prompt_params"
        elif (request.quick_response_text or request.quick_response_voice_text
              or request.quick_response_audio or request.skip_quick_response):
            error = "Realtime invoke does not support quick-response options"
        elif request.context_id is not None and request.context_id != session.request.context_id:
            error = "Realtime invoke cannot switch the active session's context_id"
        else:
            error = None
        if error:
            # No new transaction is accepted for invalid input; preserve any
            # response already playing at an adapter sharing this session.
            yield STSResponse(
                type="error", session_id=request.session_id, user_id=request.user_id,
                context_id=request.context_id, voice_text=error, metadata={"error": error},
            )
            return

        for key in ("user_id", "context_id", "channel"):
            if getattr(request, key) is None:
                setattr(request, key, getattr(session.request, key))
        turn = _TurnRuntime(
            transaction_id=request.transaction_id,
            queue=asyncio.Queue(maxsize=self.tts_queue_size + 1), request=request,
            output=ResponseOutput(),
            responses=asyncio.Queue(maxsize=self.tts_queue_size + 2),
        )
        try:
            try:
                await self._begin_response(session, turn, text_input=True)
            except Exception:
                await self._report_error(session, "Realtime text request failed", turn=turn)
            while True:
                response = await turn.responses.get()
                yield response
                if response.type in ("final", "canceled", "error"):
                    return
        finally:
            if session.turn is turn and not (turn.completed and turn.finished):
                try:
                    await self._interrupt(session, expected=turn)
                except Exception:
                    logger.warning("Realtime turn cancellation failed")
            while not turn.responses.empty():
                turn.responses.get_nowait()

    async def _emit(self, session: _Session, type: str, *, turn: _TurnRuntime = None, metadata: dict = None, **kwargs):
        if type in ("final", "canceled", "error"):
            self._complete_context(turn)
            self._record_performance(
                turn, type, reason=(metadata or {}).get("error") if type == "error" else
                ("interrupted" if type == "canceled" else None),
            )
        request = turn.request if turn is not None else session.request
        response = STSResponse(
            type=type,
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=session.request.context_id,
            transaction_id=turn.transaction_id if turn else None,
            metadata={**({"channel": request.channel} if request.channel else {}), **(metadata or {})},
            **kwargs,
        )
        if turn is not None and turn.responses is not None and type != "accepted":
            if turn.terminal:
                return
            if type in ("final", "canceled", "error"):
                turn.terminal = True
                if type != "final":
                    while not turn.responses.empty():
                        turn.responses.get_nowait()
            elif turn.responses.qsize() >= turn.responses.maxsize - 1:
                raise asyncio.QueueFull("Realtime invoke response queue is full")
            turn.responses.put_nowait(response)
            return
        await self.handle_response(response)

    async def _report_error(self, session: _Session, message: str, *, turn: _TurnRuntime = None, **metadata):
        self._complete_context(turn or session.turn)
        self._record_performance(turn or session.turn, "error", reason=message)
        if turn is None and session.turn is not None and session.turn.responses is not None:
            turn = session.turn
        try:
            await self._emit(session, "error", turn=turn, voice_text=message, metadata={"error": message, **metadata})
        except Exception:
            logger.warning("Realtime response handler failed while reporting an error")

    async def _mark_session_ready(self, session: _Session):
        if session.closing or session.started:
            return
        session.started = True
        logger.info("Realtime audio input: session=%s, %d Hz -> 24000 Hz",
                    session.request.session_id, session.audio.sample_rate)
        await self._session_connected(STSResponse(
            type="connected", session_id=session.request.session_id,
            user_id=session.request.user_id, context_id=session.request.context_id,
            metadata={**({"channel": session.request.channel} if session.request.channel else {}),
                      "input_sample_rate": session.audio.sample_rate},
        ), session.ready)

    async def _run_session(self, session: _Session):
        try:
            history_messages = []
            if self.context_manager is not None or self.context_db_path is not None:
                await self._history_store.ensure_manager(self.context_db_path)
                try:
                    history_messages = (await self._history_store.load_messages(
                        session.request.context_id, self.context_history_limit,
                    )) if session.restore_context else []
                except HistoryLoadError:
                    session.failure = "Realtime context history loading failed"
                    raise
                session.history = HistoryWriter(self._history_store, session.request.context_id)
                session.history.start()
            async with RealtimeConnection(
                model=self.model, api_key=self.openai_api_key,
                startup_timeout=self.startup_timeout, close_timeout=self.close_timeout,
            ) as connection:
                session.connection = connection
                config = make_session_config(
                    turn_detection=self.turn_detection, transcription_model=self.transcription_model,
                    language=self.language, instructions=self.instructions,
                )
                session.transcription_enabled = bool(config["audio"]["input"].get("transcription"))
                try:
                    if await connection.configure(config, history_messages):
                        await self._mark_session_ready(session)
                        async for event in connection:
                            if not session.closing:
                                await self._receive_event(session, event)
                except ProviderError as exc:
                    if not session.closing:
                        session.failure = "OpenAI Realtime error"
                        await self._report_error(session, session.failure, code=exc.code)
                if not session.closing:
                    session.failure = session.failure or "Realtime connection closed"
        except asyncio.CancelledError:
            raise
        except Exception:
            # Transport/provider exceptions can contain headers or echoed input.
            session.failure = session.failure or "Realtime connection failed"
            if session.started:
                await self._report_error(session, session.failure)
        finally:
            session.cleanup_started = True
            session.closing = True
            self._cancel_input_silence(session, end_input=True)
            self._complete_context(session.turn)
            self._record_performance(
                session.turn, "error" if session.failure else "canceled",
                reason=session.failure or "session_closed",
            )
            if session.turn is not None and session.turn.responses is not None and not session.turn.terminal:
                await self._report_error(session, session.failure or "Realtime session closed", turn=session.turn)
            session.turn = None
            workers = list(session.workers)
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if not session.ready.done():
                session.ready.set_exception(RuntimeError(session.failure or "Realtime startup was interrupted"))
            if session.started:
                try:
                    await self.stop_response(session.request.session_id, session.request.context_id)
                except Exception:
                    logger.warning("Realtime response handler failed while stopping playback")
                try:
                    await self._emit(session, "session_closed", metadata={"reason": session.failure or "connection_closed"})
                except Exception:
                    logger.warning("Realtime response handler failed while closing the session")
            if session.history is not None:
                await session.history.close(self.close_timeout)
            if self._sessions.get(session.request.session_id) is session:
                del self._sessions[session.request.session_id]
                self._release_session(session.request.session_id)

    async def _replace_turn(self, session: _Session, next_turn: _TurnRuntime | None, *, expected=_ANY_TURN):
        previous = None
        try:
            async with session.turn_lock:
                if expected is not _ANY_TURN and session.turn is not expected:
                    return False
                if next_turn is not None and session.closing:
                    return False
                previous, session.turn = session.turn, next_turn
                if previous and previous.worker:
                    previous.worker.cancel()
                if previous and previous.response_requested and not previous.completed:
                    cancel = {"type": "response.cancel"}
                    if previous.response_id:
                        cancel["response_id"] = previous.response_id
                    # Serialize cancellation with the next response.create,
                    # including responses whose provider ID has not arrived yet.
                    await session.connection.send(cancel)
            if (session.turn is next_turn and not session.closing
                    and (previous is not None or next_turn is None)):
                await self.stop_response(session.request.session_id, session.request.context_id)
        finally:
            self._complete_context(previous)
            if previous and not previous.finished:
                await self._emit(session, "canceled", turn=previous, metadata={"interrupted": True})
        return session.turn is next_turn and not session.closing

    async def _interrupt(self, session: _Session, *, expected=_ANY_TURN):
        await self._replace_turn(session, None, expected=expected)

    async def _begin_response(self, session: _Session, turn: _TurnRuntime, *, text_input: bool = False):
        request = turn.request
        self._cancel_input_silence(session, end_input=True)
        if self.performance_recorder is not None:
            turn.metrics.begin(
                request, session.request.context_id,
                request.text if text_input else (
                    session.input.caption if turn.input_item_id is not None
                    and turn.input_item_id == session.input.item_id else None
                ), self.__class__.__name__, self.tts.__class__.__name__,
            )
            turn.voice_timing_filter = VoiceTextFilter(self.voice_text_tags)
        try:
            current = await self._replace_turn(session, turn)
        except asyncio.CancelledError:
            self._record_performance(turn, "canceled", reason="request_canceled")
            raise
        except Exception:
            self._record_performance(turn, "error", reason="Realtime response setup failed")
            raise
        turn.metrics.lap("stop_response_time")
        if not current:
            self._record_performance(turn, "canceled", reason="interrupted")
            if turn.responses is not None and not turn.terminal:
                await self._emit(session, "canceled", turn=turn, metadata={"interrupted": True})
            return
        # Hooks may finalize the session or start another invocation. Never hold
        # turn_lock across application callbacks, and recheck ownership after them.
        try:
            await self._execute_on_accepted(request)
        finally:
            # One upstream socket owns one conversation. Hooks may rewrite text
            # and transaction metadata, but cannot redirect its archive/identity.
            request.context_id = session.request.context_id
            if not session.closing and session.turn is turn:
                turn.transaction_id = request.transaction_id
                await self._emit(session, "accepted", turn=turn,
                                 metadata={"block_barge_in": request.block_barge_in})
        if session.closing or session.turn is not turn:
            return
        async with session.turn_lock:
            if session.closing or session.turn is not turn:
                return
            if text_input:
                await session.connection.send({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": request.text}]},
                })
                if session.closing:
                    return
            self._begin_context(session, turn, text_input=text_input)
            turn.response_requested = True
            turn.metrics.lap("before_llm_time")
            await session.connection.send({
                "type": "response.create",
                "response": {"output_modalities": ["text"], "metadata": {"transaction_id": turn.transaction_id}},
            })

    def _start_input_item(self, session: _Session, item_id: str | None):
        if item_id and item_id != session.input.item_id:
            # Retain local evidence captured before a delayed speech_started
            # when no older input item is still open.
            self._cancel_input_silence(session, end_input=True, reset_audio=session.silence.active)
            session.silence.reset_item()
            session.input.start(item_id)

    def _blocks_input(self, session: _Session) -> bool:
        # Protect server-side generation and synthesis gaps until final delivery.
        barge_in_enabled = self.vad.get_session_data(session.request.session_id, "barge_in_enabled")
        if barge_in_enabled is None:
            barge_in_enabled = True
        return session.turn is not None and session.turn.blocks_input(barge_in_enabled)

    async def _receive_event(self, session: _Session, event: dict):
        event_type = event["type"]
        if (event_type in ("input_audio_buffer.speech_started", "input_audio_buffer.committed")
                and session.input.is_ended(event.get("item_id"))):
            # A duplicate boundary for an ended item must not reopen it or end
            # a newer admitted input. Late final ASR remains eligible below.
            return
        if event_type == "input_audio_buffer.speech_started":
            self._start_input_item(session, event.get("item_id"))
            if session.input.ignored or self._blocks_input(session):
                session.input.ignored = True
                self._cancel_input_silence(session, end_input=True)
                return
            self._admit_input_event(session)
            state = session.silence
            arm_silence = not state.started
            state.started = True
            input_generation = state.input_generation
            previous = session.turn
            await self.vad._notify_speech_started(session.request.session_id)
            if session.closing:
                return
            await self._interrupt(session, expected=previous)
            if (arm_silence and state.input_generation == input_generation
                    and not session.closing and session.turn is None):
                state.active = True
                # A delayed speech_started may arrive during already-observed
                # silence. No new PCM packet is required to evaluate that pause.
                self._observe_input_silence(session, b"")
        elif event_type == "input_audio_buffer.committed":
            self._start_input_item(session, event.get("item_id"))
            self._cancel_input_silence(session, end_input=True)
            session.silence.started = True
            if session.input.ignored or self._blocks_input(session):
                session.input.ignored = True
                # Remove the rejected user input before any later response can
                # use it. Keep its caption suppressed until a new item starts.
                if event.get("item_id"):
                    await session.connection.send({"type": "conversation.item.delete", "item_id": event["item_id"]})
                return
            # Commit can be the first observation when speech_started is absent.
            self._admit_input_event(session)
            self._end_input_event(session)
            # Reserve one extra slot for completion, keeping the reader free to
            # process interruptions even when synthesis is slower than text.
            request = STSRequest(
                session_id=session.request.session_id, user_id=session.request.user_id,
                context_id=session.request.context_id, transaction_id=str(uuid4()),
                channel=session.request.channel, metadata=deepcopy(session.request.metadata),
            )
            turn = _TurnRuntime(
                request=request, transaction_id=request.transaction_id, input_item_id=event.get("item_id"),
                queue=asyncio.Queue(maxsize=self.tts_queue_size + 1),
                output=ResponseOutput(),
            )
            await self._begin_response(session, turn)
        elif event_type == "response.created":
            response = event["response"]
            turn = session.turn
            if turn is None or (response.get("metadata") or {}).get("transaction_id") != turn.transaction_id:
                await session.connection.send({"type": "response.cancel", "response_id": response["id"]})
                return
            if turn.response_id is not None:
                return
            turn.response_id = response["id"]
            metadata = ({"request_text": turn.request.text, "recognized_text": turn.request.text}
                        if turn.request.text is not None else None)
            await self._emit(session, "start", turn=turn, metadata=metadata)
            if session.closing or session.turn is not turn:
                return
            turn.worker = asyncio.create_task(self._synthesize(session, turn))
            session.workers.add(turn.worker)
            turn.worker.add_done_callback(session.workers.discard)
        elif event_type in (
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
        ):
            item_id = event.get("item_id")
            if event_type.endswith(".completed"):
                self._transcript_input_event(session, item_id, event.get("transcript", ""), completed=True)
            elif event_type.endswith(".failed"):
                self._transcript_input_event(session, item_id, "", completed=True)
            # Apply late final ASR before the current-caption filter. History
            # outlives playback and can belong to an earlier turn.
            if session.history is not None and event_type.endswith(".completed"):
                session.history.turns.capture_transcript(item_id, event.get("transcript", ""))
            elif session.history is not None and event_type.endswith(".failed"):
                session.history.turns.capture_transcript(item_id, failed=True)
            if session.input.item_id is None:
                self._start_input_item(session, item_id)
            completed = event_type.endswith(".completed")
            failed = event_type.endswith(".failed")
            if not session.input.update_transcript(
                item_id, event.get("transcript", "") if completed else event.get("delta", ""),
                completed=completed, failed=failed,
            ) or failed:
                return
            if not completed:
                self._transcript_input_event(session, item_id, session.input.transcript)
            performance = (session.turn.metrics.active if session.turn is not None else None)
            if (performance is not None and item_id is not None
                    and session.turn.input_item_id == item_id):
                performance.request_text = session.input.caption
            if session.input.caption.strip():
                await self._emit(session, "info", metadata={
                    "partial_request_text": session.input.caption, "item_id": item_id,
                })
        elif event_type in ("response.output_text.delta", "response.done"):
            turn = session.turn
            response_id = event.get("response_id") if event_type.endswith("delta") else event["response"].get("id")
            if turn is None or not turn.matches_response(response_id):
                return
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    turn.metrics.lap("llm_first_chunk_time", first=True)
                self._append_text(session, turn, delta)
                return
            turn.metrics.lap("llm_time")
            turn.completed = True
            if event["response"].get("status") == "completed":
                if turn.context is not None:
                    output = event["response"].get("output")
                    assistant_text = "".join(
                        part.get("text", "") for item in output or []
                        if item.get("type") == "message" and item.get("role") == "assistant"
                        for part in item.get("content", []) if part.get("type") in ("text", "output_text")
                    ) if output is not None else turn.output.text
                    self._complete_context(turn, assistant_text=assistant_text)
                for remaining in turn.output.flush():
                    self._queue_text(session, turn, remaining)
                if (turn.metrics.active is not None and self.voice_text_tags
                        and not turn.voice_timing_filter.seen_tag and voice_text(turn.output.text)):
                    turn.metrics.lap("llm_first_voice_chunk_time", first=True)
                turn.queue.put_nowait(None)
            else:
                failed = event["response"].get("status") != "cancelled"
                self._record_performance(
                    turn, "error" if failed else "canceled", reason="Realtime response did not complete",
                    provider_status=event["response"].get("status", "unknown"),
                )
                if failed and turn.responses is not None:
                    await self._report_error(session, "Realtime response did not complete", turn=turn)
                await self._interrupt(session, expected=turn)
                if failed and turn.responses is None:
                    await self._report_error(session, "Realtime response did not complete", turn=turn)

    def _queue_text(self, session: _Session, turn: _TurnRuntime, text: str):
        if text.strip():
            if (turn.metrics.active is not None
                    and turn.voice_timing_filter.feed(text)):
                turn.metrics.lap("llm_first_voice_chunk_time", first=True)
            if turn.queue.qsize() >= self.tts_queue_size:
                session.failure = "Speech synthesis queue is full"
                raise RuntimeError(session.failure)
            turn.queue.put_nowait(text)

    def _append_text(self, session: _Session, turn: _TurnRuntime, text: str):
        try:
            for segment in turn.output.append(text, self.sentence_end):
                self._queue_text(session, turn, segment)
        except RuntimeError:
            session.failure = session.failure or "Realtime response text exceeds the buffering limit"
            raise

    async def _synthesize(self, session: _Session, turn: _TurnRuntime):
        voice_filter = VoiceTextFilter(self.voice_text_tags)
        try:
            while session.turn is turn and not session.closing:
                text = await turn.queue.get()
                try:
                    if text is None:
                        final_voice_text, fallback = turn.output.finish(self.voice_text_tags, voice_filter)
                        if fallback is not None:
                            await self._synthesize_chunk(session, turn, fallback)
                            if session.turn is not turn or session.closing:
                                return
                        performance = turn.metrics.active
                        if performance is not None:
                            performance.response_voice_text = final_voice_text
                        turn.finished = True
                        await self._emit(session, "final", turn=turn, text=turn.output.text, voice_text=final_voice_text)
                        return
                    await self._synthesize_chunk(session, turn, turn.output.prepare(text, voice_filter))
                finally:
                    turn.queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if session.turn is turn and not session.closing:
                session.failure = ("Realtime invoke response queue is full"
                                   if isinstance(exc, asyncio.QueueFull) else "Speech synthesis failed")
                await self._report_error(session, session.failure, turn=turn)
                await session.connection.close()

    async def _synthesize_chunk(self, session: _Session, turn: _TurnRuntime, chunk: SpeechChunk):
        performance = turn.metrics.active
        if performance is not None and chunk.voice_text:
            performance.response_voice_text = (performance.response_voice_text or "") + chunk.voice_text
        chunk = await turn.output.synthesize(chunk, self.tts, self.language)
        if session.turn is not turn or session.closing:
            return
        if chunk.audio_data:
            turn.metrics.lap("tts_first_chunk_time", first=True)
            turn.metrics.lap("tts_time")
        await self._emit(session, "chunk", turn=turn, text=chunk.text, voice_text=chunk.voice_text,
                         audio_data=chunk.audio_data, language=self.language)

    async def process_audio_samples(self, samples: bytes, context_id: str):
        """Send PCM, carrying partial samples and resampler state per session."""
        session = self._sessions.get(context_id)
        if session is None or session.closing or not session.started:
            raise RuntimeError("Realtime session is not ready")
        try:
            async with session.connection.writer() as writer:
                if session.closing:
                    return
                audio = session.audio.complete_samples(samples)
                if audio:
                    self._observe_input_silence(session, audio)
                if audio:
                    audio = session.audio.resample(audio)
                if audio:
                    await writer.send_audio(audio)
        except Exception:
            session.failure = "Realtime audio send failed"
            await self.finalize(context_id)
            raise RuntimeError(session.failure) from None

    async def finalize(self, session_id: str):
        session = self._sessions.get(session_id)
        if session is None:
            self._release_session(session_id)
            return
        if session.finalizer is None:
            session.closing = True
            self._cancel_input_silence(session, end_input=True)
            session.finalizer = asyncio.create_task(self._finalize_session(session))
        if (asyncio.current_task() is not session.task and asyncio.current_task() not in session.workers
                and (session.history is None or asyncio.current_task() is not session.history.worker)):
            await asyncio.shield(session.finalizer)

    async def _finalize_session(self, session: _Session):
        self._complete_context(session.turn)
        self._record_performance(
            session.turn, "error" if session.failure else "canceled",
            reason=session.failure or "session_finalized",
        )
        if session.turn is not None and session.turn.responses is not None and not session.turn.terminal:
            await self._report_error(session, session.failure or "Realtime session closed", turn=session.turn)
        session.turn = None
        for worker in list(session.workers):
            worker.cancel()
        # Natural disconnect may already be draining history. Do not cancel
        # that cleanup when an adapter concurrently calls finalize().
        if not session.cleanup_started:
            session.task.cancel()
        await asyncio.gather(session.task, return_exceptions=True)
        if not session.cleanup_started:
            # Cancellation before _run_session's first step never executes its
            # finally block. Release the reservation even if startup never ran.
            session.cleanup_started = True
            if session.history is not None:
                await session.history.close(self.close_timeout)
            if not session.ready.done():
                session.ready.set_exception(RuntimeError("Realtime startup was interrupted"))
            if self._sessions.get(session.request.session_id) is session:
                del self._sessions[session.request.session_id]
                self._release_session(session.request.session_id)

    async def shutdown(self):
        try:
            await asyncio.gather(*(self.finalize(session_id) for session_id in list(self._sessions)))
        finally:
            await self._history_store.join_initialization()
