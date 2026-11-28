"""Resume an intentionally deferred voice answer after the user falls silent."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
import logging
from math import isfinite
import re
import wave

from .models import STSRequest, STSResponse
from .pipeline import STSPipeline

logger = logging.getLogger(__name__)

DEFAULT_RESUME_TEXT = (
    "$The user has paused. Leave <ack> empty and always include a short spoken reply in <answer>. "
    "Use any playback context to judge what the user heard. Answer outstanding questions or requests, "
    "taking any additions or corrections into account. If the intent is unclear, ask one clarifying question. "
    "If the conversation is complete, close with a brief natural response."
)

DEFAULT_EMPTY_ANSWER_PATTERN = re.compile(
    r"\s*<ack>[^<>]*</ack>\s*<think>.*?</think>\s*<answer>\s*</answer>\s*", re.DOTALL,
)
_RESUME_KEY = "deferred_answer"


@dataclass
class _Reply:
    request: STSRequest
    detected_at: float
    can_wait: bool = True
    ack_ends_at: float = 0.0


@dataclass(frozen=True)
class _Resume:
    source_transaction_id: str
    last_voice_at: float


@dataclass
class _Session:
    last_voice_at: float = 0.0
    reply: _Reply | None = None
    wait_task: asyncio.Task | None = None
    tasks: set[asyncio.Task] = field(default_factory=set)

    def cancel_wait(self) -> None:
        if self.wait_task is not None:
            self.wait_task.cancel()
            self.wait_task = None


class DeferredAnswerScheduler:
    """Schedule one follow-up for an empty tagged answer to a VAD turn."""

    def __init__(
        self, *, resume_text: str = DEFAULT_RESUME_TEXT, silence_seconds: float = 0.8,
        empty_answer_pattern: str | re.Pattern[str] = DEFAULT_EMPTY_ANSWER_PATTERN,
        is_enabled: Callable[[str], bool] | None = None,
    ):
        if not resume_text.strip():
            raise ValueError("resume_text must not be empty")
        if not isfinite(silence_seconds) or silence_seconds < 0:
            raise ValueError("silence_seconds must be finite and nonnegative")
        self.sts: STSPipeline | None = None
        self.resume_text = resume_text
        self.silence_seconds = silence_seconds
        self.empty_answer_pattern = re.compile(empty_answer_pattern)
        self._is_enabled = is_enabled or (lambda _session_id: True)
        self._sessions: dict[str, _Session] = {}
        self._closed = False

    def bind(self, sts: STSPipeline) -> None:
        """Register the scheduler's hooks on the pipeline. A scheduler binds to one pipeline, once."""
        if self.sts is not None:
            raise RuntimeError("DeferredAnswerScheduler is already bound to a pipeline")
        self.sts = sts
        sts.vad.on_voiced(self._remember_voice)
        sts.can_accept(self._accept_request)
        sts.on_before_llm(self._validate_resume)
        sts.on_response(self._observe_response)
        sts.on_finalize(self.close_session)
        sts.on_shutdown(self.close)

    @staticmethod
    def is_resume(request: STSRequest) -> bool:
        return isinstance((request.metadata or {}).get(_RESUME_KEY), _Resume)

    async def _remember_voice(self, session_id: str) -> None:
        if not self._closed and self._is_enabled(session_id):
            session = self._sessions.get(session_id)
            if session is None:
                session = self._sessions[session_id] = _Session()
            session.last_voice_at = asyncio.get_running_loop().time()

    def _accept_request(self, request: STSRequest) -> bool:
        if self.is_resume(request):
            return self._can_resume(request)
        session = self._sessions.get(request.session_id)
        if session is not None:
            session.cancel_wait()
            session.reply = None
        if (not self._closed and request.type == "start" and request.audio_duration > 0
                and self._is_enabled(request.session_id)):
            if session is None:
                session = self._sessions[request.session_id] = _Session()
            session.reply = _Reply(request, asyncio.get_running_loop().time())
        return True

    def _is_current(self, session: _Session, reply: _Reply) -> bool:
        request = reply.request
        return (
            self._sessions.get(request.session_id) is session
            and session.reply is reply
            and self._is_enabled(request.session_id)
            and self.sts.vad.get_session_data(request.session_id, "context_id") == request.context_id
        )

    def _is_recording(self, session_id: str) -> bool:
        recording = getattr(self.sts.vad, "recording_sessions", {}).get(session_id)
        return recording is not None and recording.is_recording

    def _can_resume(self, request: STSRequest) -> bool:
        resume = request.metadata[_RESUME_KEY]
        session = self._sessions.get(request.session_id)
        return (
            session is not None and session.reply is not None
            and session.reply.request.transaction_id == resume.source_transaction_id
            and session.reply.request.context_id == request.context_id
            and self._is_current(session, session.reply)
            and session.last_voice_at == resume.last_voice_at
            and not self._is_recording(request.session_id)
        )

    async def _validate_resume(self, request: STSRequest) -> None:
        if self.is_resume(request) and not self._can_resume(request):
            request.text = None

    async def _observe_response(self, request: STSRequest, response: STSResponse) -> None:
        session = self._sessions.get(request.session_id)
        reply = session.reply if session is not None else None
        if reply is None or reply.request is not request or not reply.can_wait:
            return
        if (response.type in ("tool_call", "canceled", "cancelled", "error")
                or (response.metadata or {}).get("is_guardrail_triggered")):
            session.reply = None
        elif response.type == "chunk" and response.voice_text_tag == "ack" and response.audio_data:
            try:
                with wave.open(BytesIO(response.audio_data), "rb") as wav:
                    frames = wav.readframes(wav.getnframes())
                    duration = len(frames) / (wav.getnchannels() * wav.getsampwidth() * wav.getframerate())
            except (EOFError, wave.Error):
                return
            # WAV headers from streaming TTS may contain a placeholder frame count.
            reply.ack_ends_at = max(asyncio.get_running_loop().time() + 0.15, reply.ack_ends_at) + duration
        elif response.type == "final":
            reply.can_wait = False
            if (request.text and request.context_id and self._is_current(session, reply)
                    and self.empty_answer_pattern.fullmatch(response.text or "")):
                task = asyncio.create_task(self._answer_after_silence(session, reply))
                session.wait_task = task
                session.tasks.add(task)
                task.add_done_callback(session.tasks.discard)
            else:
                session.reply = None

    async def _answer_after_silence(self, session: _Session, reply: _Reply) -> None:
        previous = reply.request
        session_id = previous.session_id
        loop = asyncio.get_running_loop()
        try:
            while self._is_current(session, reply):
                last_voice_at = session.last_voice_at or reply.detected_at
                remaining = max(reply.ack_ends_at, last_voice_at + self.silence_seconds) - loop.time()
                if remaining <= 0 and not self._is_recording(session_id):
                    break
                await asyncio.sleep(remaining if remaining > 0 else 0.05)
            else:
                return

            session.wait_task = None
            request = STSRequest(
                type="invoke", session_id=session_id, user_id=previous.user_id, context_id=previous.context_id,
                channel=previous.channel, text=self.resume_text, allow_merge=False, wait_in_queue=True,
                system_prompt_params=previous.system_prompt_params,
                metadata={_RESUME_KEY: _Resume(previous.transaction_id, session.last_voice_at)},
            )
            async for response in self.sts.invoke(request):
                if self._sessions.get(session_id) is not session:
                    break
                await self.sts.handle_response(response)
        except Exception:
            logger.exception("Deferred answer failed for session %s", session_id)
        finally:
            if session.wait_task is asyncio.current_task():
                session.wait_task = None
            if session.reply is reply:
                session.reply = None

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            tasks = session.tasks - {asyncio.current_task()}
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        for session_id in list(self._sessions):
            await self.close_session(session_id)
