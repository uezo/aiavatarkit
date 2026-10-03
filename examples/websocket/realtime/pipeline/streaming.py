"""Adapter-facing contract for continuous speech sessions.

Unlike STSPipeline.invoke(), audio input spans a whole connection. Providers
with response boundaries may still emit per-turn final events. Implementations
use the existing STSResponse envelope and response handlers.
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import replace
import logging
from typing import TYPE_CHECKING

from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.vad.base import SpeechDetector

if TYPE_CHECKING:
    from aiavatar.sts.pipeline import ResponseHandler

logger = logging.getLogger(__name__)


class _StreamingSpeechDetector(SpeechDetector):
    """Forward the existing detector interface to a streaming pipeline."""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    async def process_samples(self, samples: bytes, session_id: str = None):
        return await self.pipeline.process_audio_samples(samples, session_id)

    async def process_stream(self, input_stream, session_id: str = None):
        async for samples in input_stream:
            await self.process_samples(samples, session_id)

    async def finalize_session(self, session_id: str):
        await self.pipeline.finalize(session_id)

    async def _notify_speech_started(self, session_id: str):
        await self._execute_on_voiced(session_id)
        for handler in self._on_recording_started:
            try:
                await handler(session_id)
            except Exception:
                logger.exception("Error in streaming on_recording_started callback")


class StreamingSTSPipeline(ABC):
    def __init__(self):
        self.response_handlers: list["ResponseHandler"] = []
        self.vad = _StreamingSpeechDetector(self)
        self._on_accepted_handlers = []
        self._prepared_sessions: dict[str, asyncio.Event] = {}

    def on_accepted(self, func=None, *, channels=None):
        if channels is not None:
            channels = frozenset([channels]) if isinstance(channels, str) else frozenset(channels)
            if not channels or any(not isinstance(channel, str) or not channel for channel in channels):
                raise ValueError("channels must contain non-empty strings")

        def register(handler):
            self._on_accepted_handlers.append((channels, handler))
            return handler

        return register if func is None else register(func)

    async def _execute_on_accepted(self, request: STSRequest):
        channel = request.channel
        for channels, handler in self._on_accepted_handlers:
            if channels is None or channel in channels:
                await handler(request)

    def _remember_session(self, request: STSRequest):
        """Called by providers after validating a new session request."""
        for key in ("user_id", "context_id", "channel", "system_prompt_params"):
            self.vad.set_session_data(request.session_id, key, getattr(request, key), create_session=True)
        if "barge_in_enabled" in (request.metadata or {}):
            self.vad.set_session_data(
                request.session_id, "barge_in_enabled", bool(request.metadata["barge_in_enabled"]),
            )

    async def prepare_session(self, request: STSRequest):
        """Connect from a session-start hook while holding provider output."""
        session_id = request.session_id
        if not session_id:
            raise ValueError("session_id is required")
        if session_id in self._prepared_sessions or session_id in self.vad._session_data:
            raise ValueError("Streaming session is already active")
        gate = asyncio.Event()
        self._prepared_sessions[session_id] = gate
        try:
            await self.start_session(request)
        except BaseException:
            if self._prepared_sessions.get(session_id) is gate:
                self._release_session(session_id)
            raise

    def activate_session(self, session_id: str):
        """Release output after the adapter has delivered its connected event."""
        if gate := self._prepared_sessions.get(session_id):
            gate.set()

    async def _session_connected(self, response: STSResponse, ready: asyncio.Future):
        gate = self._prepared_sessions.get(response.session_id)
        if gate is None:
            await self.handle_response(response)
        if not ready.done():
            ready.set_result(None)
        if gate is not None:
            await gate.wait()

    def _release_session(self, session_id: str):
        """Release state after the provider verifies it still owns the session."""
        if gate := self._prepared_sessions.pop(session_id, None):
            gate.set()
        self.vad._session_data.pop(session_id, None)

    async def invoke(self, request: STSRequest):
        # Reject at session level without accepting a new transaction, which
        # would invalidate speech already in progress at the adapter.
        message = "This streaming pipeline accepts continuous audio, not invoke requests."
        yield STSResponse(
            type="error", session_id=request.session_id, user_id=request.user_id,
            context_id=request.context_id,
            voice_text=message, metadata={"error": message},
        )

    def add_response_handler(self, handler: "ResponseHandler"):
        self.response_handlers.append(handler)

    async def handle_response(self, response: STSResponse):
        if response.type == "error" and not response.voice_text:
            response = replace(response, voice_text=(response.metadata or {}).get("error") or "Request failed")
        elif response.type == "session_closed":
            response = replace(response, text="[operation:hangup]")
        for handler in self.response_handlers:
            if handler.can_handle(response.session_id):
                await handler.handle_response(response)
                return
        logger.warning("No response handler for streaming session: %s", response.session_id)

    async def stop_response(self, session_id: str, context_id: str):
        """Clear application playback; this does not cancel provider inference."""
        for handler in self.response_handlers:
            if handler.can_handle(session_id):
                await handler.stop_response(session_id, context_id)
                return

    @abstractmethod
    async def start_session(self, request: STSRequest):
        """Start a session and emit connected before any audio or transcripts."""

    @abstractmethod
    async def process_audio_samples(self, samples: bytes, context_id: str):
        """Accept audio for a live session (context_id is the legacy method name)."""

    async def configure_session(self, session_id: str, metadata: dict):
        """Apply supported session settings; others have no effect by default."""

    @abstractmethod
    async def finalize(self, session_id: str):
        """Release one session, including its background tasks; idempotent."""

    @abstractmethod
    async def shutdown(self):
        """Release all resources owned by this pipeline."""
