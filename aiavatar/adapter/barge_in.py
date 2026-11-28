"""Coordinate response delivery with a user who is still speaking."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
from math import isfinite
from time import monotonic

from ..sts.models import STSRequest, STSResponse
from .base import Adapter
from .models import AIAvatarResponse

logger = logging.getLogger(__name__)


@dataclass
class _FirstAudio:
    transaction_id: str
    seen: bool = False


class FirstResponseAudioHold:
    """Briefly hold the first non-ACK audio in an adapter's pre-send hook."""

    def __init__(self, *, applies_to: Callable[[STSRequest], bool]):
        self._applies_to = applies_to
        self._adapter: Adapter | None = None
        self._vad = None
        self._debug = False
        self._first_audio: dict[str, _FirstAudio] = {}

    def bind(self, adapter: Adapter) -> None:
        """Hold audio in the adapter's pre-send hook for requests from its channel.

        The hold uses the adapter's pipeline and binds to one adapter, once.
        """
        if self._adapter is not None:
            raise RuntimeError("FirstResponseAudioHold is already bound to an adapter")
        self._adapter = adapter
        sts = adapter.sts
        self._vad = sts.vad
        self._debug = getattr(sts, "debug", False)
        sts.on_accepted(self._on_accepted, channels=adapter.channel)
        adapter.on_response(self._on_response)
        sts.on_finalize(self._on_finalize)
        sts.on_shutdown(self._on_shutdown)

    async def _on_accepted(self, request: STSRequest) -> None:
        if self._applies_to(request):
            self._first_audio[request.session_id] = _FirstAudio(request.transaction_id)
        else:
            self._first_audio.pop(request.session_id, None)

    async def _on_finalize(self, session_id: str) -> None:
        self._first_audio.pop(session_id, None)

    async def _on_shutdown(self) -> None:
        self._first_audio.clear()

    async def _on_response(self, message: AIAvatarResponse, response: STSResponse) -> None:
        session_id = response.session_id
        first_audio = self._first_audio.get(session_id)
        if first_audio is None or first_audio.transaction_id != response.transaction_id:
            return
        if response.type in ("final", "canceled", "cancelled", "error"):
            self._first_audio.pop(session_id, None)
            return
        if (
            first_audio.seen or response.type != "chunk" or not message.audio_data
            or (message.voice_text_tag or response.voice_text_tag) == "ack"
            or (response.metadata or {}).get("is_guardrail_triggered")
        ):
            return
        first_audio.seen = True

        recordings = getattr(self._vad, "recording_sessions", {})
        recording = recordings.get(session_id)
        if recording is None or not recording.is_recording:
            return
        threshold = self._vad.on_recording_started_min_duration
        if not isfinite(threshold) or threshold <= 0:
            return
        recorded = recording.record_duration - getattr(recording, "silence_duration", 0.0)
        budget = max(0.0, threshold - recorded)
        deadline = monotonic() + budget
        recording_id = getattr(recording, "recording_id", None)
        logged = False
        while (
            self._first_audio.get(session_id) is first_audio
            and recordings.get(session_id) is recording and recording.is_recording
            and getattr(recording, "recording_id", None) == recording_id
            and self._vad.get_session_data(session_id, "barge_in_enabled") is True
        ):
            if getattr(recording, "on_recording_started_triggered", False):
                # Let the existing VAD mute callback run before delivery resumes.
                await asyncio.sleep(0)
                return
            remaining = deadline - monotonic()
            if remaining <= 0:
                return
            if self._debug and not logged:
                logger.info("Holding first response audio: transaction=%s, budget_seconds=%.3f",
                            response.transaction_id, budget)
                logged = True
            await asyncio.sleep(min(0.02, remaining))
