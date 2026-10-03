"""Timing and immutable performance-record snapshots for one Realtime turn."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
from time import perf_counter

from aiavatar.sts.models import STSRequest
from aiavatar.sts.performance_recorder import PerformanceRecord, PerformanceRecorder

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Remain inactive until a turn with performance recording is accepted."""

    record: PerformanceRecord | None = None
    started_at: float = 0.0
    recorded: bool = False
    first_chunks: set[str] = field(default_factory=set)

    def begin(
        self, request: STSRequest, context_id: str, request_text: str | None,
        llm_name: str, tts_name: str,
    ):
        self.started_at = perf_counter()
        # Local acceptance is the legacy speech-end origin. Realtime does not
        # measure the preceding speech or provider VAD interval.
        accepted_at = datetime.now(timezone.utc)
        self.record = PerformanceRecord(
            transaction_id=request.transaction_id, session_id=request.session_id,
            user_id=request.user_id, context_id=context_id, channel=request.channel,
            stt_name=None, llm_name=llm_name, tts_name=tts_name,
            request_text=request_text,
            request_files=json.dumps(request.files or []), voice_length=request.audio_duration,
            stt_time=0, speech_end_at=accepted_at, silence_threshold_time=0,
            stt_after_threshold_time=0, turn_end_gate_time=0, turn_end_gate_held=False,
        )

    @property
    def active(self) -> PerformanceRecord | None:
        if not self.recorded:
            return self.record

    def lap(self, field_name: str, *, first: bool = False):
        performance = self.active
        if performance is None:
            return
        if first:
            if field_name in self.first_chunks:
                return
            self.first_chunks.add(field_name)
        setattr(performance, field_name, max(0.0, perf_counter() - self.started_at))

    def finish(
        self, recorder: PerformanceRecorder, request: STSRequest, transaction_id: str,
        response_text: str, status: str, reason: str | None = None, provider_status: str | None = None,
    ):
        performance = self.active
        if performance is None:
            return
        performance.transaction_id = transaction_id
        performance.session_id = request.session_id
        performance.user_id = request.user_id
        # The context was frozen when this metric record began; request hooks
        # may update the other identity and display fields before completion.
        performance.channel = request.channel
        if request.text is not None:
            performance.request_text = request.text
        performance.response_text = response_text
        self.lap("total_time")
        if status != "final":
            info = {"status": status, "reason": reason or status}
            if provider_status is not None:
                info["provider_status"] = (provider_status if provider_status in
                    {"cancelled", "failed", "incomplete"} else "unknown")
            performance.error_info = json.dumps(info)
        # Recorders may enqueue the object. Freeze the snapshot before calling
        # one so stale provider events or canceled TTS cannot change its values.
        self.recorded = True
        try:
            recorder.record(replace(performance))
        except Exception:
            logger.warning("Realtime performance recording failed")
