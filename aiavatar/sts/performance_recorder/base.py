from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional


TIME_ORIGIN_USER_SPEECH_END = "user_speech_end"
TIME_ORIGIN_PIPELINE_START = "pipeline_start"
VALID_TIME_ORIGINS = {
    TIME_ORIGIN_USER_SPEECH_END,
    TIME_ORIGIN_PIPELINE_START,
}


@dataclass
class PerformanceRecord:
    transaction_id: str
    user_id: str = None
    session_id: str = None
    context_id: str = None
    stt_name: str = None
    llm_name: str = None
    tts_name: str = None
    request_text: str = None
    response_text: str = None
    request_files: str = None
    response_voice_text: str = None
    voice_length: float = 0
    stt_time: float = 0
    stop_response_time: float = 0
    before_llm_time: float = 0
    llm_first_chunk_time: float = 0
    llm_first_voice_chunk_time: float = 0
    llm_time: float = 0
    tts_first_chunk_time: float = 0
    tts_time: float = 0
    total_time: float = 0
    quick_response_text: str = None
    error_info: str = None
    tool_calls: str = None
    speech_end_at: Optional[datetime] = None
    silence_threshold_time: Optional[float] = None
    stt_after_threshold_time: Optional[float] = None
    turn_end_gate_time: Optional[float] = None
    turn_end_gate_held: Optional[bool] = None


class PerformanceRecorder(ABC):
    _PRE_PIPELINE_TIME_FIELDS = (
        "silence_threshold_time",
        "stt_after_threshold_time",
        "turn_end_gate_time",
    )
    _PIPELINE_LAP_TIME_FIELDS = (
        "stt_time",
        "stop_response_time",
        "before_llm_time",
        "llm_first_chunk_time",
        "llm_first_voice_chunk_time",
        "llm_time",
        "tts_first_chunk_time",
        "tts_time",
        "total_time",
    )

    def __init__(self, time_origin: str = TIME_ORIGIN_USER_SPEECH_END):
        if time_origin not in VALID_TIME_ORIGINS:
            raise ValueError(
                f"Invalid time_origin: {time_origin}. "
                f"Must be one of {sorted(VALID_TIME_ORIGINS)}"
            )
        self.time_origin = time_origin

    def _prepare_record_for_storage(
        self,
        record: PerformanceRecord,
    ) -> PerformanceRecord:
        """Convert timing values to laps from the configured origin."""
        if self.time_origin == TIME_ORIGIN_PIPELINE_START:
            return record

        pre_pipeline_time = 0.0
        prepared_record = replace(record)
        for field_name in self._PRE_PIPELINE_TIME_FIELDS:
            value = getattr(record, field_name)
            if value is None:
                continue
            pre_pipeline_time += value
            setattr(prepared_record, field_name, pre_pipeline_time)

        if pre_pipeline_time <= 0:
            return record

        for field_name in self._PIPELINE_LAP_TIME_FIELDS:
            value = getattr(prepared_record, field_name)
            if value is not None and value > 0:
                setattr(
                    prepared_record,
                    field_name,
                    pre_pipeline_time + value,
                )
        return prepared_record

    @abstractmethod
    def record(self, record: PerformanceRecord):
        pass

    @abstractmethod
    def close(self):
        pass
