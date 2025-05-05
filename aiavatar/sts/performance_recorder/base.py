from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PerformanceRecord:
    transaction_id: str
    user_id: str = None
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
    llm_first_chunk_time: float = 0
    llm_first_voice_chunk_time: float = 0
    llm_time: float = 0
    tts_first_chunk_time: float = 0
    tts_time: float = 0
    total_time: float = 0


class PerformanceRecorder(ABC):
    @abstractmethod
    def record(self, record: PerformanceRecord):
        pass

    @abstractmethod
    def close(self):
        pass
