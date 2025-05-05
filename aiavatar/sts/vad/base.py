from abc import ABC, abstractmethod
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class SpeechDetector(ABC):
    def __init__(self, *, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._on_speech_detected = self.on_speech_detected_default
        self.should_mute = lambda: False

    def on_speech_detected(self, func):
        self._on_speech_detected = func
        return func

    async def on_speech_detected_default(data: bytes, recorded_duration: float, session_id: str):
        logger.info(f"Speech detected: len={recorded_duration} sec")

    @abstractmethod
    async def process_samples(self, samples: bytes, session_id: str = None):
        pass

    @abstractmethod
    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str = None):
        pass

    @abstractmethod
    async def finalize_session(self, session_id: str):
        pass


class SpeechDetectorDummy(SpeechDetector):
    async def process_samples(self, samples, session_id = None):
        pass

    async def process_stream(self, input_stream, session_id = None):
        pass

    async def finalize_session(self, session_id):
        pass
