from abc import ABC, abstractmethod
import logging
from typing import AsyncGenerator, List, Callable, Awaitable

logger = logging.getLogger(__name__)


class SpeechDetector(ABC):
    def __init__(self, *, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._on_speech_detected = self.on_speech_detected_default
        self.should_mute = lambda: False
        self._on_recording_started: List[Callable[[str], Awaitable[None]]] = []

    def on_speech_detected(self, func):
        self._on_speech_detected = func
        return func

    async def on_speech_detected_default(self, data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        logger.info(f"Speech detected: len={recorded_duration} sec")

    def on_recording_started(self, func: Callable[[str], Awaitable[None]]) -> Callable[[str], Awaitable[None]]:
        self._on_recording_started.append(func)
        return func

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
