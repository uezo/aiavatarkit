from abc import ABC, abstractmethod
import asyncio
from collections import deque
import logging
from typing import AsyncGenerator, List, Callable, Awaitable, Optional, Any

logger = logging.getLogger(__name__)


class RecordingSessionBase:
    """Base class for recording sessions with common interface for on_recording_started trigger."""
    def __init__(self, session_id: str, preroll_buffer_count: int = 5):
        self.session_id = session_id
        self.is_recording: bool = False
        self.buffer: bytearray = bytearray()
        self.silence_duration: float = 0
        self.record_duration: float = 0
        self.preroll_buffer: deque = deque(maxlen=preroll_buffer_count)
        self.data: dict = {}
        self.on_recording_started_triggered: bool = False
        # For text-based trigger condition (used by stream detector)
        self.last_recognized_text: Optional[str] = None

    def reset(self):
        self.buffer.clear()
        self.is_recording = False
        self.silence_duration = 0
        self.record_duration = 0
        self.on_recording_started_triggered = False
        self.last_recognized_text = None


class SpeechDetector(ABC):
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        on_recording_started_min_duration: float = 1.5,
        on_recording_started_min_text_length: int = 2
    ):
        self.sample_rate = sample_rate
        self._on_speech_detected = self.on_speech_detected_default
        self.should_mute = lambda: False
        self._on_recording_started: List[Callable[[str], Awaitable[None]]] = []
        self._should_trigger_recording_started: Optional[Callable[[Optional[str], Any], bool]] = None
        # Parameters for on_recording_started trigger
        self.on_recording_started_min_duration = on_recording_started_min_duration
        self.on_recording_started_min_text_length = on_recording_started_min_text_length

    def on_speech_detected(self, func):
        self._on_speech_detected = func
        return func

    async def on_speech_detected_default(self, data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        logger.info(f"Speech detected: len={recorded_duration} sec")

    def on_recording_started(self, func: Callable[[str], Awaitable[None]]) -> Callable[[str], Awaitable[None]]:
        self._on_recording_started.append(func)
        return func

    def should_trigger_recording_started(self, func: Callable[[Optional[str], Any], bool]):
        """Decorator to set custom trigger condition for on_recording_started callback.

        The function receives (text: Optional[str], session: Any) and returns bool.
        If not set, subclasses should use their default trigger logic.
        """
        self._should_trigger_recording_started = func
        return func

    def _default_should_trigger_recording_started(self, text: Optional[str], session: RecordingSessionBase) -> bool:
        """Default trigger condition for on_recording_started callback."""
        # Duration-based condition
        if session.record_duration - session.silence_duration >= self.on_recording_started_min_duration:
            return True
        # Text length-based condition
        if text and len(text) >= self.on_recording_started_min_text_length:
            return True
        return False

    async def _check_and_trigger_recording_started(self, session: RecordingSessionBase, text: Optional[str] = None):
        """Check trigger condition and fire on_recording_started callback if met."""
        if session.on_recording_started_triggered or not self._on_recording_started:
            return

        # Use text from parameter or session
        check_text = text if text is not None else session.last_recognized_text

        # Use custom function if provided, otherwise use default
        if self._should_trigger_recording_started:
            should_trigger = self._should_trigger_recording_started(check_text, session)
        else:
            should_trigger = self._default_should_trigger_recording_started(check_text, session)

        if should_trigger:
            session.on_recording_started_triggered = True
            for handler in self._on_recording_started:
                async def _run(h, session_id):
                    try:
                        await h(session_id)
                    except Exception:
                        logger.error("Error in on_recording_started callback", exc_info=True)
                asyncio.create_task(_run(handler, session.session_id))

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict) -> dict:
        allowed_keys = self.get_config().keys()
        updated = {}
        for k, v in config.items():
            if v is None:
                continue
            if k not in allowed_keys:
                continue
            try:
                setattr(self, k, v)
                updated[k] = v
            except Exception:
                pass
        return updated

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
