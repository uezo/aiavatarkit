import asyncio
from collections import deque
import logging
import math
import struct
from typing import AsyncGenerator, Callable, Optional, Dict
from . import SpeechDetector

logger = logging.getLogger(__name__)


class RecordingSession:
    def __init__(self, session_id: str, preroll_buffer_count: int = 5):
        self.session_id = session_id
        self.is_recording: bool = False
        self.buffer: bytearray = bytearray()
        self.silence_duration: float = 0
        self.record_duration: float = 0
        self.preroll_buffer: deque = deque(maxlen=preroll_buffer_count)
        self.amplitude_threshold: float = 0
        self.data: dict = {}
        self.on_recording_started_triggered: bool = False

    def reset(self):
        # Reset status data except for preroll_buffer
        self.buffer.clear()
        self.is_recording = False
        self.silence_duration = 0
        self.record_duration = 0
        self.on_recording_started_triggered = False


class StandardSpeechDetector(SpeechDetector):
    def __init__(
        self,
        *,
        volume_db_threshold: float = -40.0,
        silence_duration_threshold: float = 0.5,
        max_duration: float = 10.0,
        min_duration: float = 0.2,
        sample_rate: int = 16000,
        channels: int = 1,
        preroll_buffer_count: int = 5,
        to_linear16: Optional[Callable[[bytes], bytes]] = None,
        on_recording_started: Optional[Callable[[str, float], None]] = None,
        debug: bool = False
    ):
        self._volume_db_threshold = volume_db_threshold
        self.amplitude_threshold = 32767 * (10 ** (self.volume_db_threshold / 20.0))
        self.silence_duration_threshold = silence_duration_threshold
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.on_recording_started = on_recording_started
        self.debug = debug
        self.preroll_buffer_count = preroll_buffer_count
        self.to_linear16 = to_linear16
        self.should_mute = lambda: False
        self.recording_sessions: Dict[str, RecordingSession] = {}

    @property
    def volume_db_threshold(self) -> float:
        return self._volume_db_threshold

    @volume_db_threshold.setter
    def volume_db_threshold(self, value: float):
        self._volume_db_threshold = value
        self.amplitude_threshold = 32767 * (10 ** (value / 20.0))
        logger.debug(f"Updated volume_db_threshold to {value} dB, amplitude_threshold={self.amplitude_threshold}")

    async def execute_on_speech_detected(self, recorded_data: bytes, recorded_duration: float, session_id: str):
        try:
            await self._on_speech_detected(recorded_data, recorded_duration, session_id)
        except Exception as ex:
            logger.error(f"Error in task for session {session_id}: {ex}", exc_info=True)

    async def process_samples(self, samples: bytes, session_id: str) -> bool:
        if self.to_linear16:
            samples = self.to_linear16(samples)

        session = self.get_session(session_id)

        if self.should_mute():
            session.reset()
            session.preroll_buffer.clear()
            logger.debug("StandardSpeechDetector is muted.")
            return False

        session.preroll_buffer.append(samples)

        max_amplitude = float(max(abs(sample) for sample, in struct.iter_unpack("<h", samples)))
        sample_duration = (len(samples) / 2) / (self.sample_rate * self.channels)

        if self.debug:
            if max_amplitude > 0:
                current_db = 20 * math.log10(max_amplitude / 32767)
            else:
                current_db = -100.0
            logger.debug(f"dB: {current_db:.2f}, duration: {session.record_duration:.2f}, session: {session.session_id}")

        if not session.is_recording:
            if max_amplitude > session.amplitude_threshold:
                # Start recording
                session.reset()
                session.is_recording = True

                for f in session.preroll_buffer:
                    session.buffer.extend(f)

                session.buffer.extend(samples)
                session.record_duration += sample_duration

        else:
            # In Recording
            session.buffer.extend(samples)
            session.record_duration += sample_duration

            if max_amplitude > session.amplitude_threshold:
                session.silence_duration = 0
            else:
                session.silence_duration += sample_duration

            # Check if we've exceeded min_duration and call callback once
            if (session.record_duration - session.silence_duration >= self.min_duration and 
                not session.on_recording_started_triggered and 
                self.on_recording_started
            ):
                session.on_recording_started_triggered = True
                try:
                    asyncio.create_task(self.on_recording_started(session_id))
                except Exception as ex:
                    logger.error(f"Error in on_recording_started callback: {ex}", exc_info=True)

            if session.silence_duration >= self.silence_duration_threshold:
                recorded_duration = session.record_duration - session.silence_duration
                if recorded_duration < self.min_duration:
                    if self.debug:
                        logger.info(f"Recording too short: {recorded_duration} sec")
                else:
                    if self.debug:
                        logger.info(f"Recording finished: {recorded_duration} sec")
                    recorded_data = bytes(session.buffer)
                    asyncio.create_task(self.execute_on_speech_detected(recorded_data, recorded_duration, session.session_id))
                session.reset()

            elif session.record_duration >= self.max_duration:
                if self.debug:
                    logger.info(f"Recording too long: {session.record_duration} sec")
                session.reset()
        
        return session.is_recording

    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str):
        logger.info("STSPipeline start processing stream.")

        async for data in input_stream:
            if not data:
                break
            await self.process_samples(data, session_id)
            await asyncio.sleep(0.0001)

        self.delete_session(session_id)

        logger.info("STSPipeline finish processing stream.")

    async def finalize_session(self, session_id):
        self.delete_session(session_id)

    def get_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if session is None:
            session = RecordingSession(session_id, self.preroll_buffer_count)
            self.recording_sessions[session_id] = session
        if session.amplitude_threshold == 0:
            session.amplitude_threshold = self.amplitude_threshold
        return session

    def reset_session(self, session_id: str):
        if session := self.recording_sessions.get(session_id):
            session.reset()

    def delete_session(self, session_id: str):
        if session_id in self.recording_sessions:
            self.recording_sessions[session_id].reset()
            del self.recording_sessions[session_id]

    def get_session_data(self, session_id: str, key: str):
        session = self.recording_sessions.get(session_id)
        if session:
            return session.data.get(key)

    def set_session_data(self, session_id: str, key: str, value: any, create_session: bool = False):
        if create_session:
            session = self.get_session(session_id)
        else:
            session = self.recording_sessions.get(session_id)

        if session:
            session.data[key] = value

    def set_volume_db_threshold(self, session_id: str, value: float):
        session = self.get_session(session_id)
        session.amplitude_threshold = 32767 * (10 ** (value / 20.0))
