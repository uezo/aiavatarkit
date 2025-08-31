import asyncio
from collections import deque
import logging
import numpy as np
import struct
import threading
import torch
from typing import AsyncGenerator, Callable, Optional, Dict
from . import SpeechDetector

logger = logging.getLogger(__name__)


class RecordingSession:
    def __init__(self, session_id: str, preroll_buffer_count: int = 5, vad_iterator=None):
        self.session_id = session_id
        self.is_recording: bool = False
        self.buffer: bytearray = bytearray()
        self.silence_duration: float = 0
        self.record_duration: float = 0
        self.preroll_buffer: deque = deque(maxlen=preroll_buffer_count)
        self.amplitude_threshold: Optional[float] = None
        self.data: dict = {}
        self.vad_buffer: bytearray = bytearray()
        self.vad_iterator = vad_iterator
        self.on_recording_started_triggered: bool = False

    def reset(self):
        # Reset status data except for preroll_buffer
        self.buffer.clear()
        self.is_recording = False
        self.silence_duration = 0
        self.record_duration = 0
        self.on_recording_started_triggered = False
        self.amplitude_threshold = None
        # Don't reset vad_buffer here as it's used for continuous processing


class SileroSpeechDetector(SpeechDetector):
    def __init__(
        self,
        *,
        volume_db_threshold: Optional[float] = None,
        silence_duration_threshold: float = 0.5,
        max_duration: float = 10.0,
        min_duration: float = 0.2,
        sample_rate: int = 16000,
        channels: int = 1,
        preroll_buffer_count: int = 5,
        to_linear16: Optional[Callable[[bytes], bytes]] = None,
        debug: bool = False,
        model_path: Optional[str] = None,
        speech_probability_threshold: float = 0.5,
        chunk_size: int = 512,
        model_pool_size: int = 1,
        on_recording_started: Optional[Callable[[str, float], None]] = None
    ):
        self._volume_db_threshold = volume_db_threshold
        if volume_db_threshold is not None:
            self.amplitude_threshold = 32767 * (10 ** (self.volume_db_threshold / 20.0))
        else:
            self.amplitude_threshold = None
        self.silence_duration_threshold = silence_duration_threshold
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.debug = debug
        self.preroll_buffer_count = preroll_buffer_count
        self.to_linear16 = to_linear16
        self.should_mute = lambda: False
        self.recording_sessions: Dict[str, RecordingSession] = {}
        self.on_recording_started = on_recording_started

        # Silero VAD specific parameters
        self.speech_probability_threshold = speech_probability_threshold
        self.chunk_size = chunk_size
        self.model_pool_size = model_pool_size
        
        # Initialize Silero VAD model pool
        self._init_silero_model(model_path)

    @property
    def volume_db_threshold(self) -> float:
        return self._volume_db_threshold

    @volume_db_threshold.setter
    def volume_db_threshold(self, value: Optional[float]):
        self._volume_db_threshold = value
        if value is not None:
            self.amplitude_threshold = 32767 * (10 ** (value / 20.0))
            logger.debug(f"Updated volume_db_threshold to {value} dB, amplitude_threshold={self.amplitude_threshold}")
        else:
            self.amplitude_threshold = None
            logger.debug("Volume threshold disabled (set to None)")

    def _init_silero_model(self, model_path: Optional[str] = None):
        """Initialize Silero VAD model pool"""
        try:
            # Initialize model pool and locks
            self.model_pool = []
            self.model_locks = []
            
            for i in range(self.model_pool_size):
                if model_path:
                    model = torch.jit.load(model_path)
                    # For pre-loaded models, we need to get utils separately
                    _, utils = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        force_reload=False,
                        onnx=False
                    )
                else:
                    # Load default Silero VAD model
                    model, utils = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        force_reload=False,
                        onnx=False
                    )
                
                self.model_pool.append(model)
                self.model_locks.append(threading.Lock())
                
                # Store utility functions (same for all models)
                if i == 0:  # Only need to store once
                    self.get_speech_timestamps = utils[0]
                    self.save_audio = utils[1]
                    self.read_audio = utils[2]
                    self.VADIterator = utils[3]
                    self.collect_chunks = utils[4]
                    # Store VAD iterator class for per-session creation
                    self.VADIteratorClass = self.VADIterator
            
            logger.info(f"Silero VAD model pool initialized successfully with {self.model_pool_size} models")
            
        except Exception as ex:
            logger.error(f"Failed to initialize Silero VAD model pool: {ex}")
            raise

    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        # Convert bytes to int16 array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 and normalize to [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        return audio_float32

    def _get_model_and_lock(self, session_id: str):
        """Get model and lock for a session using consistent hashing"""
        if self.model_pool_size == 1:
            return self.model_pool[0], self.model_locks[0]
        
        # Use session_id hash to consistently assign model
        model_idx = hash(session_id) % self.model_pool_size
        return self.model_pool[model_idx], self.model_locks[model_idx]

    def _detect_speech_silero(self, audio_bytes: bytes, session_id: str) -> bool:
        try:
            # Convert bytes to numpy array
            audio_np = self._bytes_to_numpy(audio_bytes)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_np)
            
            # Get assigned model and lock for this session
            model, model_lock = self._get_model_and_lock(session_id)
            
            # Use model directly for speech probability with thread lock
            with model_lock:
                with torch.no_grad():
                    speech_prob = model(audio_tensor, self.sample_rate).item()

            # Check if speech probability exceeds threshold
            return speech_prob > self.speech_probability_threshold

        except Exception as e:
            logger.error(f"Error in Silero VAD detection: {e}")
            return False

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
            session.vad_buffer.clear()
            logger.debug("SileroSpeechDetector is muted.")
            return False

        session.preroll_buffer.append(samples)
        session.vad_buffer.extend(samples)

        # Calculate sample duration
        sample_duration = (len(samples) / 2) / (self.sample_rate * self.channels)

        # Use Silero VAD only when we have enough data (minimum chunk_size)
        speech_detected = False
        if len(session.vad_buffer) >= self.chunk_size * 2:  # chunk_size * 2 bytes (16-bit samples)
            # Take the required chunk size from the end of buffer
            vad_chunk = bytes(session.vad_buffer[-self.chunk_size * 2:])
            speech_detected = self._detect_speech_silero(vad_chunk, session_id)
            if speech_detected and session.amplitude_threshold is not None:
                # Check the volume if threshold is set
                max_amplitude = float(max(abs(sample) for sample, in struct.iter_unpack("<h", samples)))
                if max_amplitude <= session.amplitude_threshold:
                    speech_detected = False

            # Keep only the last chunk_size worth of data to avoid unbounded growth
            if len(session.vad_buffer) > self.chunk_size * 4:  # Keep 2x chunk_size
                session.vad_buffer = session.vad_buffer[-self.chunk_size * 2:]

        if self.debug:
            logger.debug(f"Speech detected: {speech_detected}, duration: {session.record_duration:.2f}, session: {session.session_id}")

        if not session.is_recording:
            if speech_detected:
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

            if speech_detected:
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
        logger.info("SileroSpeechDetector start processing stream.")

        async for data in input_stream:
            if not data:
                break
            await self.process_samples(data, session_id)
            await asyncio.sleep(0.0001)

        self.delete_session(session_id)

        logger.info("SileroSpeechDetector finish processing stream.")

    async def finalize_session(self, session_id):
        self.delete_session(session_id)

    def get_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if session is None:
            # Create VAD iterator for this session using first model
            vad_iterator = self.VADIteratorClass(
                self.model_pool[0], 
                threshold=self.speech_probability_threshold, 
                sampling_rate=self.sample_rate
            )
            session = RecordingSession(session_id, self.preroll_buffer_count, vad_iterator)
            self.recording_sessions[session_id] = session
        if session.amplitude_threshold is None:
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

    def set_speech_probability_threshold(self, threshold: float):
        """Set Silero VAD speech probability threshold"""
        self.speech_probability_threshold = threshold
        # Re-initialize VAD iterator for all existing sessions using first model
        for session in self.recording_sessions.values():
            session.vad_iterator = self.VADIteratorClass(
                self.model_pool[0], 
                threshold=self.speech_probability_threshold, 
                sampling_rate=self.sample_rate
            )
        logger.debug(f"Updated Silero VAD speech probability threshold to {threshold}")

    def reset_vad_state(self, session_id: str = None):
        """Reset VAD iterator state for specific session or all sessions"""
        if session_id:
            session = self.recording_sessions.get(session_id)
            if session and session.vad_iterator:
                session.vad_iterator.reset_states()
                logger.debug(f"Silero VAD state reset for session {session_id}")
        else:
            # Reset all sessions
            for session in self.recording_sessions.values():
                if session.vad_iterator:
                    session.vad_iterator.reset_states()
            logger.debug("Silero VAD state reset for all sessions")

    def set_volume_db_threshold(self, session_id: str, value: float):
        session = self.get_session(session_id)
        session.amplitude_threshold = 32767 * (10 ** (value / 20.0))
