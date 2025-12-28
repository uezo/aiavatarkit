import asyncio
from collections import deque
import logging
import numpy as np
import struct
import threading
import torch
from typing import AsyncGenerator, Callable, Optional, Dict, List, Awaitable
# pip install azure-cognitiveservices-speech
import azure.cognitiveservices.speech as speechsdk
from . import SpeechDetector

logger = logging.getLogger(__name__)


class RecordingSession:
    def __init__(self, session_id: str, preroll_buffer_count: int = 5):
        self.session_id = session_id
        self.is_recording: bool = False
        self.buffer: bytearray = bytearray()
        self.recognized_text: str = ""
        self.silence_duration: float = 0
        self.record_duration: float = 0
        self.preroll_buffer: deque = deque(maxlen=preroll_buffer_count)
        self.amplitude_threshold: Optional[float] = None
        self.data: dict = {}
        self.vad_buffer: bytearray = bytearray()
        self.on_recording_started_triggered: bool = False
        # Azure Speech recognition
        self.azure_push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None
        self.azure_recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.azure_recognized_event: threading.Event = threading.Event()
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

    def reset(self):
        # Reset status data except for preroll_buffer
        self.buffer.clear()
        self.recognized_text = ""
        self.is_recording = False
        self.silence_duration = 0
        self.record_duration = 0
        self.on_recording_started_triggered = False
        self.amplitude_threshold = None
        self.azure_recognized_event.clear()
        # Don't reset vad_buffer here as it's used for continuous processing


class AzureStreamSpeechDetector(SpeechDetector):
    def __init__(
        self,
        *,
        azure_subscription_key: str,
        azure_region: str,
        azure_language: str = "ja-JP",
        volume_db_threshold: Optional[float] = None,
        silence_duration_threshold: float = 0.5,
        max_duration: float = 10.0,
        sample_rate: int = 16000,
        channels: int = 1,
        preroll_buffer_count: int = 5,
        to_linear16: Optional[Callable[[bytes], bytes]] = None,
        debug: bool = False,
        model_path: Optional[str] = None,
        speech_probability_threshold: float = 0.5,
        chunk_size: int = 512,
        model_pool_size: int = 1,
        on_recording_started: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        super().__init__(sample_rate=sample_rate)
        # Azure Speech SDK settings
        self.azure_subscription_key = azure_subscription_key
        self.azure_region = azure_region
        self.azure_language = azure_language

        self._volume_db_threshold = volume_db_threshold
        if volume_db_threshold is not None:
            self.amplitude_threshold = 32767 * (10 ** (self.volume_db_threshold / 20.0))
        else:
            self.amplitude_threshold = None
        self.silence_duration_threshold = silence_duration_threshold
        self.max_duration = max_duration
        self.channels = channels
        self.debug = debug
        self.preroll_buffer_count = preroll_buffer_count
        self.to_linear16 = to_linear16
        self.recording_sessions: Dict[str, RecordingSession] = {}
        if on_recording_started:
            self._on_recording_started.append(on_recording_started)
        self._on_speech_detecting: Callable[[str, RecordingSession], Awaitable[None]] = None

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

            for _ in range(self.model_pool_size):
                if model_path:
                    model = torch.jit.load(model_path)
                else:
                    # Load default Silero VAD model
                    model, _ = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        force_reload=False,
                        onnx=False
                    )

                self.model_pool.append(model)
                self.model_locks.append(threading.Lock())

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

            # Get model and lock
            model, model_lock = self._get_model_and_lock(session_id)

            with model_lock:
                with torch.no_grad():
                    speech_prob = model(audio_tensor, self.sample_rate).item()
                return speech_prob > self.speech_probability_threshold

        except Exception as e:
            logger.error(f"Error in Silero VAD detection: {e}")
            return False

    def _start_azure_recognition(self, session: RecordingSession):
        """Start Azure Speech recognition for a session"""
        try:
            # Capture current event loop for use in callbacks from Azure SDK threads
            try:
                session.event_loop = asyncio.get_running_loop()
            except RuntimeError:
                session.event_loop = asyncio.get_event_loop()

            speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_subscription_key,
                region=self.azure_region,
            )
            speech_config.speech_recognition_language = self.azure_language

            # Set segmentation silence timeout (controls when recognized event fires)
            silence_timeout_ms = int(self.silence_duration_threshold * 1000)
            speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
                str(silence_timeout_ms)
            )

            audio_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=self.sample_rate,
                bits_per_sample=16,
                channels=self.channels
            )
            session.azure_push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
            audio_config = speechsdk.audio.AudioConfig(stream=session.azure_push_stream)

            session.azure_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            def on_recognizing(evt):
                if self.debug:
                    logger.debug(f"Azure recognizing: {evt.result.text}")
                session.recognized_text = evt.result.text

                if self._on_speech_detecting:
                    try:
                        asyncio.run_coroutine_threadsafe(self._on_speech_detecting(evt.result.text, session), session.event_loop)
                    except Exception as ex:
                        logger.error("Error in on_speech_detecting callback", exc_info=True)

                # Trigger on_recording_started callback on first recognizing event
                if not session.on_recording_started_triggered and self._on_recording_started:
                    session.on_recording_started_triggered = True
                    if self.debug:
                        logger.info(f"Recording started (Azure recognizing): {session.session_id}")
                    for handler in self._on_recording_started:
                        try:
                            asyncio.run_coroutine_threadsafe(handler(session.session_id), session.event_loop)
                        except Exception as ex:
                            logger.error("Error in on_recording_started callback", exc_info=True)

            def on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    if self.debug:
                        logger.info(f"Azure recognized: {evt.result.text}")
                    session.recognized_text = evt.result.text
                    session.azure_recognized_event.set()
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    # Only set event if currently recording, otherwise ignore
                    # Note: NoMatch during non-recording (e.g., initial silence timeout) doesn't disconnect,
                    # so no reconnection needed - connection remains active for next utterance
                    if session.is_recording:
                        if self.debug:
                            logger.info("Azure: No speech recognized (during recording)")
                        session.azure_recognized_event.set()
                    else:
                        if self.debug:
                            logger.debug("Azure: No speech recognized (ignored, not recording)")

            def on_canceled(evt):
                if self.debug:
                    logger.warning(f"Azure recognition canceled: {evt.reason}")
                session.azure_recognized_event.set()

            session.azure_recognizer.recognizing.connect(on_recognizing)
            session.azure_recognizer.recognized.connect(on_recognized)
            session.azure_recognizer.canceled.connect(on_canceled)

            session.azure_recognizer.start_continuous_recognition()
            if self.debug:
                logger.info(f"Azure recognition started for session {session.session_id}")

        except Exception as e:
            logger.error(f"Failed to start Azure recognition: {e}")

    def _stop_azure_recognition(self, session: RecordingSession):
        """Stop Azure Speech recognition for a session"""
        try:
            if session.azure_push_stream:
                session.azure_push_stream.close()
                session.azure_push_stream = None

            if session.azure_recognizer:
                session.azure_recognizer.stop_continuous_recognition()
                session.azure_recognizer = None

            if self.debug:
                logger.info(f"Azure recognition stopped for session {session.session_id}")

        except Exception as e:
            logger.error(f"Failed to stop Azure recognition: {e}")

    def _write_to_azure_stream(self, session: RecordingSession, audio_data: bytes):
        """Write audio data to Azure push stream"""
        if session.azure_push_stream:
            try:
                session.azure_push_stream.write(audio_data)
            except Exception as e:
                logger.error(f"Failed to write to Azure stream: {e}")

    def on_speech_detecting(self, func):
        self._on_speech_detecting = func
        return func

    async def execute_on_speech_detected(self, recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        try:
            await self._on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id)
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
            logger.debug("AzureStreamSpeechDetector is muted.")
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

                # Write preroll buffer to Azure stream
                for f in session.preroll_buffer:
                    session.buffer.extend(f)
                    self._write_to_azure_stream(session, f)

                session.buffer.extend(samples)
                self._write_to_azure_stream(session, samples)
                session.record_duration += sample_duration

        else:
            # In Recording
            session.buffer.extend(samples)
            self._write_to_azure_stream(session, samples)
            session.record_duration += sample_duration

            # Check if Azure recognized (speech ended)
            if session.azure_recognized_event.is_set():
                recorded_duration = session.record_duration
                if self.debug:
                    logger.info(f"Recording finished (Azure recognized): {recorded_duration} sec, text: {session.recognized_text}")
                recorded_data = bytes(session.buffer)
                asyncio.create_task(self.execute_on_speech_detected(recorded_data, session.recognized_text, None, recorded_duration, session.session_id))
                session.reset()

            elif session.record_duration >= self.max_duration:
                if self.debug:
                    logger.info(f"Recording too long: {session.record_duration} sec")
                session.reset()

        return session.is_recording

    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str):
        logger.info("AzureStreamSpeechDetector start processing stream.")

        async for data in input_stream:
            if not data:
                break
            await self.process_samples(data, session_id)
            await asyncio.sleep(0.0001)

        self.delete_session(session_id)

        logger.info("AzureStreamSpeechDetector finish processing stream.")

    async def finalize_session(self, session_id):
        self.delete_session(session_id)

    def get_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if session is None:
            session = RecordingSession(session_id, self.preroll_buffer_count)
            self.recording_sessions[session_id] = session
            # Start Azure recognition when session is created (keep connection until session deleted)
            self._start_azure_recognition(session)
        if session.amplitude_threshold is None:
            session.amplitude_threshold = self.amplitude_threshold
        return session

    def reset_session(self, session_id: str):
        if session := self.recording_sessions.get(session_id):
            session.reset()

    def delete_session(self, session_id: str):
        if session_id in self.recording_sessions:
            session = self.recording_sessions[session_id]
            self._stop_azure_recognition(session)
            session.reset()
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
        logger.debug(f"Updated Silero VAD speech probability threshold to {threshold}")

    def set_volume_db_threshold(self, session_id: str, value: float):
        session = self.get_session(session_id)
        session.amplitude_threshold = 32767 * (10 ** (value / 20.0))
