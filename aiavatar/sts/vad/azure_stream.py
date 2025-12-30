import asyncio
from collections import deque
import logging
from typing import AsyncGenerator, Callable, Optional, Dict, Awaitable
# pip install azure-cognitiveservices-speech
import azure.cognitiveservices.speech as speechsdk
from . import SpeechDetector

logger = logging.getLogger(__name__)


class RecordingSession:
    def __init__(self, session_id: str, preroll_buffer_size: int):
        self.session_id = session_id
        self.is_recording: bool = False
        self.buffer: bytearray = bytearray()
        self.record_duration: float = 0
        self.preroll_buffer: deque = deque(maxlen=preroll_buffer_size)
        self.data: dict = {}
        self.on_recording_started_triggered: bool = False
        # Azure Speech recognition
        self.azure_push_stream: Optional[speechsdk.audio.PushAudioInputStream] = None
        self.azure_recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

    def reset(self):
        # Reset status data except for preroll_buffer
        self.buffer.clear()
        self.is_recording = False
        self.record_duration = 0
        self.on_recording_started_triggered = False


class AzureStreamSpeechDetector(SpeechDetector):
    def __init__(
        self,
        *,
        azure_subscription_key: str,
        azure_region: str,
        azure_language: str = "ja-JP",
        silence_duration_threshold: float = 0.5,
        max_duration: float = 20.0,
        sample_rate: int = 16000,
        channels: int = 1,
        preroll_buffer_sec: float = 2.0,
        to_linear16: Optional[Callable[[bytes], bytes]] = None,
        debug: bool = False,
        on_recording_started: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        super().__init__(sample_rate=sample_rate)
        # Azure Speech SDK settings
        self.azure_subscription_key = azure_subscription_key
        self.azure_region = azure_region
        self.azure_language = azure_language

        self.silence_duration_threshold = silence_duration_threshold
        self.max_duration = max_duration
        self.channels = channels
        self.debug = debug
        self.preroll_buffer_sec = preroll_buffer_sec
        self.to_linear16 = to_linear16
        self.recording_sessions: Dict[str, RecordingSession] = {}
        if on_recording_started:
            self._on_recording_started.append(on_recording_started)
        self._on_speech_detecting: Callable[[str, RecordingSession], Awaitable[None]] = None

    def _calculate_preroll_buffer_size(self, chunk_bytes: int) -> int:
        """Calculate preroll buffer size based on chunk size and desired duration"""
        # bytes per second = sample_rate * channels * 2 (16-bit)
        bytes_per_sec = self.sample_rate * self.channels * 2
        # Number of chunks to store for preroll_buffer_sec
        return max(1, int((self.preroll_buffer_sec * bytes_per_sec) / chunk_bytes))

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

            # Set maximum duration for a single recognition segment (minimum 20 seconds required by Azure)
            max_duration_ms = max(20000, int(self.max_duration * 1000))
            speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationMaximumTimeMs,
                str(max_duration_ms)
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
                    logger.info(f"Azure recognizing: {evt.result.text}")

                # Start recording on first recognizing event
                if not session.is_recording:
                    session.is_recording = True
                    # Copy preroll buffer to recording buffer
                    for f in session.preroll_buffer:
                        session.buffer.extend(f)
                    if self.debug:
                        logger.info(f"Recording started: preroll_frames={len(session.preroll_buffer)}, preroll_bytes={len(session.buffer)}")

                if self._on_speech_detecting:
                    try:
                        asyncio.run_coroutine_threadsafe(self._on_speech_detecting(evt.result.text, session), session.event_loop)
                    except Exception as ex:
                        logger.error("Error in on_speech_detecting callback", exc_info=True)

                # Trigger on_recording_started callback on first recognizing event
                if not session.on_recording_started_triggered and self._on_recording_started:
                    session.on_recording_started_triggered = True
                    if self.debug:
                        logger.info(f"on_recording_started triggered: {session.session_id}")
                    for handler in self._on_recording_started:
                        try:
                            asyncio.run_coroutine_threadsafe(handler(session.session_id), session.event_loop)
                        except Exception as ex:
                            logger.error("Error in on_recording_started callback", exc_info=True)

            def on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    if not evt.result.text:
                        if self.debug:
                            logger.info("Azure recognized empty text, skipping")
                        session.reset()
                        return
                    if self.debug:
                        logger.info(f"Azure recognized: {evt.result.text}")
                    recorded_data = bytes(session.buffer)
                    recorded_duration = session.record_duration
                    if self.debug:
                        logger.info(f"Recording finished: {recorded_duration:.2f} sec, {len(recorded_data)} bytes")
                    # Trigger speech detected callback with recorded audio
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.execute_on_speech_detected(recorded_data, evt.result.text, None, recorded_duration, session.session_id),
                            session.event_loop
                        )
                    except Exception as ex:
                        logger.error("Error scheduling execute_on_speech_detected", exc_info=True)
                    session.reset()
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    if self.debug:
                        logger.debug("Azure: No speech recognized")
                    session.reset()

            def on_canceled(evt):
                if self.debug:
                    logger.warning(f"Azure recognition canceled: {evt.reason}")
                session.reset()

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
        else:
            if self.debug:
                logger.warning(f"Azure push stream is None for session {session.session_id}")

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

        session = self.get_session(session_id, len(samples))

        if self.should_mute():
            session.reset()
            session.preroll_buffer.clear()
            if self.debug:
                logger.debug("AzureStreamSpeechDetector is muted.")
            return False

        # Calculate sample duration
        sample_duration = (len(samples) / 2) / (self.sample_rate * self.channels)

        # Always send to Azure stream
        self._write_to_azure_stream(session, samples)

        # Always update preroll buffer (for next speech detection)
        session.preroll_buffer.append(samples)

        # Add to recording buffer if recording
        if session.is_recording:
            session.buffer.extend(samples)
            session.record_duration += sample_duration

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

    def get_session(self, session_id: str, chunk_bytes: int = 512):
        session = self.recording_sessions.get(session_id)
        if session is None:
            preroll_buffer_size = self._calculate_preroll_buffer_size(chunk_bytes)
            session = RecordingSession(session_id, preroll_buffer_size)
            self.recording_sessions[session_id] = session
            # Start Azure recognition when session is created
            self._start_azure_recognition(session)
            if self.debug:
                logger.info(f"Session created: {session_id}, preroll_buffer_size={preroll_buffer_size}")
        return session

    def reset_session(self, session_id: str):
        if session := self.recording_sessions.get(session_id):
            session.reset()

    def delete_session(self, session_id: str):
        if session_id in self.recording_sessions:
            session = self.recording_sessions[session_id]
            self._stop_azure_recognition(session)
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
