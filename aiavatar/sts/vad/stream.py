import asyncio
import logging
import struct
from typing import Callable, Optional, Dict, Awaitable
from .silero import SileroSpeechDetector, RecordingSession as SileroRecordingSession
from ..stt.base import SpeechRecognizer

logger = logging.getLogger(__name__)


class RecordingSession(SileroRecordingSession):
    def __init__(self, session_id: str, preroll_buffer_count: int = 5, vad_iterator=None):
        super().__init__(session_id, preroll_buffer_count, vad_iterator)
        # Segment tracking for on_speech_detecting hook
        self.segment_buffer: bytearray = bytearray()
        self.segment_duration: float = 0
        self.segment_silence_duration: float = 0  # Silence duration within segment
        self.segment_fired: bool = False  # Prevent consecutive firing
        # Recognition task tracking
        self.pending_recognition_task: Optional[asyncio.Task] = None
        self.recognition_sequence: int = 0  # Sequence number for ordering

    def reset(self):
        super().reset()
        # Reset segment tracking
        self.segment_buffer.clear()
        self.segment_duration = 0
        self.segment_silence_duration = 0
        self.segment_fired = False
        self.pending_recognition_task = None
        self.recognition_sequence = 0


class SileroStreamSpeechDetector(SileroSpeechDetector):
    def __init__(
        self,
        *,
        speech_recognizer: SpeechRecognizer,
        volume_db_threshold: Optional[float] = None,
        silence_duration_threshold: float = 0.5,
        segment_silence_threshold: float = 0.2,
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
        on_recording_started: Optional[Callable[[str], Awaitable[None]]] = None,
        on_recording_started_min_duration: float = 1.5,
        on_recording_started_min_text_length: int = 2,
        use_vad_iterator: bool = False
    ):
        super().__init__(
            volume_db_threshold=volume_db_threshold,
            silence_duration_threshold=silence_duration_threshold,
            max_duration=max_duration,
            min_duration=min_duration,
            sample_rate=sample_rate,
            channels=channels,
            preroll_buffer_count=preroll_buffer_count,
            to_linear16=to_linear16,
            debug=debug,
            model_path=model_path,
            speech_probability_threshold=speech_probability_threshold,
            chunk_size=chunk_size,
            model_pool_size=model_pool_size,
            on_recording_started=on_recording_started,
            on_recording_started_min_duration=on_recording_started_min_duration,
            use_vad_iterator=use_vad_iterator
        )
        self.speech_recognizer = speech_recognizer
        self.segment_silence_threshold = segment_silence_threshold
        self.on_recording_started_min_text_length = on_recording_started_min_text_length
        self.recording_sessions: Dict[str, RecordingSession] = {}
        self._on_speech_detecting: Optional[Callable[[str, RecordingSession], Awaitable[None]]] = None
        self._on_speech_recognition_error: Optional[Callable[[Exception, str], Awaitable[None]]] = None
        self._validate_recognized_text: Optional[Callable[[str], Optional[str]]] = None

    def on_speech_detecting(self, func: Callable[[str, RecordingSession], Awaitable[None]]):
        self._on_speech_detecting = func
        return func

    def validate_recognized_text(self, func: Callable[[str], Optional[str]]):
        self._validate_recognized_text = func
        return func

    def on_speech_recognition_error(self, func: Callable[[Exception, str], Awaitable[None]]):
        self._on_speech_recognition_error = func
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
            logger.debug("SileroStreamSpeechDetector is muted.")
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
            speech_detected = self._detect_speech_silero(vad_chunk, session)
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

            # Always add to segment buffer (need silence for accurate recognition)
            session.segment_buffer.extend(samples)
            session.segment_duration += sample_duration

            if speech_detected:
                session.silence_duration = 0
                session.segment_silence_duration = 0
                # Reset fired flag when speech resumes
                session.segment_fired = False
            else:
                session.silence_duration += sample_duration
                session.segment_silence_duration += sample_duration

            # Detect segment end: silence exceeded threshold, has content, and not already fired
            if (session.segment_silence_duration >= self.segment_silence_threshold and
                len(session.segment_buffer) > 0 and
                not session.segment_fired
            ):
                session.segment_fired = True
                segment_data = bytes(session.segment_buffer)
                # Increment sequence number for this recognition
                session.recognition_sequence += 1
                current_seq = session.recognition_sequence
                # Run recognition on segment
                async def _run_segment_recognition(data: bytes, sess: RecordingSession, seq: int):
                    try:
                        result = await self.speech_recognizer.recognize(sess.session_id, data)
                        # Only update if this is still the latest sequence
                        if result.text and seq == sess.recognition_sequence:
                            sess.last_recognized_text = result.text
                            if self._on_speech_detecting:
                                await self._on_speech_detecting(result.text, sess)
                            # Check on_recording_started trigger condition after recognition
                            await self._check_and_trigger_recording_started(sess)
                    except Exception as ex:
                        logger.error("Error in segment recognition", exc_info=True)
                        if self._on_speech_recognition_error:
                            try:
                                await self._on_speech_recognition_error(ex, sess.session_id)
                            except Exception as callback_ex:
                                logger.error(f"Error in on_speech_recognition_error callback: {callback_ex}", exc_info=True)
                session.pending_recognition_task = asyncio.create_task(_run_segment_recognition(segment_data, session, current_seq))

            # Check on_recording_started trigger condition (without text, duration-based only)
            await self._check_and_trigger_recording_started(session)

            if session.silence_duration >= self.silence_duration_threshold:
                recorded_duration = session.record_duration - session.silence_duration
                if recorded_duration < self.min_duration:
                    if self.debug:
                        logger.info(f"Recording too short: {recorded_duration} sec")
                else:
                    if self.debug:
                        logger.info(f"Recording finished: {recorded_duration} sec")

                    # Wait for pending recognition task to complete
                    if session.pending_recognition_task is not None:
                        try:
                            await session.pending_recognition_task
                        except Exception as ex:
                            logger.error("Error waiting for pending recognition", exc_info=True)

                    # Use last recognized text if available, otherwise run final recognition
                    final_text = session.last_recognized_text
                    if final_text is None:
                        # No segment was recognized, run recognition on full buffer
                        try:
                            result = await self.speech_recognizer.recognize(session.session_id, bytes(session.buffer))
                            final_text = result.text
                        except Exception as ex:
                            logger.error("Error in final recognition", exc_info=True)
                            if self._on_speech_recognition_error:
                                try:
                                    await self._on_speech_recognition_error(ex, session.session_id)
                                except Exception as callback_ex:
                                    logger.error(f"Error in on_speech_recognition_error callback: {callback_ex}", exc_info=True)

                    if final_text:
                        if self._validate_recognized_text:
                            if validation := self._validate_recognized_text(final_text):
                                if self.debug:
                                    logger.info(f"Invalid recognized text: {final_text} / validation: {validation}")
                                session.reset()
                                return session.is_recording

                        recorded_data = bytes(session.buffer)
                        asyncio.create_task(self.execute_on_speech_detected(recorded_data, final_text, None, recorded_duration, session.session_id))
                    else:
                        if self.debug:
                            logger.info("No text recognized, skipping")
                session.reset()

            elif session.record_duration >= self.max_duration:
                if self.debug:
                    logger.info(f"Recording max duration reached: {session.record_duration} sec")

                # Wait for pending recognition task to complete
                if session.pending_recognition_task is not None:
                    try:
                        await session.pending_recognition_task
                    except Exception as ex:
                        logger.error("Error waiting for pending recognition", exc_info=True)

                # Use last recognized text if available
                final_text = session.last_recognized_text
                if final_text:
                    if self._validate_recognized_text:
                        if validation := self._validate_recognized_text(final_text):
                            if self.debug:
                                logger.info(f"Invalid recognized text: {final_text} / validation: {validation}")
                            session.reset()
                            return session.is_recording

                    recorded_data = bytes(session.buffer)
                    asyncio.create_task(self.execute_on_speech_detected(recorded_data, final_text, None, session.record_duration, session.session_id))
                else:
                    if self.debug:
                        logger.info("No text recognized at max duration, skipping")
                session.reset()

        return session.is_recording

    def get_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if session is None:
            # Create VAD iterator for this session using assigned model
            model, _ = self._get_model_and_lock(session_id)
            vad_iterator = self.VADIteratorClass(
                model,
                threshold=self.speech_probability_threshold,
                sampling_rate=self.sample_rate
            )
            session = RecordingSession(session_id, self.preroll_buffer_count, vad_iterator)
            self.recording_sessions[session_id] = session
        if session.amplitude_threshold is None:
            session.amplitude_threshold = self.amplitude_threshold
        return session
