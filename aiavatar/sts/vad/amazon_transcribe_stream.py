import asyncio
from collections import deque
import logging
from typing import AsyncGenerator, Callable, Optional, Dict, List, Awaitable
# pip install amazon-transcribe
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from . import SpeechDetector

logger = logging.getLogger(__name__)

MIN_CHUNK_BYTES = 512


class RecordingSession:
    def __init__(self, session_id: str, preroll_buffer_size: int):
        self.session_id = session_id
        self.is_recording: bool = False
        self.buffer: bytearray = bytearray()
        self.record_duration: float = 0
        self.preroll_buffer: deque = deque(maxlen=preroll_buffer_size)
        self.data: dict = {}
        self.on_recording_started_triggered: bool = False
        # Amazon Transcribe Streaming
        self.transcribe_stream = None
        self.event_handler_task: Optional[asyncio.Task] = None
        # Silence duration tracking
        self.accumulated_texts: List[str] = []
        self.silence_timer_task: Optional[asyncio.Task] = None
        self.stream_generation: int = 0

    def cancel_silence_timer(self):
        if self.silence_timer_task and not self.silence_timer_task.done():
            self.silence_timer_task.cancel()
            self.silence_timer_task = None

    def reset(self, reason: str = "unknown", debug: bool = False):
        # Reset status data except for preroll_buffer
        if debug:
            logger.info(
                f"[VAD_DEBUG] reset called: reason={reason}, session={self.session_id}, "
                f"was_recording={self.is_recording}, buffer_bytes={len(self.buffer)}, "
                f"duration={self.record_duration:.2f}s, preroll_frames={len(self.preroll_buffer)}"
            )
        self.cancel_silence_timer()
        self.accumulated_texts.clear()
        self.buffer.clear()
        self.is_recording = False
        self.record_duration = 0
        self.on_recording_started_triggered = False


class _TranscriptionEventHandler(TranscriptResultStreamHandler):
    """Internal event handler for Amazon Transcribe streaming results."""

    def __init__(self, output_stream, detector: "AmazonTranscribeStreamSpeechDetector", session: RecordingSession):
        super().__init__(output_stream)
        self._detector = detector
        self._session = session
        self._generation = session.stream_generation

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        detector = self._detector
        session = self._session

        for result in transcript_event.transcript.results:
            # Ignore events from old stream or after mid-loop stream restart
            if self._generation != session.stream_generation:
                return

            if not result.alternatives:
                continue

            text = result.alternatives[0].transcript
            if not text:
                continue

            if result.is_partial:
                # Partial result
                if detector.debug:
                    logger.info(
                        f"[VAD_DEBUG] on_recognizing (partial): text='{text}', "
                        f"is_recording={session.is_recording}, buffer_bytes={len(session.buffer)}, "
                        f"duration={session.record_duration:.2f}s"
                    )

                # Cancel pending silence timer - user is still speaking
                session.cancel_silence_timer()

                # Start recording on first partial result
                if not session.is_recording:
                    session.is_recording = True
                    # Copy preroll buffer to recording buffer (limit to preroll_buffer_sec)
                    preroll_max_bytes = int(detector.preroll_buffer_sec * detector.sample_rate * detector.channels * 2)
                    total_bytes = 0
                    frames_to_copy = []
                    for f in reversed(session.preroll_buffer):
                        total_bytes += len(f)
                        frames_to_copy.append(f)
                        if total_bytes >= preroll_max_bytes:
                            break
                    for f in reversed(frames_to_copy):
                        session.buffer.extend(f)
                    if detector.debug:
                        logger.info(
                            f"[VAD_DEBUG] recording_started: preroll_total={len(session.preroll_buffer)}, "
                            f"used_frames={len(frames_to_copy)}, preroll_bytes={len(session.buffer)}"
                        )

                # Fire on_speech_detecting callbacks
                if detector._on_speech_detecting:
                    try:
                        detecting_text = "".join(session.accumulated_texts) + text if session.accumulated_texts else text
                        await detector._execute_on_speech_detecting(detecting_text, session)
                    except Exception:
                        logger.error("Error in on_speech_detecting callback", exc_info=True)

                # Trigger on_recording_started callback on first partial result
                if not session.on_recording_started_triggered and detector._on_recording_started:
                    session.on_recording_started_triggered = True
                    if detector.debug:
                        logger.info(f"on_recording_started triggered: {session.session_id}")
                    for handler in detector._on_recording_started:
                        try:
                            asyncio.create_task(handler(session.session_id))
                        except Exception:
                            logger.error("Error in on_recording_started callback", exc_info=True)

            else:
                # Final result
                if detector.debug:
                    logger.info(
                        f"[VAD_DEBUG] on_recognized (final): text='{text}', "
                        f"is_recording={session.is_recording}, buffer_bytes={len(session.buffer)}, "
                        f"duration={session.record_duration:.2f}s"
                    )

                if not text.strip():
                    if detector.debug:
                        logger.info("[VAD_DEBUG] recognized empty text, skipping")
                    if not session.accumulated_texts:
                        session.reset(reason="recognized_empty", debug=detector.debug)
                    continue

                # Accumulate final text
                session.accumulated_texts.append(text)
                session.cancel_silence_timer()

                if detector.debug:
                    logger.info(
                        f"[VAD_DEBUG] accumulated text segment {len(session.accumulated_texts)}: '{text}'"
                    )

                if detector.silence_duration_threshold > 0:
                    # Wait for silence_duration_threshold before triggering
                    session.silence_timer_task = asyncio.create_task(
                        self._on_silence_timeout()
                    )
                else:
                    # No threshold - trigger immediately (original behavior)
                    await self._trigger_speech_detected()

    async def _on_silence_timeout(self):
        """Called when silence exceeds threshold after a final result."""
        try:
            await asyncio.sleep(self._detector.silence_duration_threshold)
            await self._trigger_speech_detected()
        except asyncio.CancelledError:
            pass

    async def _trigger_speech_detected(self):
        """Trigger speech detected with accumulated text."""
        detector = self._detector
        session = self._session

        if not session.accumulated_texts:
            session.reset(reason="no_accumulated_text", debug=detector.debug)
            return

        combined_text = "".join(session.accumulated_texts)
        recorded_data = bytes(session.buffer)
        recorded_duration = session.record_duration

        if detector.debug:
            logger.info(
                f"[VAD_DEBUG] speech_detected: text='{combined_text}', "
                f"segments={len(session.accumulated_texts)}, "
                f"buffer_bytes={len(recorded_data)}, duration={recorded_duration:.2f}s"
            )

        try:
            if detector._validate_recognized_text:
                if validation := detector._validate_recognized_text(combined_text):
                    if detector.debug:
                        logger.info(f"Invalid recognized text: {combined_text} / validation: {validation}")
                    return
            asyncio.create_task(
                detector.execute_on_speech_detected(
                    recorded_data, combined_text, None, recorded_duration, session.session_id
                )
            )
        except Exception:
            logger.error("Error scheduling execute_on_speech_detected", exc_info=True)
        finally:
            session.reset(reason="recognized_speech", debug=detector.debug)
            # Restart Transcribe stream to prevent text carryover to next turn
            session.stream_generation += 1
            old_stream = session.transcribe_stream
            old_task = session.event_handler_task
            session.transcribe_stream = None
            session.event_handler_task = None
            if old_stream or old_task:
                asyncio.create_task(
                    detector._cleanup_old_transcribe_stream(old_stream, old_task, session.session_id)
                )


class AmazonTranscribeStreamSpeechDetector(SpeechDetector):
    def __init__(
        self,
        *,
        aws_region: str = "ap-northeast-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_language: str = "ja-JP",
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
        # Amazon Transcribe settings
        self.aws_region = aws_region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_language = aws_language

        self.silence_duration_threshold = silence_duration_threshold
        self.max_duration = max_duration
        self.channels = channels
        self.debug = debug
        self.debug_deeper = False
        self.preroll_buffer_sec = preroll_buffer_sec
        self.to_linear16 = to_linear16
        self.recording_sessions: Dict[str, RecordingSession] = {}
        if on_recording_started:
            self._on_recording_started.append(on_recording_started)
        self._on_speech_detecting: List[Callable[[str, RecordingSession], Awaitable[None]]] = []
        self._on_speech_recognition_error: List[Callable[[Exception, str], Awaitable[None]]] = []
        self._validate_recognized_text: Callable[[str], Optional[str]] = None

    def get_config(self) -> dict:
        return {
            "aws_language": self.aws_language,
            "silence_duration_threshold": self.silence_duration_threshold,
            "max_duration": self.max_duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "preroll_buffer_sec": self.preroll_buffer_sec,
            "debug": self.debug,
            "debug_deeper": self.debug_deeper
        }

    def _calculate_preroll_buffer_size(self) -> int:
        """Calculate preroll buffer size based on a conservative minimum chunk size"""
        bytes_per_sec = self.sample_rate * self.channels * 2
        return max(1, int((self.preroll_buffer_sec * bytes_per_sec) / MIN_CHUNK_BYTES))

    def _create_credential_resolver(self):
        """Create credential resolver from explicit credentials if provided."""
        if self.aws_access_key_id and self.aws_secret_access_key:
            from amazon_transcribe.auth import StaticCredentialResolver
            return StaticCredentialResolver(
                access_key_id=self.aws_access_key_id,
                secret_access_key=self.aws_secret_access_key,
                session_token=self.aws_session_token,
            )
        return None  # Use default credential chain

    async def _start_transcribe_streaming(self, session: RecordingSession):
        """Start Amazon Transcribe streaming for a session"""
        try:
            client = TranscribeStreamingClient(
                region=self.aws_region,
                credential_resolver=self._create_credential_resolver(),
            )

            stream = await client.start_stream_transcription(
                language_code=self.aws_language,
                media_sample_rate_hz=self.sample_rate,
                media_encoding="pcm",
            )
            session.transcribe_stream = stream

            handler = _TranscriptionEventHandler(stream.output_stream, self, session)

            async def _run_handler():
                try:
                    await handler.handle_events()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in transcription event handler: {e}", exc_info=True)
                    if self._on_speech_recognition_error:
                        await self._execute_on_speech_recognition_error(
                            Exception(f"Amazon Transcribe streaming error: {e}"),
                            session.session_id
                        )

            session.event_handler_task = asyncio.create_task(_run_handler())

            if self.debug:
                logger.info(f"Amazon Transcribe streaming started for session {session.session_id}")

        except Exception as e:
            logger.error(f"Failed to start Amazon Transcribe streaming: {e}")
            if self._on_speech_recognition_error:
                await self._execute_on_speech_recognition_error(
                    Exception(f"Failed to start Amazon Transcribe streaming: {e}"),
                    session.session_id
                )

    async def _stop_transcribe_streaming(self, session: RecordingSession):
        """Stop Amazon Transcribe streaming for a session"""
        try:
            session.cancel_silence_timer()

            # End the input stream first and wait for the event handler to
            # finish naturally, so AWS CRT doesn't deliver data to cancelled futures.
            if session.transcribe_stream:
                try:
                    await session.transcribe_stream.input_stream.end_stream()
                except Exception:
                    pass
                session.transcribe_stream = None

            if session.event_handler_task:
                try:
                    await asyncio.wait_for(session.event_handler_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    session.event_handler_task.cancel()
                session.event_handler_task = None

            if self.debug:
                logger.info(f"Amazon Transcribe streaming stopped for session {session.session_id}")

        except Exception as e:
            logger.error(f"Failed to stop Amazon Transcribe streaming: {e}")

    async def _cleanup_old_transcribe_stream(self, old_stream, old_task, session_id: str):
        """Clean up old Transcribe stream resources in the background."""
        if old_stream:
            try:
                await old_stream.input_stream.end_stream()
            except Exception:
                pass
        if old_task:
            try:
                await asyncio.wait_for(old_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                old_task.cancel()
        if self.debug:
            logger.info(f"Old Transcribe stream cleaned up for session {session_id}")

    async def _write_to_transcribe_stream(self, session: RecordingSession, audio_data: bytes):
        """Write audio data to Amazon Transcribe stream"""
        if session.transcribe_stream:
            try:
                await session.transcribe_stream.input_stream.send_audio_event(audio_chunk=audio_data)
            except Exception as e:
                logger.error(f"Failed to write to Transcribe stream: {e}")
        else:
            if self.debug:
                logger.warning(f"Transcribe stream is None for session {session.session_id}")

    def on_speech_detecting(self, func: Callable[[str, RecordingSession], Awaitable[None]]) -> Callable[[str, RecordingSession], Awaitable[None]]:
        """Register callback for speech detecting (partial results).

        The callback is called with (text: str, session: RecordingSession).
        """
        self._on_speech_detecting.append(func)
        return func

    def validate_recognized_text(self, func):
        self._validate_recognized_text = func
        return func

    def on_speech_recognition_error(self, func: Callable[[Exception, str], Awaitable[None]]) -> Callable[[Exception, str], Awaitable[None]]:
        """Register callback for speech recognition errors.

        The callback is called with (error: Exception, session_id: str).
        """
        self._on_speech_recognition_error.append(func)
        return func

    async def _execute_on_speech_detecting(self, text: str, session: RecordingSession):
        """Execute on_speech_detecting callbacks."""
        for handler in self._on_speech_detecting:
            try:
                await handler(text, session)
            except Exception:
                logger.error("Error in on_speech_detecting callback", exc_info=True)

    async def _execute_on_speech_recognition_error(self, error: Exception, session_id: str):
        """Execute on_speech_recognition_error callbacks."""
        for handler in self._on_speech_recognition_error:
            try:
                await handler(error, session_id)
            except Exception:
                logger.error("Error in on_speech_recognition_error callback", exc_info=True)

    async def execute_on_speech_detected(self, recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        await self._execute_on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id)

    async def process_samples(self, samples: bytes, session_id: str) -> bool:
        if self.to_linear16:
            samples = self.to_linear16(samples)

        if self.debug and self.debug_deeper:
            logger.info(f"process_samples: session_id={session_id}, should_mute={self.should_mute()}")

        session = self.get_session(session_id)

        # Start streaming if not yet started
        if session.transcribe_stream is None:
            await self._start_transcribe_streaming(session)

        if self.should_mute():
            session.reset(reason="muted", debug=self.debug)
            session.preroll_buffer.clear()
            return False

        # Calculate sample duration
        sample_duration = (len(samples) / 2) / (self.sample_rate * self.channels)

        # Always send to Transcribe stream
        if self.debug and self.debug_deeper:
            logger.info(f"Send samples to Transcribe: session_id={session_id}, samples={len(samples)}")
        await self._write_to_transcribe_stream(session, samples)

        # Always update preroll buffer (for next speech detection)
        session.preroll_buffer.append(samples)

        # Add to recording buffer if recording
        if session.is_recording:
            session.buffer.extend(samples)
            session.record_duration += sample_duration

            # Check max_duration
            if self.max_duration > 0 and session.record_duration >= self.max_duration:
                if session.accumulated_texts:
                    session.cancel_silence_timer()
                    combined_text = "".join(session.accumulated_texts)
                    recorded_data = bytes(session.buffer)
                    recorded_duration = session.record_duration
                    if self.debug:
                        logger.info(
                            f"[VAD_DEBUG] max_duration reached: text='{combined_text}', "
                            f"segments={len(session.accumulated_texts)}, "
                            f"buffer_bytes={len(recorded_data)}, duration={recorded_duration:.2f}s"
                        )
                    try:
                        if self._validate_recognized_text:
                            if validation := self._validate_recognized_text(combined_text):
                                if self.debug:
                                    logger.info(f"Invalid recognized text at max_duration: {combined_text} / validation: {validation}")
                                session.reset(reason="max_duration_invalid_text", debug=self.debug)
                                return session.is_recording
                        asyncio.create_task(
                            self.execute_on_speech_detected(
                                recorded_data, combined_text, None, recorded_duration, session_id
                            )
                        )
                    except Exception:
                        logger.error("Error scheduling execute_on_speech_detected", exc_info=True)
                    session.reset(reason="max_duration", debug=self.debug)
                    # Restart Transcribe stream to prevent text carryover
                    session.stream_generation += 1
                    old_stream = session.transcribe_stream
                    old_task = session.event_handler_task
                    session.transcribe_stream = None
                    session.event_handler_task = None
                    if old_stream or old_task:
                        asyncio.create_task(
                            self._cleanup_old_transcribe_stream(old_stream, old_task, session_id)
                        )
                else:
                    if self.debug:
                        logger.info(f"[VAD_DEBUG] max_duration reached but no accumulated text, resetting")
                    session.reset(reason="max_duration_no_text", debug=self.debug)

        if self.debug and session.is_recording and int(session.record_duration * 10) % 50 == 0:
            logger.info(
                f"[VAD_DEBUG] process_samples: is_recording={session.is_recording}, "
                f"buffer_bytes={len(session.buffer)}, duration={session.record_duration:.2f}s"
            )

        return session.is_recording

    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str):
        logger.info("AmazonTranscribeStreamSpeechDetector start processing stream.")

        async for data in input_stream:
            if not data:
                break
            await self.process_samples(data, session_id)
            await asyncio.sleep(0.0001)

        await self.delete_session(session_id)

        logger.info("AmazonTranscribeStreamSpeechDetector finish processing stream.")

    async def finalize_session(self, session_id):
        await self.delete_session(session_id)

    def get_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if session is None:
            preroll_buffer_size = self._calculate_preroll_buffer_size()
            session = RecordingSession(session_id, preroll_buffer_size)
            self.recording_sessions[session_id] = session
            if self.debug:
                logger.info(f"Session created: {session_id}, preroll_buffer_size={preroll_buffer_size}")
        return session

    def reset_session(self, session_id: str):
        if session := self.recording_sessions.get(session_id):
            session.reset(reason="reset_session", debug=self.debug)

    async def delete_session(self, session_id: str):
        if session_id in self.recording_sessions:
            session = self.recording_sessions[session_id]
            await self._stop_transcribe_streaming(session)
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
