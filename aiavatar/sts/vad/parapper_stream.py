"""Speech detector backed by Parapper-ASR streaming recognition protocol v1."""

import asyncio
from collections import deque
import json
import logging
import time
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional
import uuid

import websockets

from . import SpeechDetector

logger = logging.getLogger(__name__)

MAX_AUDIO_FRAME_BYTES = 3200
PREROLL_CHUNK_BYTES = 512


class ParapperRecognitionError(Exception):
    """An error reported by the Parapper-ASR protocol."""

    def __init__(self, code: str, message: str, fatal: bool = True):
        super().__init__(f"Parapper-ASR error ({code}): {message}")
        self.code = code
        self.message = message
        self.fatal = fatal


class RecordingSession:
    def __init__(self, session_id: str, protocol_session_id: str, preroll_size: int):
        self.session_id = session_id
        self.protocol_session_id = protocol_session_id
        self.is_recording = False
        self.buffer = bytearray()
        self.record_duration = 0.0
        self.preroll_buffer = deque(maxlen=preroll_size)
        self.data: dict = {}
        self.on_recording_started_triggered = False
        self.websocket = None
        self.receiver_task: Optional[asyncio.Task] = None
        self.ready_task: Optional[asyncio.Task] = None
        self.ready = asyncio.Event()
        self.done = asyncio.Event()
        self.send_lock = asyncio.Lock()
        self.closed = False
        self.ai_speaking_vclock = 0.0
        self.ai_speaking_until = 0.0

    @property
    def is_ai_speaking(self):
        return time.monotonic() < self.ai_speaking_until

    def reset(self):
        self.is_recording = False
        self.buffer.clear()
        self.record_duration = 0.0
        self.on_recording_started_triggered = False


class ParapperStreamSpeechDetector(SpeechDetector):
    """Use Parapper-ASR's server-side VAD and turn-end decision.

    ``turn.final`` is trusted as the sole turn-end signal. This component does
    not apply a client-side silence timeout or otherwise re-segment turns.
    """

    def __init__(
        self,
        *,
        url: str = "ws://127.0.0.1:8080/ws/recognition",
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        preroll_buffer_sec: float = 2.0,
        to_linear16: Optional[Callable[[bytes], bytes]] = None,
        connect_timeout: float = 10.0,
        drain_timeout: float = 10.0,
        debug: bool = False,
        on_recording_started: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        if sample_rate != 16000:
            raise ValueError("Parapper protocol v1 requires sample_rate=16000")
        if channels != 1:
            raise ValueError("Parapper protocol v1 requires channels=1")
        super().__init__(sample_rate=sample_rate)
        logger.warning(
            "ParapperStreamSpeechDetector is experimental and its API may change."
        )
        self.url = url
        self.api_key = api_key
        self.channels = channels
        self.preroll_buffer_sec = preroll_buffer_sec
        self.to_linear16 = to_linear16
        self.connect_timeout = connect_timeout
        self.drain_timeout = drain_timeout
        self.debug = debug
        self.debug_deeper = False
        self.recording_sessions: Dict[str, RecordingSession] = {}
        self._on_speech_detecting: List[Callable[[str, RecordingSession], Awaitable[None]]] = []
        self._on_speech_recognition_error: List[Callable[[Exception, str], Awaitable[None]]] = []
        self._validate_recognized_text: Optional[Callable[[str], Optional[str]]] = None
        if on_recording_started:
            self._on_recording_started.append(on_recording_started)

    def get_config(self) -> dict:
        return {
            "url": self.url,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "preroll_buffer_sec": self.preroll_buffer_sec,
            "connect_timeout": self.connect_timeout,
            "drain_timeout": self.drain_timeout,
            "debug": self.debug,
            "debug_deeper": self.debug_deeper,
        }

    def on_speech_detecting(self, func):
        self._on_speech_detecting.append(func)
        return func

    def on_speech_recognition_error(self, func):
        self._on_speech_recognition_error.append(func)
        return func

    def validate_recognized_text(self, func):
        self._validate_recognized_text = func
        return func

    async def _execute_on_speech_recognition_error(self, error, session_id):
        for handler in self._on_speech_recognition_error:
            try:
                await handler(error, session_id)
            except Exception:
                logger.error("Error in recognition error callback", exc_info=True)

    async def _connect(self, session: RecordingSession):
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            session.websocket = await websockets.connect(
                self.url, additional_headers=headers, open_timeout=self.connect_timeout
            )
            session.receiver_task = asyncio.create_task(self._receive(session))
            await session.websocket.send(json.dumps({
                "version": 1,
                "type": "session.start",
                "session_id": session.protocol_session_id,
                "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
            }))
            await asyncio.wait_for(session.ready.wait(), self.connect_timeout)
        except Exception as exc:
            session.ready.set()
            await self._report_error(exc, session.session_id)
            await self._close(session)
            raise

    async def _report_error(self, error: Exception, session_id: str):
        logger.error("Parapper-ASR session %s failed: %s", session_id, error)
        if self._on_speech_recognition_error:
            await self._execute_on_speech_recognition_error(error, session_id)

    async def _receive(self, session: RecordingSession):
        try:
            async for frame in session.websocket:
                if not isinstance(frame, str):
                    raise ParapperRecognitionError("invalid_json", "server sent a binary frame")
                message = json.loads(frame)
                await self._handle_message(session, message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not session.closed:
                await self._report_error(exc, session.session_id)
        finally:
            session.done.set()

    async def _handle_message(self, session: RecordingSession, message: dict):
        message_type = message.get("type")
        if message.get("version") != 1:
            raise ParapperRecognitionError("unsupported_version", "server message is not protocol v1")
        if message_type == "session.ready":
            session.ready.set()
        elif message_type == "speech.started":
            await self._start_recording(session)
        elif message_type == "turn.partial":
            text = message.get("text", "")
            if text:
                if not session.is_recording:
                    await self._start_recording(session)
                for handler in self._on_speech_detecting:
                    try:
                        await handler(text, session)
                    except Exception:
                        logger.error("Error in speech detecting callback", exc_info=True)
        elif message_type == "turn.final":
            await self._handle_final(session, message)
        elif message_type in ("session.done", "session.cancelled"):
            session.done.set()
        elif message_type == "error":
            raise ParapperRecognitionError(
                message.get("code", "recognition_failed"),
                message.get("message", "unknown error"),
                message.get("fatal", True),
            )

    async def _start_recording(self, session: RecordingSession):
        if session.is_recording:
            return
        session.is_recording = True
        max_bytes = int(self.preroll_buffer_sec * self.sample_rate * self.channels * 2)
        frames, total = [], 0
        for frame in reversed(session.preroll_buffer):
            frames.append(frame)
            total += len(frame)
            if total >= max_bytes:
                break
        for frame in reversed(frames):
            session.buffer.extend(frame)
        if not session.on_recording_started_triggered:
            session.on_recording_started_triggered = True
            for handler in self._on_recording_started:
                try:
                    await handler(session.session_id)
                except Exception:
                    logger.error("Error in recording started callback", exc_info=True)

    async def _handle_final(self, session: RecordingSession, message: dict):
        text = message.get("text", "")
        if not text:
            session.reset()
            return
        if self._validate_recognized_text:
            if validation := self._validate_recognized_text(text):
                if self.debug:
                    logger.info("Invalid recognized text: %s / validation: %s", text, validation)
                session.reset()
                return
        duration = message.get("audio_duration_ms")
        recorded_duration = duration / 1000 if isinstance(duration, (int, float)) else session.record_duration
        metadata = {key: value for key, value in message.items() if key not in {"version", "type", "text"}}
        try:
            await self._execute_on_speech_detected(
                bytes(session.buffer), text, metadata, recorded_duration, session.session_id
            )
        finally:
            session.reset()

    def get_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if session is None:
            preroll_size = max(1, int(self.preroll_buffer_sec * self.sample_rate * 2 / PREROLL_CHUNK_BYTES))
            session = RecordingSession(session_id, f"{session_id}-{uuid.uuid4().hex}", preroll_size)
            self.recording_sessions[session_id] = session
            session.ready_task = asyncio.create_task(self._connect(session))
        return session

    async def process_samples(self, samples: bytes, session_id: str) -> bool:
        if self.to_linear16:
            samples = self.to_linear16(samples)
        session = self.get_session(session_id)
        if self.should_mute():
            session.reset()
            session.preroll_buffer.clear()
            return False
        await session.ready_task
        if len(samples) % 2:
            raise ValueError("PCM s16le audio must have an even byte length")
        sample_duration = len(samples) / (2 * self.sample_rate * self.channels)
        async with session.send_lock:
            for offset in range(0, len(samples), MAX_AUDIO_FRAME_BYTES):
                frame = samples[offset:offset + MAX_AUDIO_FRAME_BYTES]
                if frame:
                    await session.websocket.send(frame)
        session.preroll_buffer.append(samples)
        if session.is_recording:
            session.buffer.extend(samples)
            session.record_duration += sample_duration
        return session.is_recording

    async def finalize_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if not session:
            return
        try:
            await session.ready_task
            await session.websocket.send(json.dumps({
                "version": 1, "type": "session.stop", "session_id": session.protocol_session_id
            }))
            await asyncio.wait_for(session.done.wait(), self.drain_timeout)
        finally:
            await self._close(session)
            self.recording_sessions.pop(session_id, None)

    async def _close(self, session: RecordingSession):
        session.closed = True
        if session.websocket:
            await session.websocket.close()
        if session.receiver_task and session.receiver_task is not asyncio.current_task():
            session.receiver_task.cancel()
            await asyncio.gather(session.receiver_task, return_exceptions=True)

    async def process_stream(self, input_stream: AsyncGenerator[bytes, None], session_id: str):
        try:
            async for data in input_stream:
                if not data:
                    break
                await self.process_samples(data, session_id)
        finally:
            await self.finalize_session(session_id)

    def reset_session(self, session_id: str):
        if session := self.recording_sessions.get(session_id):
            session.reset()

    def delete_session(self, session_id: str):
        session = self.recording_sessions.pop(session_id, None)
        if session:
            asyncio.create_task(self._close(session))

    def get_session_data(self, session_id: str, key: str):
        session = self.recording_sessions.get(session_id)
        return session.data.get(key) if session else None

    def set_session_data(self, session_id: str, key: str, value, create_session: bool = False):
        session = self.get_session(session_id) if create_session else self.recording_sessions.get(session_id)
        if session:
            session.data[key] = value
