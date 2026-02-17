import asyncio
import base64
import logging
from typing import Dict, Callable, Awaitable, Optional
from fastapi import APIRouter, WebSocket, WebSocketException, status
from ...sts.vad import SpeechDetector
from ...sts.vad.stream import SileroStreamSpeechDetector
from ...sts.stt import SpeechRecognizer
from .models import STTRequest, STTResponse

logger = logging.getLogger(__name__)


class SpeechRecognitionSessionData:
    def __init__(self):
        self.id: Optional[str] = None
        self.data: Dict = {}


class StreamSpeechRecognitionServer:
    """
    WebSocket server for streaming speech recognition.

    For stream VAD (SileroStreamSpeechDetector):
    - Sends partial results via on_speech_detecting callback
    - Sends final result when speech ends

    For non-stream VAD:
    - Buffers audio until speech detection completes
    - Sends final result with batch recognition
    """

    def __init__(
        self,
        *,
        vad: SpeechDetector,
        stt: SpeechRecognizer = None,
        api_key: str = None,
        debug: bool = False
    ):
        self.vad = vad
        self.stt = stt
        self.api_key = api_key
        self.debug = debug

        self.websockets: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, SpeechRecognitionSessionData] = {}

        # Callbacks
        self._on_connect: Optional[Callable[[STTRequest, SpeechRecognitionSessionData], Awaitable[None]]] = None
        self._on_disconnect: Optional[Callable[[SpeechRecognitionSessionData], Awaitable[None]]] = None

        # Check if VAD is stream type
        self._is_stream_vad = isinstance(vad, SileroStreamSpeechDetector)

        # Setup VAD callbacks
        self._setup_vad_callbacks()

    def _setup_vad_callbacks(self):
        """Setup callbacks for VAD events."""

        # Voice activity callback (common for all VAD types)
        @self.vad.on_voiced
        async def on_voiced(session_id: str):
            if session_id in self.websockets:
                await self._send_response(STTResponse(
                    type="voiced",
                    session_id=session_id
                ))

        if self._is_stream_vad:
            # Stream VAD: send partial results during recognition
            @self.vad.on_speech_detecting
            async def on_speech_detecting(text: str, session):
                session_id = session.session_id
                if session_id in self.websockets:
                    await self._send_response(STTResponse(
                        type="partial",
                        session_id=session_id,
                        text=text,
                        is_final=False
                    ))

            # Stream VAD: send final result
            @self.vad.on_speech_detected
            async def on_speech_detected_stream(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
                if session_id in self.websockets:
                    await self._send_response(STTResponse(
                        type="final",
                        session_id=session_id,
                        text=text,
                        is_final=True,
                        metadata={"duration": recorded_duration}
                    ))

            # Stream VAD: send error
            @self.vad.on_speech_recognition_error
            async def on_speech_recognition_error_stream(error: Exception, session_id: str):
                if session_id in self.websockets:
                    await self._send_response(STTResponse(
                        type="error",
                        session_id=session_id,
                        text=str(error),
                        is_final=False
                    ))
        else:
            # Non-stream VAD: batch recognition after speech ends
            @self.vad.on_speech_detected
            async def on_speech_detected_batch(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
                if session_id not in self.websockets:
                    return

                # Run STT on recorded audio
                if self.stt:
                    try:
                        result = await self.stt.recognize(session_id, recorded_data)
                        recognized_text = result.text
                    except Exception as ex:
                        logger.error(f"Error in batch recognition: {ex}", exc_info=True)
                        await self._send_response(STTResponse(
                            type="error",
                            session_id=session_id,
                            text=str(ex),
                            is_final=False
                        ))
                        return
                else:
                    recognized_text = text  # Use text from VAD if available

                await self._send_response(STTResponse(
                    type="final",
                    session_id=session_id,
                    text=recognized_text,
                    is_final=True,
                    metadata={"duration": recorded_duration}
                ))

    def on_connect(self, func: Callable[[STTRequest, SpeechRecognitionSessionData], Awaitable[None]]):
        self._on_connect = func
        return func

    def on_disconnect(self, func: Callable[[SpeechRecognitionSessionData], Awaitable[None]]):
        self._on_disconnect = func
        return func

    async def _send_response(self, response: STTResponse):
        """Send response to WebSocket client."""
        if response.session_id in self.websockets:
            try:
                await self.websockets[response.session_id].send_text(
                    response.model_dump_json()
                )
            except Exception as ex:
                logger.error(f"Error sending response: {ex}")

    async def process_websocket(self, websocket: WebSocket, session_data: SpeechRecognitionSessionData):
        """Process incoming WebSocket message."""
        data = await websocket.receive_text()
        request = STTRequest.model_validate_json(data)

        if not request.session_id:
            await websocket.send_text(STTResponse(
                type="error",
                session_id=request.session_id,
                metadata={"error": "session_id is required"}
            ).model_dump_json())
            logger.info("WebSocket disconnect: session_id is required.")
            await websocket.close()
            return

        if request.type == "start":
            self.websockets[request.session_id] = websocket
            session_data.id = request.session_id
            session_data.data["metadata"] = request.metadata
            self.sessions[session_data.id] = session_data

            logger.info(f"STT WebSocket connected: session={request.session_id}")

            await self._send_response(STTResponse(
                type="connected",
                session_id=request.session_id
            ))

            if self._on_connect:
                asyncio.create_task(self._on_connect(request, session_data))

        elif request.type == "data":
            if request.audio_data:
                audio_data = base64.b64decode(request.audio_data)
                await self.vad.process_samples(audio_data, request.session_id)

        elif request.type == "stop":
            logger.info(f"STT WebSocket disconnect: session={request.session_id}")
            await websocket.close()

    async def finalize_session(self, session_id: str):
        """Clean up session resources."""
        if hasattr(self.vad, "finalize_session"):
            await self.vad.finalize_session(session_id)
        elif hasattr(self.vad, "delete_session"):
            self.vad.delete_session(session_id)

    def _authenticate_websocket(self, websocket: WebSocket) -> Optional[str]:
        """Authenticate WebSocket connection using Authorization header or Sec-WebSocket-Protocol.

        Returns the accepted subprotocol name if authenticated via Sec-WebSocket-Protocol, or None.
        """
        # Check Authorization header (for native clients)
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer ") and auth_header[7:] == self.api_key:
            return None

        # Check Sec-WebSocket-Protocol (for browser clients)
        # Format: "Authorization.<base64_encoded_api_key>"
        for protocol in websocket.headers.get("sec-websocket-protocol", "").split(","):
            protocol = protocol.strip()
            if protocol.startswith("Authorization."):
                try:
                    b64_key = protocol[14:]
                    b64_key += "=" * (-len(b64_key) % 4)  # Restore padding stripped by browser
                    decoded_key = base64.b64decode(b64_key).decode("utf-8")
                    if decoded_key == self.api_key:
                        return protocol
                except Exception:
                    pass

        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid or missing API Key",
        )

    def get_websocket_router(self, path: str = "/ws/stt"):
        """Create FastAPI router for WebSocket endpoint."""
        router = APIRouter()

        @router.websocket(path)
        async def websocket_endpoint(websocket: WebSocket):
            subprotocol = None
            if self.api_key:
                subprotocol = self._authenticate_websocket(websocket)
            await websocket.accept(subprotocol=subprotocol)
            session_data = SpeechRecognitionSessionData()

            try:
                while True:
                    await self.process_websocket(websocket, session_data)

            except Exception as ex:
                error_message = str(ex)

                if "WebSocket is not connected" in error_message:
                    logger.info(f"STT WebSocket disconnected (1): session_id={session_data.id}")
                elif "<CloseCode.NO_STATUS_RCVD: 1005>" in error_message:
                    logger.info(f"STT WebSocket disconnected (2): session_id={session_data.id}")
                else:
                    logger.error(f"STT WebSocket error: {error_message}", exc_info=True)

            finally:
                if session_data.id:
                    if self._on_disconnect:
                        await self._on_disconnect(session_data)

                    await self.finalize_session(session_data.id)

                    if session_data.id in self.websockets:
                        del self.websockets[session_data.id]
                    if session_data.id in self.sessions:
                        del self.sessions[session_data.id]

        return router
