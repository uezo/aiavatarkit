"""Example-local streaming bridge for the ordinary AIAvatar WebSocket protocol."""

import base64
import binascii
from uuid import uuid4

from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from aiavatar.adapter.models import AIAvatarRequest, AIAvatarResponse
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.sts.models import STSRequest


class _BufferedSocket:
    """Replay one validated request through the existing adapter dispatcher."""

    def __init__(self, websocket, request):
        self._websocket = websocket
        self._request = request

    async def receive_text(self):
        return self._request.model_dump_json()

    def __getattr__(self, name):
        return getattr(self._websocket, name)


class RealtimeWebSocketServer(AIAvatarWebSocketServer):
    """Microphone-only comparison server with continuous Live PCM playback.

    Normal Realtime text/TTS responses retain the base adapter's framing.
    Live marks raw PCM with ``pcm_format``; these chunks share a session-local
    ``audio_id`` and ``continuous_audio=True`` without a fabricated frame count.
    """

    def __init__(self, *, sts, **kwargs):
        super().__init__(sts=sts, **kwargs)
        self.on_session_start(self._prepare_stream)

    async def _prepare_stream(self, request, session_data):
        request.metadata = {**(request.metadata or {}), "barge_in_enabled": True}
        stream_request = STSRequest(
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            channel=self.channel,
            metadata=request.metadata,
        )
        await self.sts.prepare_session(stream_request)
        request.context_id = stream_request.context_id

    def _apply_session_config(self, session_id, metadata, create_session=False):
        super()._apply_session_config(
            session_id, {**metadata, "barge_in_enabled": True}, create_session,
        )

    async def _request_error(self, websocket, session_data, message):
        async with session_data.send_lock:
            await websocket.send_text(AIAvatarResponse(
                type="error", session_id=session_data.id,
                voice_text=message, metadata={"error": message},
            ).model_dump_json())

    async def process_websocket(self, websocket, session_data):
        # Validate ownership before the base adapter can install a new mapping.
        # The tiny replay socket lets all accepted requests keep using its
        # existing dispatch, hooks, authentication, and disconnect cleanup.
        data = await websocket.receive_text()
        try:
            request = AIAvatarRequest.model_validate_json(data)
        except ValidationError:
            await self._request_error(websocket, session_data, "Invalid request.")
            return
        if not request.session_id:
            await self._request_error(websocket, session_data, "session_id is required.")
            return
        if request.type == "start":
            if session_data.id or request.session_id in self.websockets:
                await self._request_error(websocket, session_data, "Session is already started.")
                return
        else:
            owner = self.websockets.get(request.session_id)
            if (session_data.id != request.session_id
                    or getattr(owner, "_websocket", owner) is not websocket
                    or self.sessions.get(request.session_id) is not session_data):
                await self._request_error(websocket, session_data, "Start this connection's session first.")
                return
        if (request.type not in {"start", "data", "config", "stop", "playback"}
                or request.text is not None or request.files
                or request.system_prompt_params or request.wait_in_queue):
            await self._request_error(websocket, session_data, "This example accepts microphone audio only.")
            return
        if request.type == "playback":
            # The shared viewer reports playback for turn-taking policies.
            # Continuous listening does not need those local turn notifications.
            return
        if request.type == "data":
            try:
                if not isinstance(request.audio_data, (str, bytes)):
                    raise ValueError
                base64.b64decode(request.audio_data, validate=True)
            except (ValueError, binascii.Error):
                await self._request_error(websocket, session_data, "Invalid microphone audio.")
                return
        request.metadata = {**(request.metadata or {}), "barge_in_enabled": True}
        try:
            await super().process_websocket(_BufferedSocket(websocket, request), session_data)
        except Exception:
            if request.type != "start":
                raise
            try:
                await self._request_error(websocket, session_data, "Could not start the audio session.")
            finally:
                await websocket.close(code=1011)
            # The base route's finally block finalizes even a partially prepared
            # provider and removes the mapping installed before the start hook.
            raise WebSocketDisconnect(code=1011, reason="Audio session startup failed") from None

    async def send_response(self, response):
        session = self.sessions.get(response.session_id)
        if session is None:
            return
        if response.type == "connected":
            response.metadata = {
                **(response.metadata or {}), "realtime": True,
                "input_sample_rate": self.sts.input_sample_rate,
                "barge_in_enabled": True,
            }
        async with session.send_lock:
            socket = self.websockets.get(response.session_id)
            if socket is None or self.sessions.get(response.session_id) is not session:
                return
            if (response.transaction_id
                    and response.transaction_id != session.active_transaction_id):
                return
            if response.type == "stop":
                # Reset the stream in the same critical section as the wire
                # event: the next PCM write must announce a fresh descriptor.
                session.data.pop("continuous_audio", None)
            await socket.send_text(response.model_dump_json())
            if response.type == "connected" and not session.data.get("stream_activated"):
                # Synchronous activation after the successful write avoids an
                # untracked on_connect task racing immediate disconnection.
                session.data["stream_activated"] = True
                self.sts.activate_session(response.session_id)

    async def handle_response(self, response):
        pcm_format = (response.metadata or {}).get("pcm_format")
        if response.type != "chunk" or not pcm_format:
            await super().handle_response(response)
            return
        session = self.sessions.get(response.session_id)
        if session is None:
            return
        wire_response = AIAvatarResponse(
            type=response.type, session_id=response.session_id,
            user_id=response.user_id, context_id=response.context_id,
            transaction_id=response.transaction_id, text=response.text,
            voice_text=response.voice_text, language=response.language,
            audio_data=response.audio_data, metadata=dict(response.metadata),
            structured_content=response.structured_content,
        )
        for callback in self._on_response_handlers:
            await callback(wire_response, response)
        async with session.send_lock:
            socket = self.websockets.get(response.session_id)
            if socket is None or self.sessions.get(response.session_id) is not session:
                return
            if (response.transaction_id
                    and response.transaction_id != session.active_transaction_id):
                return
            stream = session.data.get("continuous_audio")
            if stream is None:
                stream = {"continuous_audio": True, "pcm_format": dict(pcm_format), "audio_id": str(uuid4())}
                session.data["continuous_audio"] = stream
                descriptor = wire_response.model_copy(update={
                    "audio_data": None, "metadata": {**wire_response.metadata, **stream},
                })
                await socket.send_text(descriptor.model_dump_json())
            if response.audio_data:
                if (self.sessions.get(response.session_id) is not session
                        or self.websockets.get(response.session_id) is not socket
                        or (response.transaction_id
                            and response.transaction_id != session.active_transaction_id)):
                    return
                wire_response.audio_data = base64.b64encode(response.audio_data).decode("ascii")
                wire_response.metadata = {**wire_response.metadata, **stream}
                await socket.send_text(wire_response.model_dump_json())
