import asyncio
import base64
import json
import logging
import websockets
from .. import AIAvatarRequest, AIAvatarResponse, AIAvatarException
from ..client import AIAvatarClientBase

logger = logging.getLogger(__name__)


class AIAvatarWebSocketClient(AIAvatarClientBase):
    def __init__(
        self,
        *,
        # STS Pipeline server
        url: str = "ws://localhost:8000/ws",
        # Client configurations
        face_controller = None,
        animation_controller = None,
        input_device_index = -1,
        input_sample_rate = 16000,
        input_channels = 1,
        input_chunk_size = 512,
        output_device_index = -1,
        output_chunk_size = 1024,
        audio_devices = None,
        cancel_echo = True,
        debug = False,
    ):
        super().__init__(
            face_controller=face_controller,
            animation_controller=animation_controller,
            input_device_index=input_device_index,
            input_sample_rate=input_sample_rate,
            input_channels=input_channels,
            input_chunk_size=input_chunk_size,
            output_device_index=output_device_index,
            output_chunk_size=output_chunk_size,
            audio_devices=audio_devices,
            cancel_echo=cancel_echo,
            debug=debug
        )

        self.url = url
        self.websocket_connection = None
        self.websocket_task = None

    async def send_request(self, request):
        await self.websocket_connection.send(
            AIAvatarRequest(
                type="invoke",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                text=request.text,
                audio_data=request.audio_data,
                files=request.files,
                system_prompt_params=request.system_prompt_params
            ).model_dump_json()
        )

    async def send_microphone_data(self, audio_bytes, session_id):
        b64_data = base64.b64encode(audio_bytes).decode("utf-8")

        if not self.cancel_echo or not self.audio_player.is_playing:
            await self.websocket_connection.send(json.dumps({
                "type": "data",
                "session_id": session_id,
                "audio_data": b64_data
            }))

    # Receive WebSocket messages
    async def receive_websocket_worker(self):
        while True:
            message_str = await self.websocket_connection.recv()
            response = AIAvatarResponse.model_validate_json(message_str)

            if response.type == "chunk" and response.audio_data:
                response.audio_data = base64.b64decode(response.audio_data)

            if response.type == "error":
                raise AIAvatarException(
                    message=response.metadata.get("error", "Error in processing pipeline"),
                    response=response
                )

            await self.response_queue.put(response)

    async def initialize_session(self, session_id: str, user_id: str, context_id: str):
        start_message = {
            "type": "start",
            "session_id": session_id,
            "user_id": user_id,
            "context_id": context_id
        }
        await self.websocket_connection.send(json.dumps(start_message))

    async def start_listening(self, session_id: str = "ws_session", user_id: str = "ws_user", context_id: str = None):
        async with websockets.connect(self.url) as ws:
            self.websocket_connection = ws

            self.receive_response_task = asyncio.create_task(self.receive_response_worker())
            self.send_microphone_task = asyncio.create_task(self.send_microphone_worker(session_id))
            self.websocket_task = asyncio.create_task(self.receive_websocket_worker())

            await self.initialize_session(session_id, user_id, context_id)

            try:
                await asyncio.gather(self.websocket_task, self.receive_response_task, self.send_microphone_task)
            except:
                await self.stop_listening(session_id)

    async def stop_listening(self, session_id):
        await super().stop_listening(session_id)

        try:
            if self.websocket_task:
                self.websocket_task.cancel()
        except Exception as ex:
            logger.warning(f"Error at canceling websocket_task: {ex}")

        try:
            # Send stop message
            await self.websocket_connection.send(json.dumps({
                "type": "stop",
                "session_id": session_id
            }))
        except Exception as ex:
            logger.warning(f"Error at canceling sending stop message: {ex}")
