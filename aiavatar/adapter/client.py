from abc import abstractmethod
import asyncio
import logging
from typing import List
from ..device import AudioDevice, AudioPlayer, AudioPlayer, AudioRecorder
from ..animation import AnimationController, AnimationControllerDummy
from ..face import FaceController, FaceControllerDummy
from . import AIAvatarRequest, AIAvatarResponse

logger = logging.getLogger(__name__)


class AIAvatarClientBase:
    def __init__(
        self,
        *,
        # Face and animation
        face_controller: FaceController = None,
        animation_controller: AnimationController = None,
        # Audio device
        input_device_index: int = -1,
        input_sample_rate: int = 16000,
        input_channels: int = 1,
        input_chunk_size: int = 512,
        output_device_index: int = -1,
        output_chunk_size: int = 1024,
        audio_devices: AudioDevice = None,
        cancel_echo: bool = True,
        # Debug
        debug: bool = False
    ):
        # Audio Devices
        if audio_devices:
            self.audio_devices = audio_devices
        else:
            self.audio_devices = AudioDevice(input_device_index, output_device_index)
        logger.info(f"Input device: [{self.audio_devices.input_device}] {self.audio_devices.input_device_info['name']}")
        logger.info(f"Output device: [{self.audio_devices.output_device}] {self.audio_devices.output_device_info['name']}")

        # Microphpne
        self.audio_recorder = AudioRecorder(
            sample_rate=input_sample_rate,
            device_index=self.audio_devices.input_device,
            channels=input_channels,
            chunk_size=input_chunk_size
        )

        # Audio player
        self.audio_player = AudioPlayer(
            device_index=self.audio_devices.output_device,
            chunk_size=output_chunk_size
        )

        # Avatar controllers
        self.face_controller = face_controller or FaceControllerDummy()
        self.animation_controller = animation_controller or AnimationControllerDummy()

        # Echo cancellation
        self.cancel_echo = cancel_echo

        # Image processing
        self._get_image_url = self.get_image_url_default

        # Request and response processing
        self.send_microphone_task = None
        self.receive_response_task = None
        self.response_queue: asyncio.Queue[AIAvatarResponse] = asyncio.Queue()
        self._on_responses = {}

        # Debug
        self.debug = debug
        self.last_responses: List[AIAvatarResponse] = []

    def get_image_url(self, func):
        self._get_image_url = func
        return func

    async def get_image_url_default(self, image_source: str) -> str:
        return None

    def on_response(self, response_type: str):
        def decorator(func):
            self._on_responses[response_type] = func
            return func
        return decorator

    @abstractmethod
    async def send_request(self, request: AIAvatarRequest):
        pass

    @abstractmethod
    async def send_microphone_data(self, audio_bytes: bytes, session_id: str):
        pass

    # Send request to Speech-to-Speech pipeline
    async def send_microphone_worker(self, session_id: str):
        async for data in self.audio_recorder.start_stream():
            if not self.cancel_echo or not self.audio_player.is_playing:
                await self.send_microphone_data(data, session_id)

    # Receive response from Speech-to-Speech pipeline
    async def receive_response_worker(self):
        while True:
            try:
                response = await self.response_queue.get()

                if self.debug:
                    if response.type == "start":
                        self.last_responses.clear()
                    self.last_responses.append(response)

                if on_response_func := self._on_responses.get(response.type):
                    await on_response_func(response)

                if response.type == "connected":
                    logger.info(f"Connected: {response.session_id}")

                elif response.type == "chunk":
                    await self.perform_response(response)

                elif response.type == "vision":
                    image_url = await self._get_image_url(response.metadata.get("source"))
                    if image_url:
                        await self.send_request(AIAvatarRequest(
                            type="start",
                            session_id=response.session_id,
                            user_id=response.user_id,
                            context_id=response.context_id,
                            files=[{"type": "image", "url": image_url}]
                        ))

                elif response.type == "stop":
                    await self.stop_response(response.session_id, response.context_id)

            except Exception as ex:
                logger.warning(f"Error at receive_response_worker: {ex}")

    # Perform Face, Animation and Speech
    async def perform_response(self, response: AIAvatarResponse):
        try:
            avreq = response.avatar_control_request

            if avreq:
                if avreq.face_name:
                    asyncio.create_task(self.face_controller.set_face(avreq.face_name, avreq.face_duration))
                if avreq.animation_name:
                    asyncio.create_task(self.animation_controller.animate(avreq.animation_name, avreq.animation_duration))

            if response.audio_data:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.audio_player.play, response.audio_data)

        except Exception as ex:
            logger.error(f"Error processing response: {ex}", exc_info=True)

    async def stop_response(self, session_id: str, context_id: str):
        # Clear queues
        while not self.response_queue.empty():
            try:
                _ = self.response_queue.get_nowait()
                self.response_queue.task_done()
            except:
                break

        self.audio_player.stop()

    @abstractmethod
    async def initialize_session(self, session_id: str, user_id: str, context_id: str):
        pass

    async def start_listening(self, session_id: str, user_id: str, context_id: str):
        self.receive_response_task = asyncio.create_task(self.receive_response_worker())
        self.send_microphone_task = asyncio.create_task(self.send_microphone_worker(session_id))

        await self.initialize_session(session_id, user_id, context_id)

        try:
            await asyncio.gather(self.receive_response_task, self.send_microphone_task)
        except:
            await self.stop_listening(session_id)

    async def stop_listening(self, session_id: str):
        try:
            if self.send_microphone_task:
                self.send_microphone_task.cancel()
        except Exception as ex:
            logger.warning(f"Error at canceling send_microphone_task: {ex}")

        try:
            if self.receive_response_task:
                self.receive_response_task.cancel()
        except Exception as ex:
            logger.warning(f"Error at canceling receive_response_task: {ex}")
