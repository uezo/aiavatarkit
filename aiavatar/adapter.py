import asyncio
import io
import logging
import queue
import re
import threading
from typing import AsyncGenerator, Dict
import wave
import pyaudio
from litests.models import STSRequest, STSResponse
from litests.pipeline import LiteSTS
from litests.adapter import Adapter
from .face import FaceController
from .animation import AnimationController

logger = logging.getLogger(__name__)


class AvatarRequest:
    def __init__(self, text_to_speech: str=None, animation_name: str=None, animation_duration: float = 3.0, face_name: str=None, face_duration: float=3.0):
        self.text_to_speech = text_to_speech
        self.animation_name = animation_name
        self.animation_duration = animation_duration
        self.face_name = face_name
        self.face_duration = face_duration


class AIAvatarAdapter(Adapter):
    def __init__(
        self,
        *,
        sts: LiteSTS,
        face_controller: FaceController,
        animation_controller: AnimationController,
        input_device_index: int = None,
        input_channels: int = 1,
        input_chunk_size: int = 512,
        output_device_index: int = None,
        output_chunk_size: int = 1024,
        cancel_echo: bool = True
    ):
        super().__init__(sts)

        # Avatar controllers
        self.face_controller = face_controller
        self.animation_controller = animation_controller

        # Microphpne
        self.input_sample_rate = sts.vad.sample_rate
        self.input_channels = input_channels
        self.input_chunk_size = input_chunk_size
        self.input_device_index = input_device_index
        self.is_listening = False

        # Audio player
        self.to_wave = None
        self.p = pyaudio.PyAudio()
        self.play_stream = None
        self.wave_params = None
        self.output_chunk_size = output_chunk_size
        self.output_device_index = output_device_index

        # Echo cancellation
        self.cancel_echo = cancel_echo
        self.is_playing_locally = False
        self.sts.vad.should_mute = lambda: self.cancel_echo and self.is_playing_locally

        # Response handler
        self.stop_event = threading.Event()
        self.response_queue: queue.Queue[STSResponse] = queue.Queue()
        self.response_handler_thread = threading.Thread(target=self.avatar_control_worker, daemon=True)
        self.response_handler_thread.start()

        # Image processing
        self._get_image_url = self.get_image_url_default

    def get_image_url(self, func):
        self._get_image_url = func
        return func

    # Request
    async def start_listening(self, session_id: str):
        async def start_microphone_stream() -> AsyncGenerator[bytes, None]:
            p = pyaudio.PyAudio()
            pyaudio_stream = p.open(
                rate=self.input_sample_rate,
                channels=self.input_channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.input_chunk_size,
                input_device_index=self.input_device_index
            )
            self.is_listening = True
            while self.is_listening:
                yield pyaudio_stream.read(self.input_chunk_size, exception_on_overflow=False)
                await asyncio.sleep(0.0001)

        try:
            await self.sts.vad.process_stream(start_microphone_stream(), session_id)
        finally:
            self.is_listening = False

    def stop_listening(self):
        self.is_listening = False

    # Response
    def parse_avatar_request(self, text: str) -> AvatarRequest:
        if not text:
            return None

        avreq = AvatarRequest()

        # Face
        face_pattarn = r"\[face:(\w+)\]"
        faces = re.findall(face_pattarn, text)
        if faces:
            avreq.face_name = faces[0]
            avreq.face_duration = 4.0

        # Animation
        animation_pattarn = r"\[animation:(\w+)\]"
        animations = re.findall(animation_pattarn, text)
        if animations:
            avreq.animation_name = animations[0]
            avreq.animation_duration = 4.0

        return avreq

    def avatar_control_worker(self):
        while True:
            try:
                response = self.response_queue.get()
                self.is_playing_locally = True

                # Face and Animation
                avatar_request = self.parse_avatar_request(response.text)
                if avatar_request:
                    if avatar_request.face_name:
                        try:
                            asyncio.run(self.face_controller.set_face(avatar_request.face_name, avatar_request.face_duration))
                        except Exception as fex:
                            logger.error(f"Error in setting face: {fex}")
                    if avatar_request.animation_name:
                        try:
                            asyncio.run(self.animation_controller.animate(avatar_request.animation_name, avatar_request.animation_duration))
                        except Exception as aex:
                            logger.error(f"Error in setting animation: {aex}")

                # Voice
                wave_content = self.to_wave(response.audio_data) \
                    if self.to_wave else response.audio_data
                if wave_content:
                    with wave.open(io.BytesIO(wave_content), "rb") as wf:
                        current_params = wf.getparams()
                        if not self.play_stream or self.wave_params != current_params:
                            self.wave_params = current_params
                            self.play_stream = self.p.open(
                                format=self.p.get_format_from_width(self.wave_params.sampwidth),
                                channels=self.wave_params.nchannels,
                                rate=self.wave_params.framerate,
                                output=True,
                                output_device_index=self.output_device_index
                            )

                        data = wf.readframes(self.output_chunk_size)
                        while True:
                            data = wf.readframes(self.output_chunk_size)
                            if not data:
                                break
                            self.play_stream.write(data)

            except Exception as ex:
                logger.error(f"Error processing response: {ex}", exc_info=True)

            finally:
                self.is_playing_locally = False
                self.response_queue.task_done()

    async def get_image_url_default(self, image_source: str) -> str:
        return None

    async def handle_response(self, response: STSResponse):
        if response.type == "chunk":
            self.response_queue.put(response)
        elif response.type == "final":
            if image_source_match := re.search(r"\[vision:(\w+)\]", response.text):
                image_url = await self._get_image_url(image_source_match.group(1))
                if image_url:
                    async for image_response in self.sts.invoke(STSRequest(
                        context_id=response.context_id, files=[{"type": "image", "url": image_url}]
                    )):
                        await self.sts.handle_response(image_response)

    async def stop_response(self, context_id: str):
        while not self.response_queue.empty():
            try:
                _ = self.response_queue.get_nowait()
                self.response_queue.task_done()
            except:
                break

    def close(self):
        self.stop_event.set()
        self.stop_response()
        self.response_handler_thread.join()
