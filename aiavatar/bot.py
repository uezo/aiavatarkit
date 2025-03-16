import asyncio
import logging
import re
from typing import List, Dict, Optional
from pydantic import BaseModel
from litests.models import STSRequest, STSResponse
from litests.pipeline import LiteSTS
from litests.adapter import Adapter
from litests.vad import SpeechDetector
from litests.stt import SpeechRecognizer
from litests.stt.openai import OpenAISpeechRecognizer
from litests.llm import LLMService
from litests.tts import SpeechSynthesizer
from litests.performance_recorder import PerformanceRecorder
from .device import AudioDevice, AudioPlayer, AudioPlayer, AudioRecorder, NoiseLevelDetector
from .animation import AnimationController, AnimationControllerDummy
from .face import FaceController, FaceControllerDummy

logger = logging.getLogger(__name__)


class AIAvatarRequest(BaseModel):
    type: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    text: Optional[str] = None
    audio_data: Optional[str] = None
    metadata: Optional[Dict] = None


class AvatarControlRequest(BaseModel):
    animation_name: Optional[str] = None
    animation_duration: Optional[float] = None
    face_name: Optional[str] = None
    face_duration: Optional[float] = None


class AIAvatarResponse(BaseModel):
    type: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    text: Optional[str] = None
    voice_text: Optional[str] = None
    avatar_control_request: Optional[AvatarControlRequest] = None
    audio_data: Optional[bytes] = None
    metadata: Optional[Dict] = None


class AIAvatar(Adapter):
    def __init__(
        self,
        *,
        sts: LiteSTS = None,
        face_controller: FaceController = None,
        animation_controller: AnimationController = None,
        # Audio device
        input_device_index: int = -1,
        input_channels: int = 1,
        input_chunk_size: int = 512,
        output_device_index: int = -1,
        output_chunk_size: int = 1024,
        audio_devices: AudioDevice = None,
        cancel_echo: bool = True,
        # STS Pipeline components
        vad: SpeechDetector = None,
        stt: SpeechRecognizer = None,
        llm: LLMService = None,
        tts: SpeechSynthesizer = None,
        # STS Pipeline params for default components
        volume_db_threshold: float = -50.0,
        silence_duration_threshold: float = 0.5,
        input_sample_rate: int = 16000,
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-4o-mini",
        system_prompt: str = None,
        voicevox_url: str = "http://127.0.0.1:50021",
        voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        wakeword_timeout: float = 60.0,
        performance_recorder: PerformanceRecorder = None,
        # Noise filter
        auto_noise_filter_threshold: bool = True,
        noise_margin: float = 20.0,
        # Debug
        debug: bool = False
    ):
        # Speech-to-Speech pipeline
        self.sts = sts or LiteSTS(
            vad=vad,
            vad_volume_db_threshold=volume_db_threshold,
            vad_silence_duration_threshold=silence_duration_threshold,
            vad_sample_rate=input_sample_rate,
            stt=stt or OpenAISpeechRecognizer(
                openai_api_key=openai_api_key,
                sample_rate=input_sample_rate
            ),
            llm=llm,
            llm_openai_api_key=openai_api_key,
            llm_base_url=openai_base_url,
            llm_model=openai_model,
            llm_system_prompt=system_prompt,
            tts=tts,
            tts_voicevox_url=voicevox_url,
            tts_voicevox_speaker=voicevox_speaker,
            wakewords=wakewords,
            wakeword_timeout=wakeword_timeout,
            performance_recorder=performance_recorder,
            debug=debug
        )

        # Call base after self.sts is set
        super().__init__(self.sts)

        # Audio Devices
        if audio_devices:
            self.audio_devices = audio_devices
        else:
            self.audio_devices = AudioDevice(input_device_index, output_device_index)
        logger.info(f"Input device: [{self.audio_devices.input_device}] {self.audio_devices.input_device_info['name']}")
        logger.info(f"Output device: [{self.audio_devices.output_device}] {self.audio_devices.output_device_info['name']}")

        # Microphpne
        self.audio_recorder = AudioRecorder(
            sample_rate=self.sts.vad.sample_rate,
            device_index=self.audio_devices.input_device,
            channels=input_channels,
            chunk_size=input_chunk_size
        )

        # Audio player
        self.audio_player = AudioPlayer(
            device_index=self.audio_devices.output_device,
            chunk_size=output_chunk_size
        )

        # Noise filter
        self.auto_noise_filter_threshold = auto_noise_filter_threshold
        self.noise_margin = noise_margin

        # Avatar controllers
        self.face_controller = face_controller or FaceControllerDummy()
        self.animation_controller = animation_controller or AnimationControllerDummy()

        # Echo cancellation
        self.cancel_echo = cancel_echo
        self.is_playing_locally = False

        # Image processing
        self._get_image_url = self.get_image_url_default

        # Request and response processing
        self.request_task = None
        self.response_task = None
        self.response_queue: asyncio.Queue[AIAvatarResponse] = asyncio.Queue()

        # Debug
        self.debug = debug
        self.last_response = None

    def get_image_url(self, func):
        self._get_image_url = func
        return func

    async def get_image_url_default(self, image_source: str) -> str:
        return None

    # Send microphone data to VAD
    async def send_microphone_data(self, session_id: str):
        async for data in self.audio_recorder.start_stream():
            if not self.cancel_echo or not self.audio_player.is_playing:
                await self.sts.vad.process_samples(samples=data, session_id=session_id)

    def parse_avatar_control_request(self, text: str) -> AvatarControlRequest:
        avreq = AvatarControlRequest()

        if text:
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

    # Perform Face, Animation and Speech in response from server
    async def response_worker(self):
        loop = asyncio.get_running_loop()

        while True:
            try:
                response = await self.response_queue.get()
                avreq = response.avatar_control_request

                if avreq:
                    if avreq.face_name:
                        asyncio.create_task(self.face_controller.set_face(avreq.face_name, avreq.face_duration))
                    if avreq.animation_name:
                        asyncio.create_task(self.animation_controller.animate(avreq.animation_name, avreq.animation_duration))

                if response.audio_data:
                    await loop.run_in_executor(None, self.audio_player.play, response.audio_data)

            except Exception as ex:
                logger.error(f"Error processing response: {ex}", exc_info=True)

            finally:
                self.is_playing_locally = False
                if not self.response_queue.empty():
                    self.response_queue.task_done()

    async def handle_response(self, response: STSResponse):
        if response.type == "chunk":
            await self.response_queue.put(AIAvatarResponse(
                type="chunk",
                session_id=response.session_id,
                user_id=response.user_id,
                context_id=response.context_id,
                text=response.text,
                voice_text=response.voice_text,
                avatar_control_request=self.parse_avatar_control_request(response.text),
                audio_data=response.audio_data,
                metadata={}
            ))

        elif response.type == "final":
            if image_source_match := re.search(r"\[vision:(\w+)\]", response.text):
                image_url = await self._get_image_url(image_source_match.group(1))
                if image_url:
                    response.type = "vision"    # Overwrite response type
                    async for image_response in self.sts.invoke(STSRequest(
                        context_id=response.context_id, files=[{"type": "image", "url": image_url}]
                    )):
                        await self.sts.handle_response(image_response)

        elif response.type == "stop":
            await self.stop_response()

        if self.debug and response.type == "final":
            self.last_response = response

    async def stop_response(self, session_id: str, context_id: str):
        # Clear queue
        while not self.response_queue.empty():
            try:
                _ = self.response_queue.get_nowait()
                self.response_queue.task_done()
            except:
                break
        # Stop voice
        self.audio_player.stop()

    async def close(self):
        await self.stop_response()

    # Entrypoint
    async def start_listening(self, session_id: str = "local_session", user_id: str = "local_user", context_id: str = None):
        # Set noise filter
        if self.auto_noise_filter_threshold:
            noise_level_detector = NoiseLevelDetector(
                rate=self.audio_recorder.sample_rate,
                channels=self.audio_recorder.channels,
                device_index=self.audio_devices.input_device
            )
            noise_level = noise_level_detector.get_noise_level()
            volume_threshold_db = int(noise_level) + self.noise_margin

            logger.info(f"Set volume threshold: {volume_threshold_db}dB")
            self.sts.volume_db_threshold = volume_threshold_db
        else:
            logger.info(f"Set volume threshold: {self.sts.vad.volume_db_threshold}dB")

        # Start tasks
        self.response_task = asyncio.create_task(self.response_worker())
        if user_id:
            self.sts.vad.set_session_data(session_id, "user_id", user_id, True)
        if context_id:
            self.sts.vad.set_session_data(session_id, "context_id", context_id, True)
        self.request_task = asyncio.create_task(self.send_microphone_data(session_id))

        try:
            await asyncio.gather(self.response_task, self.request_task)
        except:
            await self.stop_listening()

    async def stop_listening(self):
        try:
            if self.request_task:
                self.request_task.cancel()
        except Exception as ex:
            logger.warning(f"Error at canceling request_task: {ex}")

        try:
            if self.response_task:
                self.response_task.cancel()
        except Exception as ex:
            logger.warning(f"Error at canceling response_task: {ex}")

        await self.sts.shutdown()
