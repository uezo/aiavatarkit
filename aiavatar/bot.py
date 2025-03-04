import asyncio
from logging import getLogger, NullHandler
from time import time
import traceback
from uuid import uuid4
from litests import LiteSTS
from litests.models import STSRequest, STSResponse
from litests.vad import SpeechDetector
from litests.stt import SpeechRecognizer
from litests.stt.openai import OpenAISpeechRecognizer
from litests.llm import LLMService
from litests.tts import SpeechSynthesizer
from litests.performance_recorder import PerformanceRecorder
from .adapter import AIAvatarAdapter
from .device import AudioDevice, NoiseLevelDetector
from .face import FaceController, FaceControllerDummy
from .animation import AnimationController, AnimationControllerDummy

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class AIAvatar:
    def __init__(
        self,
        *,
        sts_pipeline: LiteSTS = None,
        input_device: int = -1,
        output_device: int = -1,
        audio_devices: AudioDevice = None,
        animation_controller: AnimationController = None,
        face_controller: FaceController = None,
        context_timeout: int = 3600,
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
        performance_recorder: PerformanceRecorder = None,
        # Noise filter
        auto_noise_filter_threshold: bool = True,
        noise_margin: float = 20.0,
        # Adapter
        adapter: AIAvatarAdapter = None,
        cancel_echo: bool = True,
        # Wakeword listener
        wakewords: list = None,
        wakeword_timeout: float = 60.0,
        # Debug
        debug: bool = False
    ):
        # Speech-to-Speech pipeline
        self.sts = sts_pipeline or LiteSTS(
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
            performance_recorder=performance_recorder,
            debug=debug
        )

        # Chat
        self.on_turn_end = self.on_turn_end_default
        self.context_timeout = context_timeout
        self.last_request_at = 0
        self.current_context_id = None

        # Wakewords
        self.wakewords = wakewords
        self.wakeword_timeout = wakeword_timeout
        @self.sts.llm.request_filter
        def filter_request(text: str):
            return self.wakeword_request_filter(text)

        # Audio Devices
        if audio_devices:
            self.audio_devices = audio_devices
        else:
            self.audio_devices = AudioDevice(input_device, output_device)
        logger.info(f"Input device: [{self.audio_devices.input_device}] {self.audio_devices.input_device_info['name']}")
        logger.info(f"Output device: [{self.audio_devices.output_device}] {self.audio_devices.output_device_info['name']}")

        self.auto_noise_filter_threshold = auto_noise_filter_threshold
        self.noise_margin = noise_margin

        # Adapter
        self.adapter = adapter or AIAvatarAdapter(
            sts=self.sts,
            face_controller=face_controller or FaceControllerDummy(),
            animation_controller=animation_controller or AnimationControllerDummy(),
            input_device_index=self.audio_devices.input_device,
            output_device_index=self.audio_devices.output_device,
            cancel_echo=cancel_echo
        )

    def wakeword_request_filter(self, text: str):
        now = time()

        if not self.wakewords:
            self.last_request_at = now
            return text

        if self.wakeword_timeout > now - self.last_request_at:
            self.last_request_at = now
            return text

        for ww in self.wakewords:
            if ww in text:
                logger.info(f"Wake by '{ww}': {text}")
                self.last_request_at = now
                return text

        return None

    async def on_turn_end_default(self, request: STSRequest, response: STSResponse) -> bool:
        return False

    def get_context_id(self):
        if not self.current_context_id or self.context_timeout > time() - self.last_request_at:
            self.current_context_id = str(uuid4())
            logger.info(f"New context: {self.current_context_id}")
        return self.current_context_id

    async def chat(self, text: str):
        try:
            request = STSRequest(context_id=self.get_context_id(), text=text)
            async for response in self.sts.invoke(request):
                await self.sts.handle_response(response)
                if response.type == "final":
                    await self.on_turn_end(request, response)

        except Exception as ex:
            logger.error(f"Error at chatting loop: {str(ex)}\n{traceback.format_exc()}")

    def stop_chat(self):
        asyncio.run(self.sts.stop_response())

    async def start_listening(self):
        if self.auto_noise_filter_threshold:
            noise_level_detector = NoiseLevelDetector(
                rate=self.adapter.input_sample_rate,
                channels=self.adapter.input_channels,
                device_index=self.audio_devices.input_device
            )
            noise_level = noise_level_detector.get_noise_level()
            volume_threshold_db = int(noise_level) + self.noise_margin

            logger.info(f"Set volume threshold: {volume_threshold_db}dB")
            self.sts.volume_db_threshold = volume_threshold_db
        else:
            logger.info(f"Set volume threshold: {self.sts.vad.volume_db_threshold}dB")

        await self.adapter.start_listening(self.get_context_id())

    def stop_listening(self):
        self.adapter.stop_listening()
