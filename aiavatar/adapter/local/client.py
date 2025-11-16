import logging
import re
from typing import List
from ...sts.models import STSRequest, STSResponse
from ...sts.pipeline import STSPipeline
from ...sts.vad import SpeechDetector
from ...sts.stt import SpeechRecognizer
from ...sts.stt.openai import OpenAISpeechRecognizer
from ...sts.llm import LLMService
from ...sts.llm.context_manager import ContextManager
from ...sts.tts import SpeechSynthesizer
from ...sts.session_state_manager import SessionStateManager
from ...sts.performance_recorder import PerformanceRecorder
from ...sts.voice_recorder import VoiceRecorder
from ...device import NoiseLevelDetector
from ..models import AvatarControlRequest, AIAvatarResponse, AIAvatarException
from ..client import AIAvatarClientBase

logger = logging.getLogger(__name__)


class AIAvatar(AIAvatarClientBase):
    def __init__(
        self,
        *,
        # Quick start
        volume_db_threshold: float = -50.0,
        silence_duration_threshold: float = 0.5,
        input_sample_rate: int = 16000,
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-4.1",
        system_prompt: str = None,
        voicevox_speaker: int = 46,
        voicevox_url: str = "http://127.0.0.1:50021",

        # STS Pipeline and its components
        sts: STSPipeline = None,
        vad: SpeechDetector = None,
        stt: SpeechRecognizer = None,
        llm: LLMService = None,
        tts: SpeechSynthesizer = None,

        # STS Pipeline params for default components
        vad_volume_db_threshold: float = -50.0,
        vad_silence_duration_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        stt_sample_rate: int = 16000,
        llm_openai_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "gpt-4.1",
        llm_system_prompt: str = None,
        llm_context_manager: ContextManager = None,
        tts_voicevox_url: str = "http://127.0.0.1:50021",
        tts_voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        wakeword_timeout: float = 60.0,
        merge_request_threshold: float = 0.0,
        merge_request_prefix: str = "$Previous user's request and your response have been canceled. Please respond again to the following request:\n\n",
        db_connection_str: str = "aiavatar.db",
        session_state_manager: SessionStateManager = None,
        performance_recorder: PerformanceRecorder = None,
        voice_recorder: VoiceRecorder = None,
        voice_recorder_enabled: bool = True,
        voice_recorder_dir: str = "recorded_voices",

        # Noise filter
        auto_noise_filter_threshold: bool = True,
        noise_margin: float = 20.0,

        # Client configurations
        face_controller = None,
        animation_controller = None,
        input_device_index = -1,
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

        # Speech-to-Speech pipeline
        self.sts = sts or STSPipeline(
            # VAD
            vad=vad,
            vad_volume_db_threshold=vad_volume_db_threshold or volume_db_threshold,
            vad_silence_duration_threshold=vad_silence_duration_threshold or silence_duration_threshold,
            vad_sample_rate=vad_sample_rate or input_sample_rate,
            # STT (Overwrite default)
            stt=stt or OpenAISpeechRecognizer(
                openai_api_key=openai_api_key,
                sample_rate=stt_sample_rate or input_sample_rate
            ),
            # LLM
            llm=llm,
            llm_openai_api_key=llm_openai_api_key or openai_api_key,
            llm_base_url=llm_base_url or openai_base_url,
            llm_model=llm_model or openai_model,
            llm_system_prompt=llm_system_prompt or system_prompt,
            llm_context_manager=llm_context_manager,
            # TTS
            tts=tts,
            tts_voicevox_url=tts_voicevox_url or voicevox_url,
            tts_voicevox_speaker=tts_voicevox_speaker or voicevox_speaker,
            # Pipeline
            wakewords=wakewords,
            wakeword_timeout=wakeword_timeout,
            merge_request_threshold=merge_request_threshold,
            merge_request_prefix=merge_request_prefix,
            db_connection_str=db_connection_str,
            session_state_manager=session_state_manager,
            performance_recorder=performance_recorder,
            voice_recorder=voice_recorder,
            voice_recorder_enabled=voice_recorder_enabled,
            voice_recorder_dir=voice_recorder_dir,
            debug=debug
        )
        self.sts.handle_response = self.handle_response
        self.sts.stop_response = self.stop_response

        # Noise filter
        self.auto_noise_filter_threshold = auto_noise_filter_threshold
        self.noise_margin = noise_margin

    async def send_request(self, request):
        async for r in self.sts.invoke(STSRequest(
            type=request.type,
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            text=request.text,
            audio_data=request.audio_data,
            files=request.files,
            system_prompt_params=request.system_prompt_params
        )):
            await self.sts.handle_response(r)

    async def send_microphone_data(self, audio_bytes, session_id):
        await self.sts.vad.process_samples(samples=audio_bytes, session_id=session_id)

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

    async def handle_response(self, response: STSResponse):
        aiavatar_response = AIAvatarResponse(
            type=response.type,
            session_id=response.session_id,
            user_id=response.user_id,
            context_id=response.context_id,
            text=response.text,
            voice_text=response.voice_text,
            audio_data=response.audio_data,
            metadata=response.metadata or {}
        )

        if response.type == "chunk":
            # Stop response if guardrail triggered
            if response.metadata.get("is_guardrail_triggered"):
                await self.stop_response(response.session_id, response.context_id)
            # Parse avatar control
            aiavatar_response.avatar_control_request = self.parse_avatar_control_request(response.text)

        elif response.type == "final":
            if response.text:
                if image_source_match := re.search(r"\[vision:(\w+)\]", response.text):
                    aiavatar_response.type = "vision"
                    aiavatar_response.metadata={"source": image_source_match.group(1)}

        elif response.type == "error":
            raise AIAvatarException(
                message=response.metadata.get("error", "Error in processing pipeline"),
                response=response
            )

        await self.response_queue.put(aiavatar_response)

    async def initialize_session(self, session_id: str, user_id: str, context_id: str):
        if user_id:
            self.sts.vad.set_session_data(session_id, "user_id", user_id, True)
        if context_id:
            self.sts.vad.set_session_data(session_id, "context_id", context_id, True)

    async def start_listening(self, session_id: str = "local_session", user_id: str = "local_user", context_id: str = None):
        # Set noise filter
        if hasattr(self.sts.vad, "set_volume_db_threshold"):
            if self.auto_noise_filter_threshold:
                noise_level_detector = NoiseLevelDetector(
                    rate=self.audio_recorder.sample_rate,
                    channels=self.audio_recorder.channels,
                    device_index=self.audio_devices.input_device
                )
                noise_level = noise_level_detector.get_noise_level()
                volume_threshold_db = int(noise_level) + self.noise_margin

                logger.info(f"Set volume threshold: {volume_threshold_db}dB")
                self.sts.vad.set_volume_db_threshold(session_id, volume_threshold_db)
            else:
                logger.info(f"Set volume threshold: {self.sts.vad.volume_db_threshold}dB")

        await super().start_listening(session_id, user_id, context_id)
