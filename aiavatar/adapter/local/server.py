import asyncio
import logging
import re
from typing import List
from ...database import PoolProvider
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
from ..models import AvatarControlRequest, AIAvatarRequest, AIAvatarResponse, AIAvatarException
from ..base import Adapter

logger = logging.getLogger(__name__)


class AIAvatarLocalServer(Adapter):
    def __init__(
        self,
        response_queue: asyncio.Queue[AIAvatarResponse],
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
        timestamp_interval_seconds: float = 0.0,
        timestamp_prefix: str = "$Current date and time: ",
        timestamp_timezone: str = "UTC",
        mute_on_barge_in: bool = False,
        db_pool_provider: PoolProvider = None,
        db_connection_str: str = "aiavatar.db",
        session_state_manager: SessionStateManager = None,
        performance_recorder: PerformanceRecorder = None,
        voice_recorder: VoiceRecorder = None,
        voice_recorder_enabled: bool = True,
        voice_recorder_dir: str = "recorded_voices",
        invoke_queue_idle_timeout: float = 10.0,
        invoke_timeout: float = 60.0,
        use_invoke_queue: bool = False,

        # Debug
        debug: bool = False,
    ):
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
            timestamp_interval_seconds=timestamp_interval_seconds,
            timestamp_prefix=timestamp_prefix,
            timestamp_timezone=timestamp_timezone,
            db_pool_provider=db_pool_provider,
            db_connection_str=db_connection_str,
            session_state_manager=session_state_manager,
            performance_recorder=performance_recorder,
            voice_recorder=voice_recorder,
            voice_recorder_enabled=voice_recorder_enabled,
            voice_recorder_dir=voice_recorder_dir,
            invoke_queue_idle_timeout=invoke_queue_idle_timeout,
            invoke_timeout=invoke_timeout,
            use_invoke_queue=use_invoke_queue,
            debug=debug
        )

        # Call base after self.sts is set
        super().__init__(self.sts)

        # Client communication queue
        self.response_queue = response_queue

        # Mute immediately on barge-in
        if mute_on_barge_in:
            @self.sts.vad.on_recording_started
            async def mute_on_barge_in(session_id: str):
                await self.stop_response(session_id, "")

        # Debug
        self.debug = debug
        self.last_response = None

    async def start_session(self, session_id: str, user_id: str, context_id: str) -> AIAvatarResponse:
        request = AIAvatarRequest(
            type="start",
            session_id=session_id,
            user_id=user_id,
            context_id=context_id
        )

        for on_session_start in self._on_session_start_handlers:
            await on_session_start(request, None)

        await self.handle_response(STSResponse(
            type="connected",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id
        ))

        return AIAvatarResponse(
            type="connected",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id
        )

    def set_session_data(self, session_id: str, key: str, value: any, create_session: bool = False):
        self.sts.vad.set_session_data(session_id, key, value, create_session)

    async def send_request(self, request: AIAvatarRequest):
        async for r in self.sts.invoke(STSRequest(
            type=request.type,
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            text=request.text,
            audio_data=request.audio_data,
            files=request.files,
            system_prompt_params=request.system_prompt_params,
            allow_merge=request.allow_merge,
            wait_in_queue=request.wait_in_queue
        )):
            await self.handle_response(r)

    async def handle_microphone_data(self, audio_bytes, session_id):
        await self.sts.vad.process_samples(samples=audio_bytes, session_id=session_id)

    def parse_avatar_control_request(self, text: str) -> AvatarControlRequest:
        avreq = AvatarControlRequest()

        if not text:
            return avreq

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

        # Callback for each response chunk
        for on_resp in self._on_response_handlers:
            await on_resp(aiavatar_response, response)

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

    async def stop_response(self, session_id: str, context_id: str):
        # Send stop message to client
        await self.response_queue.put(AIAvatarResponse(
            type="stop",
            session_id=session_id,
            context_id=context_id
        ))
