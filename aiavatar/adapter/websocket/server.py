import asyncio
import base64
import io
import logging
import re
import wave
from typing import List, Dict
from fastapi import APIRouter, WebSocket
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
from ..models import AvatarControlRequest, AIAvatarRequest, AIAvatarResponse
from ..base import Adapter

logger = logging.getLogger(__name__)


class WebSocketSessionData:
    def __init__(self):
        self.id = None
        self.data = {}


class AIAvatarWebSocketServer(Adapter):
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

        # WebSocket processing
        response_audio_chunk_size: int = 0, # 0 = Send whole audio data at once
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
            db_connection_str=db_connection_str,
            session_state_manager=session_state_manager,
            performance_recorder=performance_recorder,
            voice_recorder=voice_recorder,
            voice_recorder_enabled=voice_recorder_enabled,
            voice_recorder_dir=voice_recorder_dir,
            debug=debug
        )

        # Call base after self.sts is set
        super().__init__(self.sts)
        self.websockets: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, WebSocketSessionData] = {}

        # Callbacks
        self._on_connect = None
        self._on_disconnect = None

        # WebSocket processing
        self.response_audio_chunk_size = response_audio_chunk_size

        # Debug
        self.debug = debug
        self.last_response = None

    def on_connect(self, func) -> dict:
        self._on_connect = func
        return func

    def on_disconnect(self, func) -> dict:
        self._on_disconnect = func
        return func

    # Request
    async def process_websocket(self, websocket: WebSocket, session_data: WebSocketSessionData):
        data = await websocket.receive_text()
        request = AIAvatarRequest.model_validate_json(data)

        if not request.session_id:
            await websocket.send_text(AIAvatarResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"error": "WebSocket disconnect: session_id is required."}
            ).model_dump_json())
            logger.info("WebSocket disconnect: session_id is required.")
            await websocket.close()
            return

        if request.type == "start":
            self.websockets[request.session_id] = websocket
            session_data.id = request.session_id

            logger.info(f"WebSocket connected for session: {request.session_id}")

            # Store session data to initialize
            if request.user_id:
                self.sts.vad.set_session_data(request.session_id, "user_id", request.user_id, True)
            if request.context_id:
                self.sts.vad.set_session_data(request.session_id, "context_id", request.context_id, True)
            session_data.data["metadata"] = request.metadata
            self.sessions[session_data.id] = session_data

            await self.send_response(AIAvatarResponse(
                type="connected",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id
            ))

            if self._on_connect:
                asyncio.create_task(self._on_connect(request, session_data))

        elif request.type == "invoke":
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

        elif request.type == "data":
            audio_data = base64.b64decode(request.audio_data)
            await self.sts.vad.process_samples(audio_data, request.session_id)

        elif request.type == "config":
            if hasattr(self.sts.vad, "volume_db_threshold"):
                volume_db_threshold = request.metadata.get("volume_db_threshold")
                if volume_db_threshold:
                    self.sts.vad.set_volume_db_threshold(request.session_id, volume_db_threshold)

        elif request.type == "stop":
            logger.info(f"WebSocket disconnect for session: {request.session_id}")
            await websocket.close()

    # Response
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

    async def send_response(self, aiavatar_response: AIAvatarResponse):
        await self.websockets[aiavatar_response.session_id].send_text(
            aiavatar_response.model_dump_json()
        )

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

            # Language
            aiavatar_response.language = response.language

            # Face and Animation
            aiavatar_response.avatar_control_request = self.parse_avatar_control_request(response.text)

            # Voice
            if response.audio_data:
                if self.response_audio_chunk_size > 0:
                    # Extract WAV audio data and parameters
                    with wave.open(io.BytesIO(response.audio_data), "rb") as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        audio_data = wav_file.readframes(frames)

                    pcm_format = {
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "sample_width": sample_width
                    }

                    # Send response without audio data first
                    aiavatar_response.metadata["pcm_format"] = pcm_format
                    aiavatar_response.audio_data = None
                    await self.send_response(aiavatar_response)

                    # Send audio data in chunks
                    for i in range(0, len(audio_data), self.response_audio_chunk_size):
                        # Encode chunk as base64
                        b64_chunk = base64.b64encode(audio_data[i:i + self.response_audio_chunk_size]).decode("utf-8")
                        chunk_response = AIAvatarResponse(
                            type=response.type,
                            session_id=response.session_id,
                            user_id=response.user_id,
                            context_id=response.context_id,
                            audio_data=b64_chunk,
                            metadata={"pcm_format": pcm_format}
                        )
                        await self.send_response(chunk_response)
                    # Return here not to send response again at the end of this method
                    return

                else:
                    b64_chunk = base64.b64encode(response.audio_data).decode("utf-8")
                    aiavatar_response.audio_data = b64_chunk

        elif response.type == "tool_call":
            aiavatar_response.metadata["tool_call"] = response.tool_call.to_dict()

        elif response.type == "final":
            if image_source_match := re.search(r"\[vision:(\w+)\]", response.text):
                aiavatar_response.type = "vision"
                aiavatar_response.metadata={"source": image_source_match.group(1)}

        elif response.type == "stop":
            await self.stop_response(response)

        await self.send_response(aiavatar_response)

    async def stop_response(self, session_id: str, context_id: str):
        # Send stop message to client
        await self.send_response(AIAvatarResponse(
            type="stop",
            session_id=session_id,
            context_id=context_id
        ))

    def get_websocket_router(self, path: str = "/ws"):
        router = APIRouter()

        @router.websocket(path)
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            session_data = WebSocketSessionData()

            try:
                while True:
                    await self.process_websocket(websocket, session_data)

            except Exception as ex:
                error_message = str(ex)

                if "WebSocket is not connected" in error_message:
                    logger.info(f"WebSocket disconnected (1): session_id={session_data.id}")
                elif "<CloseCode.NO_STATUS_RCVD: 1005>" in error_message:
                    logger.info(f"WebSocket disconnected (2): session_id={session_data.id}")
                else:
                    raise

            finally:
                if session_data.id:
                    if self._on_disconnect:
                        await self._on_disconnect(session_data)

                    await self.sts.finalize(session_data.id)
                    if session_data.id in self.websockets:
                        del self.websockets[session_data.id]
                    if session_data.id in self.sessions:
                        del self.sessions[session_data.id]

        return router
