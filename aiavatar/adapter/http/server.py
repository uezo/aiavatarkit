import base64
import logging
import re
import time
from typing import List, Dict, Optional, Any
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sse_starlette.sse import EventSourceResponse   # pip install sse-starlette
from pydantic import BaseModel, ConfigDict
from ...sts import STSPipeline
from ...sts.models import STSRequest, STSResponse
from ...sts.vad import SpeechDetectorDummy
from ...sts.stt import SpeechRecognizer
from ...sts.stt.openai import OpenAISpeechRecognizer
from ...sts.stt.speaker_registry import SpeakerRegistry, MatchTopKResult
from ...sts.llm import LLMService
from ...sts.llm.context_manager import ContextManager
from ...sts.tts import SpeechSynthesizer
from ...sts.session_state_manager import SessionStateManager
from ...sts.performance_recorder import PerformanceRecorder
from ...sts.voice_recorder import VoiceRecorder
from ..models import AvatarControlRequest, AIAvatarRequest, AIAvatarResponse
from .. import Adapter

logger = logging.getLogger(__name__)


class SynthesizeRequest(BaseModel):
    text: str
    style_info: Optional[Dict] = None
    language: Optional[str] = None


class Candidate(BaseModel):
    speaker_id: str
    similarity: float
    metadata: Dict[str, Any]
    is_new: bool = False


class MatchTopKResultModel(BaseModel):
    chosen: Candidate
    candidates: List[Candidate]

    @classmethod
    def parse(cls, match_result: MatchTopKResult) -> "MatchTopKResultModel":
        """Parse MatchTopKResult from speaker_registry to MatchTopKResultModel"""
        return cls(
            chosen=Candidate(
                speaker_id=match_result.chosen.speaker_id,
                similarity=match_result.chosen.similarity,
                metadata=match_result.chosen.metadata,
                is_new=match_result.chosen.is_new
            ),
            candidates=[
                Candidate(
                    speaker_id=c.speaker_id,
                    similarity=c.similarity,
                    metadata=c.metadata,
                    is_new=c.is_new
                ) for c in match_result.candidates
            ]
        )


class TranscribeResponse(BaseModel):
    text: Optional[str] = None
    preprocess_metadata: Optional[dict] = None
    postprocess_metadata: Optional[dict] = None
    speakers: Optional[MatchTopKResultModel] = None


class PostSpeakerNameRequest(BaseModel):
    speaker_id: str
    name: str


class PostChatMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: Optional[str] = None
    inputs: Optional[Dict[str, str]] = None
    user: str
    conversation_id: Optional[str] = None
    files: Optional[List[Dict[str, str]]] = None


class PostChatMessagesResponse(BaseModel):
    event: str
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None
    answer: Optional[str] = None
    created_at: Optional[int] = None

    class Config:
        exclude_none = True


class AIAvatarHttpServer(Adapter):
    def __init__(
        self,
        *,
        # Quick start
        input_sample_rate: int = 16000,
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-4.1",
        system_prompt: str = None,
        voicevox_speaker: int = 46,
        voicevox_url: str = "http://127.0.0.1:50021",

        # STS Pipeline and its components
        sts: STSPipeline = None,
        stt: SpeechRecognizer = None,
        llm: LLMService = None,
        tts: SpeechSynthesizer = None,

        # STS Pipeline params for default components
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

        # Optional component
        speaker_registry: SpeakerRegistry = None,

        # API server auth
        api_key: str = None,
        # Debug
        debug: bool = False            
    ):
        # Speech-to-Speech pipeline
        self.sts = sts or STSPipeline(
            # VAD
            vad=SpeechDetectorDummy(),
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

        # Optional components
        self.speaker_registry = speaker_registry

        # Custom logic
        self._on_response_chunk = None

        # Debug
        self.debug = debug

        # API Key
        self.api_key = api_key
        self._bearer_scheme = HTTPBearer(auto_error=False)

    def on_response_chunk(self, func):
        self._on_response_chunk = func
        return func

    def api_key_auth(self, credentials: HTTPAuthorizationCredentials):
        if not credentials or credentials.scheme.lower() != "bearer" or credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )
        return credentials.credentials

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

    def get_api_router(self, path: str = "/chat", stt: SpeechRecognizer = None, tts: SpeechSynthesizer = None):
        router = APIRouter()
        bearer_scheme = HTTPBearer(auto_error=False)

        @router.post(path)
        async def post_chat(
            request: AIAvatarRequest,
            credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
        ):
            if not request.session_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "session_id is required."}
                )

            if self.api_key:
                self.api_key_auth(credentials)

            async def stream_response():
                if request.audio_data:
                    request.audio_data = base64.b64decode(request.audio_data)

                async for response in self.sts.invoke(STSRequest(
                    type=request.type,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=request.context_id,
                    text=request.text,
                    audio_data=request.audio_data,
                    files=request.files,
                    system_prompt_params=request.system_prompt_params
                )):
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

                    if self._on_response_chunk:
                        await self._on_response_chunk(aiavatar_response)

                    if response.type == "chunk":
                        # Language
                        aiavatar_response.language = response.language

                        # Face and Animation
                        aiavatar_response.avatar_control_request = self.parse_avatar_control_request(response.text)

                        # Voice
                        if response.audio_data:
                            b64_chunk = base64.b64encode(response.audio_data).decode("utf-8")
                            aiavatar_response.audio_data = b64_chunk

                    if response.type == "tool_call":
                        aiavatar_response.metadata["tool_call"] = response.tool_call.to_dict()

                    elif response.type == "final":
                        if response.text:
                            if image_source_match := re.search(r"\[vision:(\w+)\]", response.text):
                                aiavatar_response.type = "vision"
                                aiavatar_response.metadata={"source": image_source_match.group(1)}

                    elif response.type == "stop":
                        await self.stop_response(response)

                    yield aiavatar_response.model_dump_json()

            return EventSourceResponse(stream_response())

        @router.post("/chat-messages", response_model=PostChatMessagesResponse, summary="Dify-compatible endpoint")
        async def post_chat(
            request: PostChatMessagesRequest,
            credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
        ):
            if self.api_key:
                self.api_key_auth(credentials)

            message_id = f"message_{uuid4()}"
            async def stream_response():
                created_at_int = None
                async for response in self.sts.invoke(STSRequest(
                    type="start",
                    session_id=request.conversation_id or f"sess_temp_{message_id}",
                    user_id=request.user,
                    context_id=request.conversation_id,
                    text=request.query,
                    files=request.files,
                    system_prompt_params=request.inputs
                )):
                    if not created_at_int:
                        created_at_int = int(time.time())

                    if self._on_response_chunk:
                        await self._on_response_chunk(response) # Handle STSResponse instead

                    if response.type == "chunk":
                        chat_messages_response = PostChatMessagesResponse(
                            event="message_replace" if response.metadata.get("is_guardrail_triggered") is True else "message",
                            message_id=message_id,
                            conversation_id=response.context_id,
                            answer=response.text,
                            created_at=created_at_int
                        )
                        yield chat_messages_response.model_dump_json()

                    elif response.type == "final":
                        chat_messages_response = PostChatMessagesResponse(
                            event="message_end",
                            message_id=message_id,
                            conversation_id=response.context_id,
                        )
                        yield chat_messages_response.model_dump_json()

                    elif response.type == "stop":
                        await self.stop_response(response)

            return EventSourceResponse(stream_response())

        @router.post("/transcribe")
        async def post_transcribe(
            audio: UploadFile = File(...),
            session_id: Optional[str] = Form(None),
            credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
        ):
            if self.api_key:
                self.api_key_auth(credentials)

            audio_bytes = await audio.read()

            if not audio_bytes:
                return JSONResponse(
                    status_code=400,
                    content={"error": "audio data is required."}
                )

            result = await (stt or self.sts.stt).recognize(session_id=session_id, data=audio_bytes)

            speakers = None
            if self.speaker_registry:
                try:
                    match_result = self.speaker_registry.match_topk_from_pcm(audio_bytes=audio_bytes, sample_rate=(stt or self.sts.stt).sample_rate)
                    if match_result:
                        speakers = MatchTopKResultModel.parse(match_result)
                except Exception as ex:
                    logger.warning(f"Error at speaker matching: {ex}")

            return TranscribeResponse(
                text=result.text,
                preprocess_metadata=result.preprocess_metadata,
                postprocess_metadata=result.postprocess_metadata,
                speakers=speakers
            )

        @router.post("/transcribe/speaker")
        async def post_transcribe_speaker(
            request: PostSpeakerNameRequest,
            credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
        ):
            if self.api_key:
                self.api_key_auth(credentials)

            if self.speaker_registry:
                self.speaker_registry.set_metadata(request.speaker_id, "name", request.name)
                return JSONResponse(content={"speaker_id": request.speaker_id, "name": request.name})
            else:
                return JSONResponse(content={})

        @router.post("/synthesize")
        async def post_synthesize(
            request: SynthesizeRequest,
            credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
        ):
            if self.api_key:
                self.api_key_auth(credentials)

            if not request.text:
                return JSONResponse(
                    status_code=400,
                    content={"error": "text is required."}
                )

            audio_bytes = await (tts or self.sts.tts).synthesize(
                text=request.text,
                style_info=request.style_info,
                language=request.language
            )

            return Response(
                content=audio_bytes,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=voice.wav"}
            )

        return router

    async def handle_response(self, response: STSResponse):
        # Do nothing here
        pass

    async def stop_response(self, session_id: str, context_id: str):
        # Do nothing here
        pass
