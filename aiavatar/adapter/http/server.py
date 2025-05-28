import base64
import logging
import re
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sse_starlette.sse import EventSourceResponse   # pip install sse-starlette
from ...sts import STSPipeline
from ...sts.models import STSRequest, STSResponse
from ...sts.vad import SpeechDetectorDummy
from ...sts.stt import SpeechRecognizer
from ...sts.stt.openai import OpenAISpeechRecognizer
from ...sts.llm import LLMService
from ...sts.tts import SpeechSynthesizer
from ...sts.performance_recorder import PerformanceRecorder
from ..models import AvatarControlRequest, AIAvatarRequest, AIAvatarResponse
from .. import Adapter

logger = logging.getLogger(__name__)


class AIAvatarHttpServer(Adapter):
    def __init__(
        self,
        *,
        # STS Pipeline components
        sts: STSPipeline = None,
        stt: SpeechRecognizer = None,
        llm: LLMService = None,
        tts: SpeechSynthesizer = None,
        # STS Pipeline params for default components
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
        api_key: str = None,
        # Debug
        debug: bool = False            
    ):
        # Speech-to-Speech pipeline
        self.sts = sts or STSPipeline(
            vad=SpeechDetectorDummy(),
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

        # Debug
        self.debug = debug

        # API Key
        self.api_key = api_key
        self._bearer_scheme = HTTPBearer(auto_error=False)

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

    def get_api_router(self, path: str = "/chat"):
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

        return router

    async def handle_response(self, response: STSResponse):
        # Do nothing here
        pass

    async def stop_response(self, session_id: str, context_id: str):
        # Do nothing here
        pass
