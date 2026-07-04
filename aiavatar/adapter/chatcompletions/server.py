import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from ...database import PoolProvider
from ...sts import STSPipeline
from ...sts.models import STSRequest, STSResponse
from ...sts.vad import SpeechDetectorDummy
from ...sts.stt import SpeechRecognizerDummy
from ...sts.llm import LLMService
from ...sts.llm.context_manager import ContextManager
from ...sts.tts import SpeechSynthesizerDummy
from ...sts.session_state_manager import SessionStateManager
from ...sts.performance_recorder import PerformanceRecorder
from .. import Adapter
from ..channel_context_bridge import ChannelContextBridge, SQLiteChannelContextBridge, UserContext

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: Optional[Union[str, List[Union[str, Dict[str, Any]]]]] = None


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "openclaw"
    messages: List[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    user: Optional[str] = None
    files: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None


class ChatCompletionDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class AIAvatarChatCompletionsServer(Adapter):
    def __init__(
        self,
        *,
        # Quick start
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-4.1",
        system_prompt: str = None,

        # STS Pipeline and its components
        sts: STSPipeline = None,
        llm: LLMService = None,

        # STS Pipeline params for default components
        llm_openai_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "gpt-4.1",
        llm_system_prompt: str = None,
        llm_context_manager: ContextManager = None,
        timestamp_interval_seconds: float = 0.0,
        timestamp_prefix: str = "$Current date and time: ",
        timestamp_timezone: str = "UTC",
        db_pool_provider: PoolProvider = None,
        db_connection_str: str = "aiavatar.db",
        session_state_manager: SessionStateManager = None,
        performance_recorder: PerformanceRecorder = None,
        invoke_queue_idle_timeout: float = 10.0,
        invoke_timeout: float = 60.0,
        use_invoke_queue: bool = False,

        # Chat Completions adapter
        channel_context_bridge: ChannelContextBridge = None,
        channel_id: str = "chatcompletions",
        session_timeout: float = 3600,
        insert_channel_tag: bool = False,
        response_content_field: str = "voice_text",

        # Debug
        debug: bool = False
    ):
        if response_content_field not in ("text", "voice_text"):
            raise ValueError("response_content_field must be 'text' or 'voice_text'")

        logger.warning(
            "AIAvatarChatCompletionsServer is an experimental implementation and should not be used in production."
        )

        self.sts = sts or STSPipeline(
            vad=SpeechDetectorDummy(),
            stt=SpeechRecognizerDummy(),
            llm=llm,
            llm_openai_api_key=llm_openai_api_key or openai_api_key,
            llm_base_url=llm_base_url or openai_base_url,
            llm_model=llm_model or openai_model,
            llm_system_prompt=llm_system_prompt or system_prompt,
            llm_context_manager=llm_context_manager,
            tts=SpeechSynthesizerDummy(),
            timestamp_interval_seconds=timestamp_interval_seconds,
            timestamp_prefix=timestamp_prefix,
            timestamp_timezone=timestamp_timezone,
            db_pool_provider=db_pool_provider,
            db_connection_str=db_connection_str,
            session_state_manager=session_state_manager,
            performance_recorder=performance_recorder,
            invoke_queue_idle_timeout=invoke_queue_idle_timeout,
            invoke_timeout=invoke_timeout,
            use_invoke_queue=use_invoke_queue,
            insert_channel_tag=insert_channel_tag,
            skip_tts_channels=[channel_id],
            debug=debug
        )

        super().__init__(self.sts)

        self.debug = debug
        self.channel_id = channel_id
        self.session_timeout = session_timeout
        self.response_content_field = response_content_field
        self.channel_context_bridge = channel_context_bridge or SQLiteChannelContextBridge(
            db_path=db_connection_str,
            timeout=session_timeout,
        )
        self._bearer_scheme = HTTPBearer(auto_error=False)

    def get_config(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "session_timeout": self.session_timeout,
            "response_content_field": self.response_content_field,
            "debug": self.debug,
        }

    def _get_response_content(self, response: STSResponse) -> str:
        value = getattr(response, self.response_content_field, None)
        if value is None and self.response_content_field == "voice_text":
            value = response.text
        return value or ""

    def _get_bearer_token(self, credentials: HTTPAuthorizationCredentials) -> str:
        if not credentials or credentials.scheme.lower() != "bearer" or not credentials.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token",
            )
        return credentials.credentials

    def _extract_text(self, request: ChatCompletionsRequest) -> str:
        user_messages = [m for m in request.messages if m.role == "user"]
        message = user_messages[-1] if user_messages else (request.messages[-1] if request.messages else None)
        if not message:
            raise HTTPException(status_code=400, detail="messages is required")

        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item.get("text"), str):
                        parts.append(item["text"])
            return "\n".join([p for p in parts if p])
        return ""

    def _make_chunk(
        self,
        *,
        completion_id: str,
        created: int,
        model: str,
        delta: ChatCompletionDelta,
        finish_reason: Optional[str] = None,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)]
        )

    async def _build_sts_request(
        self,
        chat_request: ChatCompletionsRequest,
        token: str,
    ) -> STSRequest:
        channel_user = await self.channel_context_bridge.get_channel_user(
            self.channel_id,
            token,
            auto_create=True,
        )
        user_context = await self.channel_context_bridge.get_context(channel_user.user_id)
        text = self._extract_text(chat_request)

        return STSRequest(
            type="start",
            session_id=str(uuid4()),
            user_id=channel_user.user_id,
            context_id=user_context.context_id if user_context else None,
            text=text,
            files=chat_request.files,
            channel=self.channel_id,
            metadata={
                "chatcompletions_token": token,
                **(chat_request.metadata or {}),
            },
        )

    async def _invoke_non_streaming(
        self,
        request: STSRequest,
        *,
        completion_id: str,
        created: int,
        model: str,
    ) -> ChatCompletionResponse:
        content = ""
        final_context_id = request.context_id
        async for response in self.sts.invoke(request):
            for on_resp in self._on_response_handlers:
                await on_resp(None, response)

            if response.type == "chunk":
                content += self._get_response_content(response)
            elif response.type == "final":
                final_context_id = response.context_id
                final_content = self._get_response_content(response)
                content = final_content if final_content else content
            elif response.type == "error":
                raise HTTPException(status_code=500, detail=response.metadata or {"error": "Error in STS pipeline"})
            elif response.type == "stop":
                await self.stop_response(response.session_id, response.context_id)

        if final_context_id:
            await self.channel_context_bridge.upsert_context(UserContext(
                user_id=request.user_id,
                context_id=final_context_id,
            ))

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=content),
                    finish_reason="stop",
                )
            ],
        )

    def get_api_router(self, path: str = "/v1/chat/completions"):
        router = APIRouter()

        @router.post(path)
        async def post_chat_completions(
            chat_request: ChatCompletionsRequest,
            credentials: HTTPAuthorizationCredentials = Depends(self._bearer_scheme),
        ):
            token = self._get_bearer_token(credentials)
            sts_request = await self._build_sts_request(chat_request, token)

            for on_req in self._on_request_handlers:
                await on_req(sts_request)

            completion_id = f"chatcmpl-{uuid4()}"
            created = int(time.time())

            if not chat_request.stream:
                return await self._invoke_non_streaming(
                    sts_request,
                    completion_id=completion_id,
                    created=created,
                    model=chat_request.model,
                )

            async def stream_response():
                yield self._make_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=chat_request.model,
                    delta=ChatCompletionDelta(role="assistant"),
                ).model_dump_json(exclude_none=True)

                final_context_id = sts_request.context_id
                async for response in self.sts.invoke(sts_request):
                    for on_resp in self._on_response_handlers:
                        await on_resp(None, response)

                    if response.type == "chunk":
                        yield self._make_chunk(
                            completion_id=completion_id,
                            created=created,
                            model=chat_request.model,
                            delta=ChatCompletionDelta(content=self._get_response_content(response)),
                        ).model_dump_json(exclude_none=True)
                    elif response.type == "final":
                        final_context_id = response.context_id
                    elif response.type == "error":
                        yield json.dumps({
                            "error": response.metadata or {"message": "Error in STS pipeline"}
                        }, ensure_ascii=False)
                    elif response.type == "stop":
                        await self.stop_response(response.session_id, response.context_id)

                if final_context_id:
                    await self.channel_context_bridge.upsert_context(UserContext(
                        user_id=sts_request.user_id,
                        context_id=final_context_id,
                    ))

                yield self._make_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=chat_request.model,
                    delta=ChatCompletionDelta(),
                    finish_reason="stop",
                ).model_dump_json(exclude_none=True)
                yield "[DONE]"

            return EventSourceResponse(stream_response())

        return router

    async def handle_response(self, response: STSResponse):
        pass

    async def stop_response(self, session_id: str, context_id: str):
        pass
