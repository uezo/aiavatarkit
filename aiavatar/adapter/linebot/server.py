from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Awaitable, Callable
from uuid import uuid4
import aiofiles
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status, Request, BackgroundTasks, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# pip install line-bot-sdk>=3.21.0
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    Configuration,
    AsyncApiClient,
    AsyncMessagingApi,
    AsyncMessagingApiBlob,
    TextMessage,
    ReplyMessageRequest
)
from linebot.v3.webhooks import (
    Event,
    MessageEvent,
    TextMessageContent,
    StickerMessageContent,
    LocationMessageContent,
    ImageMessageContent
)
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
from ..models import AvatarControlRequest
from .. import Adapter
from ..session_manager import ChannelSession, ChannelSessionManager, SQLiteChannelSessionManager

logger = logging.getLogger(__name__)


# Schema
class GetSessionResponse(BaseModel):
    session_id: str
    channel_user_id: str
    user_id: str
    context_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    data: Dict



class AIAvatarLineBotServer(Adapter):
    def __init__(
        self,
        *,
        channel_access_token: str = None,
        channel_secret: str = None,
        linebot_session_timeout: float = 3600,
        image_upload_dir: str = "linebot_images",
        image_download_url_base: str = None,
        default_error_message: str = "Error 😢",
        session_manager: ChannelSessionManager = None,

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

        # API server auth
        api_key: str = None,
        # Debug
        debug: bool = False
    ):
        # Speech-to-Speech pipeline
        self.sts = sts or STSPipeline(
            # VAD
            vad=SpeechDetectorDummy(),
            # STT
            stt = SpeechRecognizerDummy(),
            # LLM
            llm=llm,
            llm_openai_api_key=llm_openai_api_key or openai_api_key,
            llm_base_url=llm_base_url or openai_base_url,
            llm_model=llm_model or openai_model,
            llm_system_prompt=llm_system_prompt or system_prompt,
            llm_context_manager=llm_context_manager,
            # TTS
            tts=SpeechSynthesizerDummy(),
            # Pipeline
            timestamp_interval_seconds=timestamp_interval_seconds,
            timestamp_prefix=timestamp_prefix,
            timestamp_timezone=timestamp_timezone,
            db_pool_provider=db_pool_provider,
            db_connection_str=db_connection_str,
            session_state_manager=session_state_manager,
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

        # LINE API
        line_api_configuration = Configuration(
            access_token=channel_access_token
        )
        self.line_api_client = AsyncApiClient(line_api_configuration)
        self.line_api = AsyncMessagingApi(self.line_api_client)
        self.line_api_blob = AsyncMessagingApiBlob(self.line_api_client)
        self.webhook_parser = WebhookParser(channel_secret)

        # Handlers and parsers
        self._event_handlers = {
            "message": self.handle_message_event
        }
        self._default_event_handler = None
        self._message_parsers = {
            "text": self.parse_text_message,
            "image": self.parse_image_message,
            "sticker": self.parse_sticker_message,
            "location": self.parse_location_message
        }

        # Session management
        self.linebot_session_timeout = linebot_session_timeout
        self.session_manager = session_manager or SQLiteChannelSessionManager(
            db_path=db_connection_str,
            timeout=linebot_session_timeout,
        )

        # Image
        self.image_upload_dir = Path(image_upload_dir)
        self.image_download_url_base = image_download_url_base

        self._edit_channel_session = None
        self._preprocess_request = None
        self._preprocess_response = None
        self._process_avatar_control_request = None
        self._on_send_error_message = None
        self.default_error_message = default_error_message

    def get_config(self) -> dict:
        return {
            "linebot_session_timeout": self.linebot_session_timeout,
            "default_error_message": self.default_error_message,
            "debug": self.debug,
        }

    def api_key_auth(self, credentials: HTTPAuthorizationCredentials):
        if not credentials or credentials.scheme.lower() != "bearer" or credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )
        return credentials.credentials

    # Session
    def edit_channel_session(self, func: Callable[[ChannelSession], Awaitable[None]]):
        self._edit_channel_session = func
        return func

    # Message parsers
    async def parse_text_message(self, message: TextMessageContent) -> Tuple[str, bytes]:
        return message.text, None

    async def parse_image_message(self, message: ImageMessageContent) -> Tuple[str, bytes]:
        return "", await self.line_api_blob.get_message_content(message.id)

    async def parse_sticker_message(self, message: StickerMessageContent) -> Tuple[str, bytes]:
        sticker_keywords = ", ".join([k for k in message.keywords])
        return f"You received a sticker from user in messenger app: {sticker_keywords}", None

    async def parse_location_message(self, message: LocationMessageContent) -> Tuple[str, bytes]:
        return f"You received a location info from user in messenger app:\n    - address: {message.address}\n    - latitude: {message.latitude}\n    - longitude: {message.longitude}", None

    # Image persister
    async def save_image(self, user_id: str, image_bytes: bytes) -> str:
        image_id = f"{user_id}_{uuid4()}"

        if not self.image_upload_dir.exists():
            self.image_upload_dir.mkdir()
        image_path = Path(self.image_upload_dir / f"{image_id}.png")
        async with aiofiles.open(image_path, "wb") as f:
            await f.write(image_bytes)

        return image_id

    # Pre processor for request / response
    def preprocess_request(self, func: Callable[[STSRequest], Awaitable[None]]):
        self._preprocess_request = func
        return func

    def preprocess_response(self, func: Callable[[STSResponse], Awaitable[None]]):
        self._preprocess_response = func
        return func

    # Event handlers
    def event(self, event_type: str):
        def decorator(func):
            self._event_handlers[event_type] = func
            return func
        return decorator

    def default_event_handler(self, func: Callable[[MessageEvent, ChannelSession], Awaitable[None]]):
        self._default_event_handler = func
        return func

    async def handle_message_event(self, event: MessageEvent, channel_session: ChannelSession):
        if message_parser := self._message_parsers.get(event.message.type):
            text, image_bytes = await message_parser(event.message)
        else:
            logger.info(f"Unhandled message type: {event.message.type}")
            return

        files = None
        if image_bytes:
            image_id = await self.save_image(user_id=channel_session.user_id, image_bytes=image_bytes)
            files = [{"url": f"{self.image_download_url_base}/image/{image_id}"}]

        request = STSRequest(
            session_id=channel_session.session_id,
            user_id=channel_session.user_id,
            context_id=channel_session.context_id,
            text=text,
            files=files
        )

        if self._preprocess_request:
            await self._preprocess_request(request)

        async for response in self.sts.invoke(request):
            if not response.metadata:
                response.metadata = {}
            response.metadata["reply_token"] = event.reply_token
            if response.type == "final" and response.context_id:
                await self.session_manager.update_context_id("line", channel_session.channel_user_id, response.context_id)
            await self.sts.handle_response(response)

    # Processors
    async def process_webhook(self, request_body: str, signature: str):
        # NOTE: 並列処理
        for event in self.webhook_parser.parse(request_body, signature):
            await self.process_event(event)

    async def process_event(self, event: Event):
        channel_session = None
        try:
            if event_handler := self._event_handlers.get(event.type) or self._default_event_handler:
                channel_session = await self.session_manager.get_session("line", event.source.user_id)
                if self._edit_channel_session:
                    await self._edit_channel_session(channel_session)
                await event_handler(
                    event=event,
                    channel_session=channel_session
                )
            else:
                logger.info(f"Unhandled event: {event}")

        except Exception as ex:
            logger.exception(f"Error at process_event: {event}")
            await self.send_error_message(
                event=event,
                channel_session=channel_session,
                ex=ex
            )

    def on_send_error_message(self, func: Callable[[ReplyMessageRequest, ChannelSession, Event, Exception], Awaitable[None]]):
        self._on_send_error_message = func
        return func

    async def send_error_message(self, event: Event, channel_session: ChannelSession, ex: Exception):
        if not hasattr(event, "reply_token"):
            return

        reply_message_request = ReplyMessageRequest(
            replyToken=event.reply_token,
            messages=[TextMessage(text=self.default_error_message)]
        )

        if self._on_send_error_message:
            await self._on_send_error_message(reply_message_request, channel_session, event, ex)

        await self.line_api.reply_message(reply_message_request)

    # Response
    def process_avatar_control_request(self, func: Callable[[AvatarControlRequest, ReplyMessageRequest], Awaitable[None]]):
        self._process_avatar_control_request = func
        return func

    async def handle_response(self, response: STSResponse):
        if response.type == "final":
            # Message
            if self._preprocess_response:
                await self._preprocess_response(response)

            # Build reply message
            reply_message_request = ReplyMessageRequest(
                replyToken=response.metadata["reply_token"],
                messages=[TextMessage(text=response.voice_text)]
            )

            # Facical expression
            avatar_control_request = self.parse_avatar_control_request(response.text)
            if self._process_avatar_control_request:
                await self._process_avatar_control_request(avatar_control_request, reply_message_request)

            # Send reply message
            await self.line_api.reply_message(reply_message_request)

    async def stop_response(self, session_id: str, context_id: str):
        # Do nothing here
        pass

    # API Router
    def get_api_router(self):
        router = APIRouter()
        bearer_scheme = HTTPBearer(auto_error=False)

        @router.post("/webhook")
        async def post_webhook(request: Request, background_tasks: BackgroundTasks):
            background_tasks.add_task(
                self.process_webhook,
                request_body=(await request.body()).decode("utf-8"),
                signature=request.headers.get("X-Line-Signature", "")
            )
            return "ok"

        @router.get("/image/{image_id}")
        async def get_image(image_id: str):
            async with aiofiles.open(Path(self.image_upload_dir / f"{image_id}.png"), "rb") as fs:
                image_bytes = await fs.read()
                return Response(content=image_bytes, media_type="image/png")

        @router.get("/session/{line_user_id}")
        async def get_session(
            line_user_id,
            credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
        ):
            if self.api_key:
                self.api_key_auth(credentials)

            channel_session = await self.session_manager.get_session("line", line_user_id)

            return GetSessionResponse(
                session_id=channel_session.session_id,
                channel_user_id=channel_session.channel_user_id,
                user_id=channel_session.user_id,
                context_id=channel_session.context_id,
                updated_at=channel_session.updated_at,
                data=channel_session.data
            )

        @router.delete("/session/{line_user_id}")
        async def delete_session(
            line_user_id,
            credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
        ):
            if self.api_key:
                self.api_key_auth(credentials)

            await self.session_manager.delete_session("line", line_user_id)

            return JSONResponse(content={"result": "success"})

        return router
