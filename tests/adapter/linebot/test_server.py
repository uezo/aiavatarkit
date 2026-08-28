import sys
from types import ModuleType, SimpleNamespace
from uuid import UUID

import pytest
import pytest_asyncio


try:
    from linebot.v3 import WebhookParser as _WebhookParser  # noqa: F401
except ModuleNotFoundError:
    linebot_module = ModuleType("linebot")
    linebot_v3_module = ModuleType("linebot.v3")
    linebot_messaging_module = ModuleType("linebot.v3.messaging")
    linebot_webhooks_module = ModuleType("linebot.v3.webhooks")

    class StubModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            if "replyToken" in kwargs:
                self.reply_token = kwargs["replyToken"]

    class StubWebhookParser:
        def __init__(self, channel_secret):
            self.channel_secret = channel_secret

    class StubConfiguration:
        def __init__(self, access_token=None):
            self.access_token = access_token

    class StubAsyncApiClient:
        def __init__(self, configuration):
            self.configuration = configuration

    class StubMessagingApi:
        def __init__(self, api_client):
            self.api_client = api_client

    linebot_v3_module.WebhookParser = StubWebhookParser
    for name, value in {
        "Configuration": StubConfiguration,
        "AsyncApiClient": StubAsyncApiClient,
        "AsyncMessagingApi": StubMessagingApi,
        "AsyncMessagingApiBlob": StubMessagingApi,
        "TextMessage": StubModel,
        "ReplyMessageRequest": StubModel,
        "PushMessageRequest": StubModel,
    }.items():
        setattr(linebot_messaging_module, name, value)
    for name in (
        "Event",
        "MessageEvent",
        "TextMessageContent",
        "StickerMessageContent",
        "LocationMessageContent",
        "ImageMessageContent",
    ):
        setattr(linebot_webhooks_module, name, StubModel)

    linebot_module.v3 = linebot_v3_module
    sys.modules["linebot"] = linebot_module
    sys.modules["linebot.v3"] = linebot_v3_module
    sys.modules["linebot.v3.messaging"] = linebot_messaging_module
    sys.modules["linebot.v3.webhooks"] = linebot_webhooks_module

from aiavatar.adapter.linebot import server as server_module
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer
from aiavatar.sts.models import STSResponse


class FakeSessionStateManager:
    def __init__(self):
        self.cleared_sessions = []

    async def clear_session(self, session_id):
        self.cleared_sessions.append(session_id)


class FakePipeline:
    def __init__(self, fail_on_invoke=False):
        self.response_handlers = []
        self.skip_tts_channels = []
        self.invoked_requests = []
        self.finalized_sessions = []
        self.handler_owned_during_invoke = []
        self.session_state_manager = FakeSessionStateManager()
        self.fail_on_invoke = fail_on_invoke

    def add_response_handler(self, handler):
        self.response_handlers.append(handler)

    async def invoke(self, request):
        self.invoked_requests.append(request)
        self.handler_owned_during_invoke.append(any(
            handler.can_handle(request.session_id)
            for handler in self.response_handlers
        ))
        if self.fail_on_invoke:
            raise RuntimeError("invoke failed")
        for response_type in ("start", "chunk", "final"):
            yield STSResponse(
                type=response_type,
                session_id=request.session_id,
                user_id=request.user_id,
                context_id="context-updated",
                text="LINE response",
                voice_text="LINE voice response",
            )

    async def handle_response(self, response):
        for handler in self.response_handlers:
            if handler.can_handle(response.session_id):
                await handler.handle_response(response)
                return
        raise AssertionError(f"No response handler for {response.session_id}")

    async def finalize(self, session_id):
        self.finalized_sessions.append(session_id)


class ForeignResponseHandler:
    def __init__(self):
        self.responses = []

    def can_handle(self, session_id):
        return session_id.startswith("foreign_")

    async def handle_response(self, response):
        self.responses.append(response)


class FakeLineApi:
    def __init__(self):
        self.replies = []
        self.pushes = []

    async def reply_message(self, request):
        self.replies.append(request)

    async def push_message(self, request):
        self.pushes.append(request)


class FakeChannelContextBridge:
    def __init__(self):
        self.contexts = []
        self.channel_users = []

    async def get_context(self, user_id):
        return SimpleNamespace(context_id="context-current")

    async def upsert_context(self, context):
        self.contexts.append(context)

    async def find_channel_users(self, user_id):
        return self.channel_users


def assert_linebot_session_id(session_id):
    prefix, value = session_id.split("_", 1)
    assert prefix == "linebot"
    UUID(value)


@pytest_asyncio.fixture
async def linebot_server_factory():
    servers = []

    def create(**kwargs):
        server = AIAvatarLineBotServer(**kwargs)
        servers.append(server)
        return server

    yield create

    for server in servers:
        if close := getattr(server.line_api_client, "close", None):
            await close()


@pytest.mark.asyncio
async def test_default_pipeline_receives_llm_generation_params(
    monkeypatch,
    linebot_server_factory,
):
    captured = {}

    def create_pipeline(**kwargs):
        captured.update(kwargs)
        return FakePipeline()

    monkeypatch.setattr(server_module, "STSPipeline", create_pipeline)
    linebot_server_factory(
        llm_temperature=0.0,
        llm_reasoning_effort="none",
        channel_access_token="test-channel-access-token",
        channel_secret="test-channel-secret",
        channel_context_bridge=FakeChannelContextBridge(),
    )

    assert captured["llm_model"] == "gpt-5.6-terra"
    assert captured["llm_temperature"] == 0.0
    assert captured["llm_reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_linebot_reply_routes_shared_pipeline_response(linebot_server_factory):
    pipeline = FakePipeline()
    foreign_handler = ForeignResponseHandler()
    pipeline.response_handlers.append(foreign_handler)
    bridge = FakeChannelContextBridge()
    server = linebot_server_factory(
        sts=pipeline,
        channel_access_token="test-channel-access-token",
        channel_secret="test-channel-secret",
        channel_context_bridge=bridge,
    )
    server.line_api = FakeLineApi()
    callback_types = []

    @server.on_response
    async def on_response(aiavatar_response, sts_response):
        callback_types.append(aiavatar_response.type)
        assert server.can_handle(sts_response.session_id)

    event = SimpleNamespace(
        message=SimpleNamespace(type="text", text="Hello"),
        reply_token="reply-token",
    )
    await server.handle_message_event(event, user_id="user-1", context_id="context-1")

    request = pipeline.invoked_requests[0]
    assert_linebot_session_id(request.session_id)
    assert request.channel == "linebot"
    assert pipeline.skip_tts_channels == ["linebot"]
    assert pipeline.handler_owned_during_invoke == [True]
    assert foreign_handler.responses == []
    assert callback_types == ["start", "chunk", "final"]
    assert len(server.line_api.replies) == 1
    assert server.line_api.replies[0].reply_token == "reply-token"
    assert server.line_api.replies[0].messages[0].text == "LINE voice response"
    assert server.line_api.pushes == []
    assert bridge.contexts[0].context_id == "context-updated"
    assert pipeline.finalized_sessions == [request.session_id]
    assert pipeline.session_state_manager.cleared_sessions == [request.session_id]
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_linebot_direct_reply_routes_shared_pipeline_response(
    linebot_server_factory,
):
    pipeline = FakePipeline()
    bridge = FakeChannelContextBridge()
    server = linebot_server_factory(
        sts=pipeline,
        channel_access_token="test-channel-access-token",
        channel_secret="test-channel-secret",
        channel_context_bridge=bridge,
    )
    server.line_api = FakeLineApi()

    await server.handle_reply_request(
        reply_token="follow-reply-token",
        user_id="user-1",
        context_id="context-1",
        text="Follow greeting",
    )

    request = pipeline.invoked_requests[0]
    assert request.text == "Follow greeting"
    assert request.channel == "linebot"
    assert len(server.line_api.replies) == 1
    assert server.line_api.replies[0].reply_token == "follow-reply-token"
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_linebot_push_routes_response_to_resolved_user(linebot_server_factory):
    pipeline = FakePipeline()
    bridge = FakeChannelContextBridge()
    bridge.channel_users = [SimpleNamespace(
        channel_id="linebot",
        channel_user_id="U-line-user",
    )]
    server = linebot_server_factory(
        sts=pipeline,
        channel_access_token="test-channel-access-token",
        channel_secret="test-channel-secret",
        channel_context_bridge=bridge,
    )
    server.line_api = FakeLineApi()

    await server.handle_push_request(user_id="user-1", text="Hello")

    request = pipeline.invoked_requests[0]
    assert_linebot_session_id(request.session_id)
    assert request.context_id == "context-current"
    assert len(server.line_api.pushes) == 1
    assert server.line_api.pushes[0].to == "U-line-user"
    assert server.line_api.pushes[0].messages[0].text == "LINE voice response"
    assert server.line_api.replies == []
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_linebot_releases_session_when_invoke_fails(linebot_server_factory):
    pipeline = FakePipeline(fail_on_invoke=True)
    server = linebot_server_factory(
        sts=pipeline,
        channel_access_token="test-channel-access-token",
        channel_secret="test-channel-secret",
        channel_context_bridge=FakeChannelContextBridge(),
    )
    server.line_api = FakeLineApi()
    event = SimpleNamespace(
        message=SimpleNamespace(type="text", text="Hello"),
        reply_token="reply-token",
    )

    with pytest.raises(RuntimeError, match="invoke failed"):
        await server.handle_message_event(event, user_id="user-1", context_id=None)

    session_id = pipeline.invoked_requests[0].session_id
    assert pipeline.finalized_sessions == [session_id]
    assert pipeline.session_state_manager.cleared_sessions == [session_id]
    assert server.sessions == {}
