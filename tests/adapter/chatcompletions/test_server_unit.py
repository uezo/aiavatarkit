from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from aiavatar.adapter.chatcompletions import server as server_module
from aiavatar.adapter.chatcompletions.server import (
    AIAvatarChatCompletionsServer,
    ChatCompletionsRequest,
    ChatMessage,
)
from aiavatar.sts.models import STSResponse


class FakeSessionStateManager:
    def __init__(self):
        self.cleared_sessions = []

    async def clear_session(self, session_id):
        self.cleared_sessions.append(session_id)


class FakePipeline:
    def __init__(self, fail_on_invoke=False):
        self.response_handlers = []
        self.skip_tts_channels = ["phone"]
        self.invoked_requests = []
        self.finalized_sessions = []
        self.handler_owned_during_invoke = []
        self.session_state_manager = FakeSessionStateManager()
        self.fail_on_invoke = fail_on_invoke

    def add_response_handler(self, handler):
        self.response_handlers.append(handler)

    async def handle_response(self, response):
        for handler in self.response_handlers:
            if handler.can_handle(response.session_id):
                await handler.handle_response(response)
                return
        raise AssertionError(f"No response handler for {response.session_id}")

    async def invoke(self, request):
        self.invoked_requests.append(request)
        self.handler_owned_during_invoke.append(any(
            handler.can_handle(request.session_id)
            for handler in self.response_handlers
        ))
        await self.handle_response(STSResponse(
            type="accepted",
            session_id=request.session_id,
        ))
        if self.fail_on_invoke:
            raise RuntimeError("invoke failed")
        yield STSResponse(
            type="chunk",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            text="chunk text",
            voice_text="chunk voice",
        )
        yield STSResponse(
            type="final",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id="context-updated",
            text="final text",
            voice_text="final voice",
        )

    async def finalize(self, session_id):
        self.finalized_sessions.append(session_id)


class ForeignResponseHandler:
    def __init__(self):
        self.responses = []

    def can_handle(self, session_id):
        return session_id.startswith("foreign_")

    async def handle_response(self, response):
        self.responses.append(response)


class FakeChannelContextBridge:
    def __init__(self):
        self.channel_user_requests = []
        self.contexts = []

    async def get_channel_user(self, channel_id, channel_user_id, auto_create=False):
        self.channel_user_requests.append((channel_id, channel_user_id, auto_create))
        return SimpleNamespace(user_id="application-user")

    async def get_context(self, user_id):
        return SimpleNamespace(context_id="context-current")

    async def upsert_context(self, context):
        self.contexts.append(context)


def get_endpoint(server):
    router = server.get_api_router()
    return next(
        route.endpoint
        for route in router.routes
        if route.path == "/v1/chat/completions"
    )


def credentials():
    return HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="external-user-token",
    )


def assert_chatcompletions_session_id(session_id):
    prefix, value = session_id.split("_", 1)
    assert prefix == "chatcompletions"
    UUID(value)


def test_default_pipeline_receives_llm_generation_params(monkeypatch):
    captured = {}

    def create_pipeline(**kwargs):
        captured.update(kwargs)
        return FakePipeline()

    monkeypatch.setattr(server_module, "STSPipeline", create_pipeline)
    AIAvatarChatCompletionsServer(
        llm_temperature=0.0,
        llm_reasoning_effort="none",
        channel_context_bridge=FakeChannelContextBridge(),
    )

    assert captured["llm_model"] == "gpt-5.6-terra"
    assert captured["llm_temperature"] == 0.0
    assert captured["llm_reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_non_streaming_response_owns_session_on_shared_pipeline():
    pipeline = FakePipeline()
    foreign_handler = ForeignResponseHandler()
    pipeline.response_handlers.append(foreign_handler)
    bridge = FakeChannelContextBridge()
    server = AIAvatarChatCompletionsServer(
        sts=pipeline,
        channel_context_bridge=bridge,
    )

    response = await get_endpoint(server)(
        ChatCompletionsRequest(messages=[ChatMessage(role="user", content="Hello")]),
        credentials(),
    )

    request = pipeline.invoked_requests[0]
    assert_chatcompletions_session_id(request.session_id)
    assert request.channel == "chatcompletions"
    assert request.user_id == "application-user"
    assert request.context_id == "context-current"
    assert request.metadata["chatcompletions_token"] == "external-user-token"
    assert pipeline.skip_tts_channels == ["phone", "chatcompletions"]
    assert pipeline.handler_owned_during_invoke == [True]
    assert foreign_handler.responses == []
    assert response.choices[0].message.content == "final voice"
    assert bridge.contexts[0].context_id == "context-updated"
    assert pipeline.finalized_sessions == [request.session_id]
    assert pipeline.session_state_manager.cleared_sessions == [request.session_id]
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_streaming_response_owns_session_until_done():
    pipeline = FakePipeline()
    bridge = FakeChannelContextBridge()
    server = AIAvatarChatCompletionsServer(
        sts=pipeline,
        channel_context_bridge=bridge,
    )

    response = await get_endpoint(server)(
        ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        ),
        credentials(),
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

    request = pipeline.invoked_requests[0]
    body = "".join(chunks)
    assert_chatcompletions_session_id(request.session_id)
    assert pipeline.handler_owned_during_invoke == [True]
    assert '"role":"assistant"' in body
    assert '"content":"chunk voice"' in body
    assert '"finish_reason":"stop"' in body
    assert "[DONE]" in body
    assert bridge.contexts[0].context_id == "context-updated"
    assert pipeline.finalized_sessions == [request.session_id]
    assert pipeline.session_state_manager.cleared_sessions == [request.session_id]
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_streaming_response_releases_session_when_client_closes():
    pipeline = FakePipeline()
    server = AIAvatarChatCompletionsServer(
        sts=pipeline,
        channel_context_bridge=FakeChannelContextBridge(),
    )

    response = await get_endpoint(server)(
        ChatCompletionsRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True,
        ),
        credentials(),
    )
    iterator = response.body_iterator
    await anext(iterator)

    session_id = next(iter(server.sessions))
    assert server.can_handle(session_id)

    await iterator.aclose()

    assert pipeline.finalized_sessions == [session_id]
    assert pipeline.session_state_manager.cleared_sessions == [session_id]
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_non_streaming_response_releases_session_when_invoke_fails():
    pipeline = FakePipeline(fail_on_invoke=True)
    server = AIAvatarChatCompletionsServer(
        sts=pipeline,
        channel_context_bridge=FakeChannelContextBridge(),
    )

    with pytest.raises(RuntimeError, match="invoke failed"):
        await get_endpoint(server)(
            ChatCompletionsRequest(messages=[ChatMessage(role="user", content="Hello")]),
            credentials(),
        )

    session_id = pipeline.invoked_requests[0].session_id
    assert pipeline.finalized_sessions == [session_id]
    assert pipeline.session_state_manager.cleared_sessions == [session_id]
    assert server.sessions == {}
