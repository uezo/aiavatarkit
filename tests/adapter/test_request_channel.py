import asyncio
from types import SimpleNamespace

import pytest

from aiavatar.adapter.http.server import (
    AIAvatarHttpServer,
    PostChatMessagesRequest,
    PostSpeakerNameRequest,
)
from aiavatar.adapter.local.server import AIAvatarLocalServer
from aiavatar.adapter.models import AIAvatarRequest
from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.stt.speaker_registry.base import Candidate, MatchTopKResult


class FakePipeline:
    def __init__(self, fail_on_invoke=False):
        self.response_handlers = []
        self.invoked_requests = []
        self.handler_owned_during_invoke = []
        self.dispatched_responses = []
        self.fail_on_invoke = fail_on_invoke

    def add_response_handler(self, handler):
        self.response_handlers.append(handler)

    async def handle_response(self, response):
        for handler in self.response_handlers:
            if handler.can_handle(response.session_id):
                self.dispatched_responses.append(response)
                await handler.handle_response(response)
                return

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
            type="final",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            text="done",
        )


class FakeSpeakerRegistry:
    def __init__(self):
        self.match_calls = []
        self.metadata_calls = []

    async def match_topk_from_pcm(self, audio_bytes, sample_rate):
        self.match_calls.append((audio_bytes, sample_rate))
        return MatchTopKResult(
            chosen=Candidate("speaker-1", 1.0, {}, is_new=True),
            candidates=[],
        )

    async def set_metadata(self, speaker_id, key, value):
        self.metadata_calls.append((speaker_id, key, value))


class FakeSpeechRecognizer:
    sample_rate = 16000

    async def recognize(self, session_id, data):
        return SimpleNamespace(
            text="hello",
            preprocess_metadata={},
            postprocess_metadata={},
        )


class FakeUpload:
    async def read(self):
        return b"audio"


@pytest.mark.asyncio
async def test_http_adapter_sets_its_channel_on_pipeline_request():
    pipeline = FakePipeline()
    server = AIAvatarHttpServer(sts=pipeline, channel="internal-api")
    router = server.get_api_router()
    endpoint = next(route.endpoint for route in router.routes if route.path == "/chat")
    response = await endpoint(
        AIAvatarRequest(type="invoke", session_id="session-1", text="hello"),
        None,
    )

    async for _ in response.body_iterator:
        pass

    assert server.channel == "internal-api"
    assert pipeline.invoked_requests[0].channel == "internal-api"
    assert pipeline.handler_owned_during_invoke == [True]
    assert [response.type for response in pipeline.dispatched_responses] == ["accepted"]
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_http_adapter_awaits_speaker_registry_calls():
    speaker_registry = FakeSpeakerRegistry()
    server = AIAvatarHttpServer(
        sts=FakePipeline(),
        speaker_registry=speaker_registry,
    )
    router = server.get_api_router(stt=FakeSpeechRecognizer())
    transcribe_endpoint = next(
        route.endpoint for route in router.routes if route.path == "/transcribe"
    )
    speaker_endpoint = next(
        route.endpoint
        for route in router.routes
        if route.path == "/transcribe/speaker"
    )

    response = await transcribe_endpoint(FakeUpload(), None, None)
    await speaker_endpoint(PostSpeakerNameRequest(speaker_id="speaker-1", name="Alice"), None)

    assert response.speakers.chosen.speaker_id == "speaker-1"
    assert speaker_registry.match_calls == [(b"audio", 16000)]
    assert speaker_registry.metadata_calls == [("speaker-1", "name", "Alice")]


@pytest.mark.asyncio
async def test_http_adapter_owns_dify_session_only_while_streaming():
    pipeline = FakePipeline()
    server = AIAvatarHttpServer(sts=pipeline)
    router = server.get_api_router()
    endpoint = next(route.endpoint for route in router.routes if route.path == "/chat-messages")
    response = await endpoint(
        PostChatMessagesRequest(
            query="hello",
            user="user-1",
            conversation_id="conversation-1",
        ),
        None,
    )

    async for _ in response.body_iterator:
        pass

    assert pipeline.invoked_requests[0].session_id == "conversation-1"
    assert pipeline.invoked_requests[0].channel == "http"
    assert pipeline.handler_owned_during_invoke == [True]
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_http_adapter_releases_session_when_stream_fails():
    pipeline = FakePipeline(fail_on_invoke=True)
    server = AIAvatarHttpServer(sts=pipeline)
    router = server.get_api_router()
    endpoint = next(route.endpoint for route in router.routes if route.path == "/chat")
    response = await endpoint(
        AIAvatarRequest(type="invoke", session_id="session-1", text="hello"),
        None,
    )

    with pytest.raises(RuntimeError, match="invoke failed"):
        async for _ in response.body_iterator:
            pass

    assert pipeline.handler_owned_during_invoke == [True]
    assert server.sessions == {}


def test_http_adapter_reference_counts_overlapping_sessions():
    server = AIAvatarHttpServer(sts=FakePipeline())

    server._register_session("session-1")
    server._register_session("session-1")
    server._unregister_session("session-1")

    assert server.can_handle("session-1")
    assert server.sessions == {"session-1": 1}

    server._unregister_session("session-1")

    assert not server.can_handle("session-1")
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_http_adapter_server_side_invoke_owns_session_and_sets_channel():
    pipeline = FakePipeline()
    server = AIAvatarHttpServer(sts=pipeline, channel="restapi")

    responses = [response async for response in server.invoke(STSRequest(
        session_id="restapi-session-1",
        user_id="user-1",
        text="hello",
    ))]

    assert responses[-1].type == "final"
    assert pipeline.invoked_requests[0].channel == "restapi"
    assert pipeline.handler_owned_during_invoke == [True]
    assert [response.type for response in pipeline.dispatched_responses] == ["accepted"]
    assert server.sessions == {}


@pytest.mark.asyncio
async def test_local_adapter_sets_its_channel_on_pipeline_request():
    pipeline = FakePipeline()
    with pytest.warns(DeprecationWarning):
        server = AIAvatarLocalServer(
            asyncio.Queue(),
            sts=pipeline,
            channel="legacy-local",
        )

    await server.send_request(
        AIAvatarRequest(type="invoke", session_id="session-1", text="hello")
    )

    assert server.channel == "legacy-local"
    assert pipeline.invoked_requests[0].channel == "legacy-local"
    assert pipeline.handler_owned_during_invoke == [True]
    assert server.can_handle("session-1")
