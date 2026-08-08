import json

import pytest

from aiavatar.adapter.models import AIAvatarRequest
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData
from aiavatar.sts.models import STSResponse
from aiavatar.sts.vad import SpeechDetectorDummy


class FakeVAD:
    def __init__(self):
        self.sessions = {}
        self.voiced_handler = None

    def on_voiced(self, func):
        self.voiced_handler = func
        return func

    def set_session_data(self, session_id, key, value, create_session=False):
        self.sessions.setdefault(session_id, {})[key] = value

    def get_session_data(self, session_id, key):
        return self.sessions.get(session_id, {}).get(key)


class FakePipeline:
    def __init__(self, vad=None):
        self.vad = vad or FakeVAD()
        self.response_handlers = []
        self.accepted_hook_channels = []
        self.invoked_requests = []
        self.responses = []

    def add_response_handler(self, handler):
        self.response_handlers.append(handler)

    def on_accepted(self, func=None, *, channels=None):
        def register(callback):
            self.accepted_hook_channels.append(channels)
            return callback

        if func is not None:
            return register(func)
        return register

    async def invoke(self, request):
        self.invoked_requests.append(request)
        yield STSResponse(type="final", session_id=request.session_id)

    async def handle_response(self, response):
        self.responses.append(response)


class FakeWebSocket:
    def __init__(self, *requests):
        self.requests = [json.dumps(request) for request in requests]
        self.sent_messages = []
        self.closed = False

    async def receive_text(self):
        return self.requests.pop(0)

    async def send_text(self, message):
        self.sent_messages.append(message)

    async def close(self):
        self.closed = True


def test_channel_defaults_to_websocket():
    pipeline = FakePipeline()

    server = AIAvatarWebSocketServer(sts=pipeline)

    assert server.channel == "websocket"
    assert pipeline.accepted_hook_channels == ["websocket"]


def test_aiavatar_request_does_not_expose_channel_field():
    request = AIAvatarRequest.model_validate({
        "type": "invoke",
        "channel": "client-supplied-channel",
    })

    assert "channel" not in AIAvatarRequest.model_fields
    assert "channel" not in request.model_dump()


@pytest.mark.asyncio
async def test_adapter_channel_ignores_client_value_and_sets_pipeline_channel():
    pipeline = FakePipeline()
    server = AIAvatarWebSocketServer(sts=pipeline, channel="m5stack")
    session_data = WebSocketSessionData()
    websocket = FakeWebSocket(
        {
            "type": "start",
            "session_id": "session-1",
            "channel": "client-supplied-channel",
        },
        {
            "type": "invoke",
            "session_id": "session-1",
            "channel": "another-client-channel",
            "text": "hello",
        },
    )
    request_payloads = []
    session_start_payloads = []

    @server.on_request
    async def on_request(request):
        request_payloads.append(request.model_dump())

    @server.on_session_start
    async def on_session_start(request, _session_data):
        session_start_payloads.append(request.model_dump())

    await server.process_websocket(websocket, session_data)
    await server.process_websocket(websocket, session_data)

    assert pipeline.accepted_hook_channels == ["m5stack"]
    assert all("channel" not in payload for payload in request_payloads)
    assert all("channel" not in payload for payload in session_start_payloads)
    assert pipeline.vad.get_session_data("session-1", "channel") == "m5stack"
    assert pipeline.invoked_requests[0].channel == "m5stack"


@pytest.mark.asyncio
async def test_dummy_vad_supports_invoke_only_websocket_sessions():
    pipeline = FakePipeline(vad=SpeechDetectorDummy())
    server = AIAvatarWebSocketServer(sts=pipeline, channel="websocket_ss")
    session_data = WebSocketSessionData()
    websocket = FakeWebSocket(
        {
            "type": "start",
            "session_id": "session-1",
            "user_id": "user-1",
            "context_id": "context-1",
        },
        {
            "type": "invoke",
            "session_id": "session-1",
            "user_id": "user-1",
            "text": "hello",
        },
    )

    await server.process_websocket(websocket, session_data)
    await server.process_websocket(websocket, session_data)

    assert pipeline.vad.get_session_data("session-1", "user_id") == "user-1"
    assert pipeline.vad.get_session_data("session-1", "context_id") == "context-1"
    assert pipeline.vad.get_session_data("session-1", "channel") == "websocket_ss"
    assert pipeline.invoked_requests[0].context_id == "context-1"
    assert pipeline.invoked_requests[0].channel == "websocket_ss"
