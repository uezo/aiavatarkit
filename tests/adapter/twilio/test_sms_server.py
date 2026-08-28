import sys
from types import ModuleType, SimpleNamespace

import pytest


try:
    from twilio.rest import Client as _TwilioClient  # noqa: F401
    from twilio.twiml.voice_response import VoiceResponse as _VoiceResponse  # noqa: F401
except ModuleNotFoundError:
    twilio_module = ModuleType("twilio")
    twilio_rest_module = ModuleType("twilio.rest")
    twilio_twiml_module = ModuleType("twilio.twiml")
    twilio_voice_module = ModuleType("twilio.twiml.voice_response")

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

    class StubVoiceResponse:
        def append(self, value):
            pass

    class StubConnect:
        def stream(self, **kwargs):
            pass

    twilio_rest_module.Client = StubClient
    twilio_voice_module.VoiceResponse = StubVoiceResponse
    twilio_voice_module.Connect = StubConnect
    sys.modules["twilio"] = twilio_module
    sys.modules["twilio.rest"] = twilio_rest_module
    sys.modules["twilio.twiml"] = twilio_twiml_module
    sys.modules["twilio.twiml.voice_response"] = twilio_voice_module


from aiavatar.adapter.twilio import (
    AIAvatarTwilioServer,
    AIAvatarTwilioSMSServer,
    TwilioSessionData,
    TwilioSMSMessage,
)
from aiavatar.adapter.twilio import server as server_module
from aiavatar.sts.models import STSRequest, STSResponse


class FakePipeline:
    def __init__(self):
        self.response_handlers = []
        self.skip_tts_channels = []
        self.accepted_hook_channels = []
        self.invoked_requests = []
        self.finalized_sessions = []
        self.session_state_manager = FakeSessionStateManager()

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
        yield STSResponse(
            type="start",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
        )
        yield STSResponse(
            type="chunk",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            text="partial",
            voice_text="partial",
        )
        yield STSResponse(
            type="final",
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            text="response with control tags",
            voice_text="SMS response",
        )

    async def handle_response(self, response):
        for handler in self.response_handlers:
            if handler.can_handle(response.session_id):
                await handler.handle_response(response)
                return
        raise AssertionError(f"No response handler for {response.session_id}")

    async def finalize(self, session_id):
        self.finalized_sessions.append(session_id)


class FakeSessionStateManager:
    def __init__(self):
        self.cleared_sessions = []

    async def clear_session(self, session_id):
        self.cleared_sessions.append(session_id)


class FakeMessages:
    def __init__(self):
        self.sent = []

    def create(self, *, body, from_, to):
        self.sent.append({"body": body, "from": from_, "to": to})
        return SimpleNamespace(sid="SM-outbound")


class FakeTwilioClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_default_pipeline_receives_llm_generation_params(monkeypatch):
    captured = {}

    def create_pipeline(**kwargs):
        captured.update(kwargs)
        return FakePipeline()

    monkeypatch.setattr(server_module, "STSPipeline", create_pipeline)
    AIAvatarTwilioServer(
        stt=object(),
        llm_temperature=0.0,
        llm_reasoning_effort="none",
    )

    assert captured["llm_model"] == "gpt-5.6-terra"
    assert captured["llm_temperature"] == 0.0
    assert captured["llm_reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_voice_server_overrides_request_channel_with_adapter_channel():
    pipeline = FakePipeline()
    phone = AIAvatarTwilioServer(
        sts=pipeline,
        webhook_base_url="https://example.com/twilio",
        channel="support-phone",
    )
    phone.sessions["phone-session-1"] = TwilioSessionData()

    await phone.invoke(STSRequest(
        session_id="phone-session-1",
        channel="sms",
        text="Hello",
    ))

    assert pipeline.invoked_requests[0].channel == "support-phone"


@pytest.mark.asyncio
async def test_sms_server_routes_shared_pipeline_response_to_sms():
    pipeline = FakePipeline()
    client = FakeTwilioClient()
    phone = AIAvatarTwilioServer(
        sts=pipeline,
        twilio_client=client,
        phone_number="+10000000000",
        webhook_base_url="https://example.com/twilio",
    )
    sms = AIAvatarTwilioSMSServer(
        sts=pipeline,
        twilio_client=client,
        phone_number="+10000000000",
    )
    received_messages = []
    callback_types = []
    handled_while_registered = []

    @sms.on_sms_received
    async def on_sms_received(message):
        received_messages.append(message.message_sid)

    @sms.on_session_start
    async def on_session_start(request, message):
        assert message.message_sid == "SM-inbound"
        request.user_id = "application-user"
        request.context_id = "context-1"

    @sms.on_response
    async def on_response(aiavatar_response, sts_response):
        callback_types.append(aiavatar_response.type)
        handled_while_registered.append(sms.can_handle(sts_response.session_id))

    await sms.process_message(
        TwilioSMSMessage(
            message_sid="SM-inbound",
            from_number="+12222222222",
            to_number="+10000000000",
            body="Hello",
        ),
        session_id="sms-session-1",
    )

    assert len(pipeline.response_handlers) == 2
    assert pipeline.accepted_hook_channels == ["phone"]
    assert pipeline.skip_tts_channels == ["sms"]
    assert received_messages == ["SM-inbound"]
    assert callback_types == ["start", "chunk", "final"]
    assert all(handled_while_registered)
    assert not phone.can_handle("sms-session-1")
    assert not sms.can_handle("sms-session-1")
    assert pipeline.finalized_sessions == ["sms-session-1"]
    assert pipeline.session_state_manager.cleared_sessions == ["sms-session-1"]

    invoked = pipeline.invoked_requests[0]
    assert invoked.session_id == "sms-session-1"
    assert invoked.user_id == "application-user"
    assert invoked.context_id == "context-1"
    assert invoked.channel == "sms"
    assert invoked.skip_quick_response is True

    assert client.messages.sent == [{
        "body": "SMS response",
        "from": "+10000000000",
        "to": "+12222222222",
    }]

    assert "/sms" not in {route.path for route in phone.get_router().routes}
    assert {route.path for route in sms.get_router().routes} == {"/sms", "/sms/send"}


@pytest.mark.asyncio
async def test_sms_server_push_uses_internal_user_and_sms_response_handler():
    pipeline = FakePipeline()
    client = FakeTwilioClient()
    sms = AIAvatarTwilioSMSServer(
        sts=pipeline,
        twilio_client=client,
        phone_number="+10000000000",
    )
    session_start_calls = []

    @sms.on_session_start
    async def on_session_start(request, message):
        session_start_calls.append((request.user_id, message.from_number))

    await sms.handle_push_request(
        user_id="application-user",
        context_id="context-1",
        text="Notify the user",
        to="+12222222222",
        session_id="sms-push-session-1",
    )

    assert session_start_calls == []
    assert not sms.can_handle("sms-push-session-1")
    assert pipeline.finalized_sessions == ["sms-push-session-1"]
    assert pipeline.session_state_manager.cleared_sessions == ["sms-push-session-1"]

    invoked = pipeline.invoked_requests[0]
    assert invoked.user_id == "application-user"
    assert invoked.context_id == "context-1"
    assert invoked.channel == "sms"
    assert client.messages.sent == [{
        "body": "SMS response",
        "from": "+10000000000",
        "to": "+12222222222",
    }]


@pytest.mark.asyncio
async def test_sms_server_clears_state_when_finalize_fails():
    pipeline = FakePipeline()
    client = FakeTwilioClient()
    sms = AIAvatarTwilioSMSServer(
        sts=pipeline,
        twilio_client=client,
        phone_number="+10000000000",
    )

    async def fail_finalize(session_id):
        pipeline.finalized_sessions.append(session_id)
        raise RuntimeError("finalize failed")

    pipeline.finalize = fail_finalize

    with pytest.raises(RuntimeError, match="finalize failed"):
        await sms.process_message(
            TwilioSMSMessage(
                message_sid="SM-inbound",
                from_number="+12222222222",
                to_number="+10000000000",
                body="Hello",
            ),
            session_id="sms-session-finalize-error",
        )

    assert not sms.can_handle("sms-session-finalize-error")
    assert pipeline.session_state_manager.cleared_sessions == ["sms-session-finalize-error"]
