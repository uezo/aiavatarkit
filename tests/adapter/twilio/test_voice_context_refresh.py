from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from test_sms_server import (
    AIAvatarTwilioServer,
    FakePipeline,
    STSRequest,
    STSResponse,
    TwilioSessionData,
    server_module,
)


class RefreshPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.accepted_hooks = []

    def on_accepted(self, func=None, *, channels=None):
        def register(callback):
            self.accepted_hooks.append(callback)
            return callback

        return register(func) if func else register


@pytest.fixture
def refresh_server(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(server_module, "monotonic", lambda: clock.now)
    values = {"user_id": "app-user", "context_id": "A"}
    pipeline = RefreshPipeline()
    pipeline.vad = SimpleNamespace(
        get_session_data=lambda session_id, key: values.get(key),
        set_session_data=lambda session_id, key, value: values.__setitem__(key, value),
    )
    phone = AIAvatarTwilioServer(sts=pipeline)
    session = TwilioSessionData()
    phone.sessions["session"] = session
    resolver = AsyncMock(return_value="B")
    phone.set_context_resolver(resolver, after_idle_seconds=300)
    return SimpleNamespace(
        phone=phone, session=session, resolver=resolver, clock=clock,
        values=values, refresh=pipeline.accepted_hooks[-1],
    )


@pytest.mark.asyncio
async def test_refresh_waits_for_idle_and_only_start_updates_activity(refresh_server):
    h = refresh_server
    request = STSRequest(session_id="session", context_id="A")
    await h.refresh(request)
    h.clock.now += 299
    await h.refresh(request)
    h.resolver.assert_not_awaited()

    for event in ("accepted", "chunk", "final"):
        await h.phone.handle_response(STSResponse(type=event, session_id="session"))
        assert h.session.last_response_at == 1000.0

    await h.phone.handle_response(STSResponse(type="start", session_id="session"))
    assert h.session.last_response_at == h.clock.now
    h.clock.now += 299
    await h.refresh(request)
    h.resolver.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("context_id", ["B", None])
async def test_idle_refresh_updates_request_and_session(refresh_server, context_id):
    h = refresh_server
    h.clock.now += 300
    h.resolver.return_value = context_id
    request = STSRequest(session_id="session", user_id="channel-user", context_id="A")

    await h.refresh(request)

    h.resolver.assert_awaited_once_with("app-user")
    assert request.user_id == "app-user"
    assert request.context_id == context_id
    assert h.values["context_id"] == context_id


@pytest.mark.asyncio
async def test_refresh_does_not_apply_after_session_replacement(refresh_server):
    h = refresh_server
    h.clock.now += 300

    async def replace_session(user_id):
        h.phone.sessions["session"] = TwilioSessionData()
        return "B"

    h.resolver.side_effect = replace_session
    request = STSRequest(session_id="session", user_id="channel-user", context_id="A")

    await h.refresh(request)

    h.resolver.assert_awaited_once_with("app-user")
    assert request.user_id == "channel-user"
    assert request.context_id == "A"
    assert h.values["context_id"] == "A"
