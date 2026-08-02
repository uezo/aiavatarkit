from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiavatar.adapter.linebot.tools.channel_link import LinebotChannelLinkTool


class FakeChannelContextBridge:
    def __init__(self):
        self.linked_users = []

    async def link_channel_user(self, channel_id, channel_user_id, user_id):
        self.linked_users.append((channel_id, channel_user_id, user_id))


def test_callback_uses_configured_adapter_channel():
    bridge = FakeChannelContextBridge()
    tool = LinebotChannelLinkTool(
        channel_id="line-login-channel-id",
        client_secret="secret",
        base_url="https://example.com",
        channel="custom-linebot",
        channel_context_bridge=bridge,
    )

    async def verify_state(state, code):
        assert state == "state"
        assert code == "code"
        return "application-user", "line-user"

    tool.verify_state = verify_state
    app = FastAPI()
    app.include_router(tool.get_callback_router())

    response = TestClient(app).get("/login-callback?state=state&code=code")

    assert response.status_code == 200
    assert bridge.linked_users == [(
        "custom-linebot",
        "line-user",
        "application-user",
    )]
