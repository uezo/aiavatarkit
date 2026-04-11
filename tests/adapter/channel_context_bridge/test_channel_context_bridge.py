from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aiavatar.adapter.channel_context_bridge.base import (
    ChannelUser,
    UserContext,
    SQLiteChannelContextBridge,
)


@pytest.fixture
def db_path(tmp_path: Path):
    return tmp_path / "channel_context_bridge.db"


@pytest.fixture
def bridge(db_path) -> SQLiteChannelContextBridge:
    return SQLiteChannelContextBridge(db_path=str(db_path), timeout=1)


# --- Channel User tests ---

@pytest.mark.asyncio
async def test_get_channel_user_not_found(bridge):
    result = await bridge.get_channel_user("linebot", "user_1")
    assert result is None


@pytest.mark.asyncio
async def test_get_channel_user_auto_create(bridge):
    user = await bridge.get_channel_user("linebot", "user_1", auto_create=True)
    assert user.channel_id == "linebot"
    assert user.channel_user_id == "user_1"
    assert user.user_id == "user_1"
    assert user.data == {}


@pytest.mark.asyncio
async def test_upsert_and_get_channel_user(bridge):
    cu = ChannelUser(
        channel_id="twilio",
        channel_user_id="+1234567890",
        user_id="app_user",
        data={"role": "admin"},
    )
    await bridge.upsert_channel_user(cu)

    loaded = await bridge.get_channel_user("twilio", "+1234567890")
    assert loaded.user_id == "app_user"
    assert loaded.data == {"role": "admin"}


@pytest.mark.asyncio
async def test_upsert_channel_user_updates(bridge):
    cu = ChannelUser(
        channel_id="linebot",
        channel_user_id="user_2",
        user_id="user_2",
    )
    await bridge.upsert_channel_user(cu)

    cu.user_id = "linked_user"
    await bridge.upsert_channel_user(cu)

    loaded = await bridge.get_channel_user("linebot", "user_2")
    assert loaded.user_id == "linked_user"


@pytest.mark.asyncio
async def test_delete_channel_user(bridge):
    await bridge.get_channel_user("linebot", "user_3", auto_create=True)
    await bridge.delete_channel_user("linebot", "user_3")

    result = await bridge.get_channel_user("linebot", "user_3")
    assert result is None


@pytest.mark.asyncio
async def test_find_channel_users(bridge):
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="websocket", channel_user_id="ws_1", user_id="app_user"
    ))
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="twilio", channel_user_id="+1234567890", user_id="app_user"
    ))
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="linebot", channel_user_id="line_1", user_id="other_user"
    ))

    users = await bridge.find_channel_users("app_user")
    assert len(users) == 2
    channel_ids = {u.channel_id for u in users}
    assert channel_ids == {"websocket", "twilio"}


@pytest.mark.asyncio
async def test_find_channel_users_empty(bridge):
    users = await bridge.find_channel_users("nonexistent")
    assert users == []


@pytest.mark.asyncio
async def test_different_channels_same_channel_user_id(bridge):
    """Same channel_user_id on different channels should be independent."""
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="linebot", channel_user_id="user_x", user_id="line_user"
    ))
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="twilio", channel_user_id="user_x", user_id="twilio_user"
    ))

    line = await bridge.get_channel_user("linebot", "user_x")
    twilio = await bridge.get_channel_user("twilio", "user_x")
    assert line.user_id == "line_user"
    assert twilio.user_id == "twilio_user"


# --- User Context tests ---

@pytest.mark.asyncio
async def test_get_context_not_found(bridge):
    result = await bridge.get_context("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_upsert_and_get_context(bridge):
    await bridge.upsert_context(UserContext(
        user_id="user_1",
        context_id="ctx_123",
    ))

    ctx = await bridge.get_context("user_1")
    assert ctx.user_id == "user_1"
    assert ctx.context_id == "ctx_123"


@pytest.mark.asyncio
async def test_upsert_context_updates(bridge):
    await bridge.upsert_context(UserContext(
        user_id="user_2", context_id="ctx_old"
    ))
    await bridge.upsert_context(UserContext(
        user_id="user_2", context_id="ctx_new"
    ))

    ctx = await bridge.get_context("user_2")
    assert ctx.context_id == "ctx_new"


@pytest.mark.asyncio
async def test_get_context_timeout(bridge):
    """Expired context should return None."""
    await bridge.upsert_context(UserContext(
        user_id="user_3",
        context_id="ctx_expired",
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=5),
    ))

    ctx = await bridge.get_context("user_3")
    assert ctx is None


@pytest.mark.asyncio
async def test_context_independent_of_channel_user(bridge):
    """Context is per user_id, not per channel."""
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="linebot", channel_user_id="line_1", user_id="shared_user"
    ))
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="twilio", channel_user_id="+1111111111", user_id="shared_user"
    ))
    await bridge.upsert_context(UserContext(
        user_id="shared_user", context_id="ctx_shared"
    ))

    ctx = await bridge.get_context("shared_user")
    assert ctx.context_id == "ctx_shared"


# --- link_channel_user tests ---

@pytest.mark.asyncio
async def test_link_channel_user_new(bridge):
    """link_channel_user auto-creates channel user and returns context."""
    await bridge.upsert_context(UserContext(
        user_id="app_user", context_id="ctx_existing"
    ))

    ctx = await bridge.link_channel_user("twilio", "+9999999999", "app_user")
    assert ctx.context_id == "ctx_existing"

    cu = await bridge.get_channel_user("twilio", "+9999999999")
    assert cu.user_id == "app_user"


@pytest.mark.asyncio
async def test_link_channel_user_updates_existing(bridge):
    """link_channel_user updates user_id on existing channel user."""
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="twilio", channel_user_id="+8888888888", user_id="+8888888888"
    ))
    await bridge.upsert_context(UserContext(
        user_id="linked_user", context_id="ctx_linked"
    ))

    ctx = await bridge.link_channel_user("twilio", "+8888888888", "linked_user")
    assert ctx.context_id == "ctx_linked"

    cu = await bridge.get_channel_user("twilio", "+8888888888")
    assert cu.user_id == "linked_user"


@pytest.mark.asyncio
async def test_link_channel_user_no_context(bridge):
    """link_channel_user returns None when linked user has no context."""
    ctx = await bridge.link_channel_user("linebot", "line_user_1", "new_app_user")
    assert ctx is None

    cu = await bridge.get_channel_user("linebot", "line_user_1")
    assert cu.user_id == "new_app_user"
