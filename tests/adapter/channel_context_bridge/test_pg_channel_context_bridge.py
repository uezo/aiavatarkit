from datetime import datetime, timedelta, timezone
from uuid import uuid4
import os

import pytest

from aiavatar.adapter.channel_context_bridge.postgres import PostgreSQLChannelContextBridge
from aiavatar.adapter.channel_context_bridge.base import ChannelUser, UserContext

AIAVATAR_DB_HOST = os.getenv("AIAVATAR_DB_HOST", "localhost")
AIAVATAR_DB_PORT = int(os.getenv("AIAVATAR_DB_PORT", 5432))
AIAVATAR_DB_NAME = os.getenv("AIAVATAR_DB_NAME", "aiavatar")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER", "postgres")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")

pytestmark = pytest.mark.skipif(
    not AIAVATAR_DB_PASSWORD,
    reason="PostgreSQL credentials not configured",
)


@pytest.fixture
def bridge() -> PostgreSQLChannelContextBridge:
    return PostgreSQLChannelContextBridge(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        dbname=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
        timeout=1,
    )


@pytest.fixture
def unique_id():
    return f"test_{uuid4()}"


# --- Channel User tests ---

@pytest.mark.asyncio
async def test_get_channel_user_not_found(bridge, unique_id):
    result = await bridge.get_channel_user("linebot", unique_id)
    assert result is None


@pytest.mark.asyncio
async def test_get_channel_user_auto_create(bridge, unique_id):
    user = await bridge.get_channel_user("linebot", unique_id, auto_create=True)
    assert user.channel_id == "linebot"
    assert user.channel_user_id == unique_id
    assert user.user_id == unique_id


@pytest.mark.asyncio
async def test_upsert_and_get_channel_user(bridge, unique_id):
    cu = ChannelUser(
        channel_id="twilio",
        channel_user_id=unique_id,
        user_id="app_user",
        data={"role": "admin"},
    )
    await bridge.upsert_channel_user(cu)

    loaded = await bridge.get_channel_user("twilio", unique_id)
    assert loaded.user_id == "app_user"
    assert loaded.data == {"role": "admin"}


@pytest.mark.asyncio
async def test_delete_channel_user(bridge, unique_id):
    await bridge.get_channel_user("linebot", unique_id, auto_create=True)
    await bridge.delete_channel_user("linebot", unique_id)

    result = await bridge.get_channel_user("linebot", unique_id)
    assert result is None


@pytest.mark.asyncio
async def test_find_channel_users(bridge):
    app_user_id = f"app_{uuid4()}"

    await bridge.upsert_channel_user(ChannelUser(
        channel_id="websocket", channel_user_id=f"ws_{uuid4()}", user_id=app_user_id
    ))
    await bridge.upsert_channel_user(ChannelUser(
        channel_id="twilio", channel_user_id=f"+1{uuid4().hex[:10]}", user_id=app_user_id
    ))

    users = await bridge.find_channel_users(app_user_id)
    assert len(users) == 2
    channel_ids = {u.channel_id for u in users}
    assert channel_ids == {"websocket", "twilio"}


# --- User Context tests ---

@pytest.mark.asyncio
async def test_get_context_not_found(bridge, unique_id):
    result = await bridge.get_context(unique_id)
    assert result is None


@pytest.mark.asyncio
async def test_upsert_and_get_context(bridge, unique_id):
    await bridge.upsert_context(UserContext(
        user_id=unique_id, context_id="ctx_123"
    ))

    ctx = await bridge.get_context(unique_id)
    assert ctx.user_id == unique_id
    assert ctx.context_id == "ctx_123"


@pytest.mark.asyncio
async def test_get_context_timeout(bridge, unique_id):
    await bridge.upsert_context(UserContext(
        user_id=unique_id,
        context_id="ctx_expired",
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=5),
    ))

    ctx = await bridge.get_context(unique_id)
    assert ctx is None


@pytest.mark.asyncio
async def test_upsert_context_updates(bridge, unique_id):
    await bridge.upsert_context(UserContext(
        user_id=unique_id, context_id="ctx_old"
    ))
    await bridge.upsert_context(UserContext(
        user_id=unique_id, context_id="ctx_new"
    ))

    ctx = await bridge.get_context(unique_id)
    assert ctx.context_id == "ctx_new"


@pytest.mark.asyncio
async def test_delete_context(bridge, unique_id):
    await bridge.upsert_context(UserContext(
        user_id=unique_id, context_id="ctx_to_delete"
    ))
    ctx = await bridge.get_context(unique_id)
    assert ctx is not None

    await bridge.delete_context(unique_id)

    ctx = await bridge.get_context(unique_id)
    assert ctx is None


@pytest.mark.asyncio
async def test_delete_context_nonexistent(bridge, unique_id):
    """Deleting a non-existent context should not raise."""
    await bridge.delete_context(unique_id)


# --- link_channel_user tests ---

@pytest.mark.asyncio
async def test_link_channel_user(bridge, unique_id):
    app_user_id = f"app_{uuid4()}"
    await bridge.upsert_context(UserContext(
        user_id=app_user_id, context_id="ctx_linked"
    ))

    ctx = await bridge.link_channel_user("twilio", unique_id, app_user_id)
    assert ctx.context_id == "ctx_linked"

    cu = await bridge.get_channel_user("twilio", unique_id)
    assert cu.user_id == app_user_id


@pytest.mark.asyncio
async def test_link_channel_user_no_context(bridge, unique_id):
    app_user_id = f"app_{uuid4()}"
    ctx = await bridge.link_channel_user("linebot", unique_id, app_user_id)
    assert ctx is None

    cu = await bridge.get_channel_user("linebot", unique_id)
    assert cu.user_id == app_user_id
