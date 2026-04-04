from datetime import datetime, timedelta, timezone
from uuid import uuid4
import os

import pytest

from aiavatar.adapter.session_manager.postgres import PostgreSQLChannelSessionManager, ChannelSession

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
def session_manager() -> PostgreSQLChannelSessionManager:
    return PostgreSQLChannelSessionManager(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        dbname=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
        timeout=1,
    )


@pytest.fixture
def unique_user():
    return f"user_{uuid4()}"


@pytest.mark.asyncio
async def test_get_session_creates_new(session_manager, unique_user):
    session = await session_manager.get_session("line", unique_user)
    assert session.channel_id == "line"
    assert session.channel_user_id == unique_user
    assert session.user_id == unique_user
    assert session.session_id.startswith("ch_sess_")


@pytest.mark.asyncio
async def test_upsert_and_get(session_manager, unique_user):
    session = ChannelSession(
        channel_id="twilio",
        channel_user_id=unique_user,
        session_id=f"ch_sess_{uuid4()}",
        user_id="app_user",
        context_id="ctx_pg",
        data={"hello": "world"},
    )
    await session_manager.upsert_session(session)

    loaded = await session_manager.get_session("twilio", unique_user)
    assert loaded.session_id == session.session_id
    assert loaded.user_id == "app_user"
    assert loaded.context_id == "ctx_pg"
    assert loaded.data == {"hello": "world"}


@pytest.mark.asyncio
async def test_timeout_creates_new(session_manager, unique_user):
    session = await session_manager.get_session("websocket", unique_user)
    session.updated_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    await session_manager.upsert_session(session)

    refreshed = await session_manager.get_session("websocket", unique_user)
    assert refreshed.session_id != session.session_id
    assert refreshed.updated_at > session.updated_at


@pytest.mark.asyncio
async def test_update_context_id(session_manager, unique_user):
    session = await session_manager.get_session("line", unique_user)
    assert session.context_id is None

    await session_manager.update_context_id("line", unique_user, "ctx_new")

    loaded = await session_manager.get_session("line", unique_user)
    assert loaded.context_id == "ctx_new"
    assert loaded.session_id == session.session_id
    assert loaded.updated_at >= session.updated_at


@pytest.mark.asyncio
async def test_delete_session(session_manager, unique_user):
    await session_manager.get_session("line", unique_user)
    await session_manager.delete_session("line", unique_user)

    new_session = await session_manager.get_session("line", unique_user)
    assert new_session.channel_user_id == unique_user
    assert new_session.session_id.startswith("ch_sess_")
