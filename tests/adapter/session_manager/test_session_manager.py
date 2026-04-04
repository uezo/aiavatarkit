from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from aiavatar.adapter.session_manager.base import (
    ChannelSession,
    SQLiteChannelSessionManager,
)


@pytest.fixture
def db_path(tmp_path: Path):
    return tmp_path / "channel_sessions.db"


@pytest.fixture
def session_manager(db_path) -> SQLiteChannelSessionManager:
    return SQLiteChannelSessionManager(db_path=str(db_path), timeout=1)


@pytest.mark.asyncio
async def test_get_session_creates_new(session_manager):
    user_id = f"user_{uuid4()}"
    session = await session_manager.get_session("line", user_id)
    assert session.channel_id == "line"
    assert session.channel_user_id == user_id
    assert session.user_id == user_id
    assert session.session_id.startswith("ch_sess_")
    assert session.data == {}


@pytest.mark.asyncio
async def test_upsert_and_get(session_manager):
    user_id = f"user_{uuid4()}"
    session = ChannelSession(
        channel_id="twilio",
        channel_user_id=user_id,
        session_id=f"ch_sess_{uuid4()}",
        user_id="app_user",
        context_id="ctx_123",
        data={"foo": "bar"},
    )
    await session_manager.upsert_session(session)

    loaded = await session_manager.get_session("twilio", user_id)
    assert loaded.session_id == session.session_id
    assert loaded.user_id == "app_user"
    assert loaded.context_id == "ctx_123"
    assert loaded.data == {"foo": "bar"}


@pytest.mark.asyncio
async def test_timeout_creates_new(session_manager):
    user_id = f"user_{uuid4()}"
    session = await session_manager.get_session("websocket", user_id)
    session.updated_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    await session_manager.upsert_session(session)

    refreshed = await session_manager.get_session("websocket", user_id)
    assert refreshed.session_id != session.session_id
    assert refreshed.updated_at > session.updated_at
    assert refreshed.user_id == user_id


@pytest.mark.asyncio
async def test_delete_session(session_manager):
    user_id = f"user_{uuid4()}"
    await session_manager.get_session("line", user_id)
    await session_manager.delete_session("line", user_id)

    new_session = await session_manager.get_session("line", user_id)
    assert new_session.channel_user_id == user_id
    assert new_session.session_id.startswith("ch_sess_")


@pytest.mark.asyncio
async def test_update_context_id(session_manager):
    user_id = f"user_{uuid4()}"
    session = await session_manager.get_session("line", user_id)
    assert session.context_id is None

    await session_manager.update_context_id("line", user_id, "ctx_new")

    loaded = await session_manager.get_session("line", user_id)
    assert loaded.context_id == "ctx_new"
    assert loaded.session_id == session.session_id
    assert loaded.updated_at >= session.updated_at


@pytest.mark.asyncio
async def test_different_channels_same_user(session_manager):
    """Same channel_user_id on different channels should be independent sessions."""
    user_id = f"user_{uuid4()}"
    line_session = await session_manager.get_session("line", user_id)
    twilio_session = await session_manager.get_session("twilio", user_id)

    assert line_session.session_id != twilio_session.session_id
    assert line_session.channel_id == "line"
    assert twilio_session.channel_id == "twilio"
