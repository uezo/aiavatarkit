import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import uuid4

import pytest

from aiavatar.adapter.linebot.session_manager.base import (
    LineBotSession,
    SQLiteLineBotSessionManager,
)


@pytest.fixture
def db_path(tmp_path: Path):
    return tmp_path / "linebot_sessions.db"


@pytest.fixture
def session_manager(db_path) -> SQLiteLineBotSessionManager:
    return SQLiteLineBotSessionManager(db_path=str(db_path), timeout=1)


@pytest.mark.asyncio
async def test_get_session_creates_new(session_manager):
    user_id = f"user_{uuid4()}"
    session = await session_manager.get_session(user_id)
    assert session.linebot_user_id == user_id
    assert session.user_id == user_id
    assert session.id.startswith("linebot_sess_")
    assert session.data == {}


@pytest.mark.asyncio
async def test_upsert_and_get(session_manager):
    user_id = f"user_{uuid4()}"
    session = LineBotSession(
        id=f"linebot_sess_{uuid4()}",
        linebot_user_id=user_id,
        user_id="app_user",
        context_id="ctx_123",
        data={"foo": "bar"},
    )
    await session_manager.upsert_session(session)

    loaded = await session_manager.get_session(user_id)
    assert loaded.id == session.id
    assert loaded.user_id == "app_user"
    assert loaded.context_id == "ctx_123"
    assert loaded.data == {"foo": "bar"}


@pytest.mark.asyncio
async def test_timeout_creates_new(session_manager):
    user_id = f"user_{uuid4()}"
    session = await session_manager.get_session(user_id)
    # Simulate old timestamp
    session.updated_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    await session_manager.upsert_session(session)

    # Should create a fresh session because timeout=1s
    refreshed = await session_manager.get_session(user_id)
    assert refreshed.id != session.id
    assert refreshed.updated_at > session.updated_at
    assert refreshed.user_id == user_id


@pytest.mark.asyncio
async def test_delete_session(session_manager):
    user_id = f"user_{uuid4()}"
    await session_manager.get_session(user_id)  # create
    await session_manager.delete_session(user_id)

    # Should return a new session after delete
    new_session = await session_manager.get_session(user_id)
    assert new_session.linebot_user_id == user_id
    assert new_session.id.startswith("linebot_sess_")
