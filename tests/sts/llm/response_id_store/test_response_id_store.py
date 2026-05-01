import os
import sqlite3
from datetime import datetime, timezone, timedelta
import pytest
from aiavatar.sts.llm.response_id_store import SQLiteResponseIdStore


@pytest.fixture
def db_path(tmp_path):
    return os.path.join(tmp_path, "test_response_ids.db")


@pytest.fixture
def store(db_path) -> SQLiteResponseIdStore:
    return SQLiteResponseIdStore(db_path=db_path)


@pytest.mark.asyncio
async def test_get_nonexistent(store):
    result = await store.get("non_existent_context")
    assert result is None


@pytest.mark.asyncio
async def test_set_and_get(store):
    await store.set("ctx_1", "resp_abc")
    result = await store.get("ctx_1")
    assert result == "resp_abc"


@pytest.mark.asyncio
async def test_set_overwrite(store):
    await store.set("ctx_1", "resp_old")
    await store.set("ctx_1", "resp_new")
    result = await store.get("ctx_1")
    assert result == "resp_new"


@pytest.mark.asyncio
async def test_delete(store):
    await store.set("ctx_1", "resp_abc")
    await store.delete("ctx_1")
    result = await store.get("ctx_1")
    assert result is None


@pytest.mark.asyncio
async def test_delete_nonexistent(store):
    # Should not raise
    await store.delete("non_existent_context")


@pytest.mark.asyncio
async def test_delete_older_than(store, db_path):
    # Insert a fresh record via the store
    await store.set("ctx_new", "resp_new")

    # Insert an old record directly into the database
    old_time = datetime.now(timezone.utc) - timedelta(seconds=7200)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO response_ids (context_id, response_id, updated_at) VALUES (?, ?, ?)",
                ("ctx_old", "resp_old", old_time),
            )
    finally:
        conn.close()

    # Delete records older than 1 hour
    await store.delete_older_than(3600)

    # Old record should be gone
    assert await store.get("ctx_old") is None
    # New record should remain
    assert await store.get("ctx_new") == "resp_new"
