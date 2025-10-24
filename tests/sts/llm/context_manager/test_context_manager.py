from datetime import datetime, timezone
import json
import os
import sqlite3
import pytest
from aiavatar.sts.llm.context_manager import SQLiteContextManager


@pytest.fixture
def db_path(tmp_path):
    return os.path.join(tmp_path, "test_context.db")


@pytest.fixture
def context_manager(db_path) -> SQLiteContextManager:
    return SQLiteContextManager(db_path=db_path, context_timeout=3600)


@pytest.mark.asyncio
async def test_get_histories_empty(context_manager):
    context_id = "non_existent_context"
    histories = await context_manager.get_histories(context_id)
    assert histories == []


@pytest.mark.asyncio
async def test_add_and_get_histories(context_manager):
    context_id = "test_context"
    data_list = [
        {"message": "Hello, world!", "role": "user"},
        {"message": "Hi! How can I help you today?", "role": "assistant"}
    ]

    await context_manager.add_histories(context_id, data_list)

    histories = await context_manager.get_histories(context_id)
    assert len(histories) == 2

    assert histories[0]["message"] == "Hello, world!"
    assert histories[0]["role"] == "user"
    assert histories[1]["message"] == "Hi! How can I help you today?"
    assert histories[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_add_and_get_histories_with_shared(context_manager):
    context_id = "test_specific_context"
    data_list = [
        {"message": "Hello, world!", "role": "user"},
        {"message": "Hi! How can I help you today?", "role": "assistant"}
    ]
    await context_manager.add_histories(context_id, data_list)

    shared_context_id = "test_shared_context"
    shared_data_list = [
        {"message": "Shared Request", "role": "user"},
        {"message": "Shared Answer", "role": "assistant"}
    ]
    await context_manager.add_histories(shared_context_id, shared_data_list)

    histories = await context_manager.get_histories(context_id=[context_id, shared_context_id])
    assert len(histories) == 4

    assert histories[0]["message"] == "Hello, world!"
    assert histories[0]["role"] == "user"
    assert histories[1]["message"] == "Hi! How can I help you today?"
    assert histories[1]["role"] == "assistant"
    assert histories[2]["message"] == "Shared Request"
    assert histories[2]["role"] == "user"
    assert histories[3]["message"] == "Shared Answer"
    assert histories[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_histories_limit(context_manager):
    context_id = "test_limit"
    data_list = [
        {"index": 1}, {"index": 2}, {"index": 3}, {"index": 4}, {"index": 5}
    ]
    await context_manager.add_histories(context_id, data_list)

    histories_all = await context_manager.get_histories(context_id, limit=100)
    assert len(histories_all) == 5

    histories_limited = await context_manager.get_histories(context_id, limit=3)
    assert len(histories_limited) == 3
    assert histories_limited[0]["index"] == 3
    assert histories_limited[-1]["index"] == 5


@pytest.mark.asyncio
async def test_get_histories_timeout(context_manager):
    context_id = "test_timeout"
    
    old_data = {"message": "Old data"}
    new_data = {"message": "New data"}

    await context_manager.add_histories(context_id, [new_data])

    old_timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    conn = sqlite3.connect(context_manager.db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO chat_histories (created_at, context_id, serialized_data)
                VALUES (?, ?, ?)
                """,
                (old_timestamp, context_id, json.dumps(old_data))
            )
    finally:
        conn.close()

    histories = await context_manager.get_histories(context_id)
    assert len(histories) == 1
    assert histories[0]["message"] == "New data"
