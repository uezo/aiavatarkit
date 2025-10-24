from datetime import datetime, timezone
import json
import os
from uuid import uuid4
import pytest
from aiavatar.sts.llm.context_manager.postgres import PostgreSQLContextManager

AIAVATAR_DB_PORT = os.getenv("AIAVATAR_DB_PORT")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")


@pytest.fixture
def context_manager() -> PostgreSQLContextManager:
    return PostgreSQLContextManager(
        port=AIAVATAR_DB_PORT,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
        context_timeout=3600
    )


@pytest.mark.asyncio
async def test_get_histories_empty(context_manager):
    context_id = "non_existent_context"
    histories = await context_manager.get_histories(context_id)
    assert histories == []


@pytest.mark.asyncio
async def test_add_and_get_histories(context_manager):
    context_id = f"test_context_{uuid4()}"
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
    test_id = str(uuid4())
    context_id = f"test_specific_context_{test_id}"
    data_list = [
        {"message": "Hello, world!", "role": "user"},
        {"message": "Hi! How can I help you today?", "role": "assistant"}
    ]
    await context_manager.add_histories(context_id, data_list)

    shared_context_id = f"test_shared_context_{test_id}"
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
    context_id = f"test_context_limit_{uuid4()}"
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
    context_id = f"test_context_timeout_{uuid4()}"
    
    old_data = {"message": "Old data"}
    new_data = {"message": "New data"}

    await context_manager.add_histories(context_id, [new_data])

    old_timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    conn = context_manager.connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_histories (created_at, context_id, serialized_data)
                    VALUES (%s, %s, %s)
                    """,
                    (old_timestamp, context_id, json.dumps(old_data))
                )
    finally:
        conn.close()

    histories = await context_manager.get_histories(context_id)
    assert len(histories) == 1
    assert histories[0]["message"] == "New data"
