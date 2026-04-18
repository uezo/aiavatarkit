from datetime import datetime, timezone
import json
import os
from uuid import uuid4
import pytest
import pytest_asyncio
from aiavatar.sts.llm.context_manager.postgres import PostgreSQLContextManager

AIAVATAR_DB_PORT = os.getenv("AIAVATAR_DB_PORT")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")


@pytest_asyncio.fixture
async def context_manager():
    manager = PostgreSQLContextManager(
        port=AIAVATAR_DB_PORT,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
        context_timeout=3600
    )
    yield manager
    # Close pool after test
    if manager._pool is not None:
        await manager._pool.close()


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

    old_timestamp = datetime(2000, 1, 1)
    pool = await context_manager.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chat_histories (created_at, context_id, serialized_data)
            VALUES ($1, $2, $3)
            """,
            old_timestamp, context_id, json.dumps(old_data)
        )

    histories = await context_manager.get_histories(context_id)
    assert len(histories) == 1
    assert histories[0]["message"] == "New data"


@pytest.mark.asyncio
async def test_get_histories_with_timestamp(context_manager):
    context_id = f"test_timestamp_{uuid4()}"
    data_list = [
        {"message": "Hello", "role": "user"},
        {"message": "Hi there", "role": "assistant"}
    ]
    await context_manager.add_histories(context_id, data_list)

    # Without timestamp
    histories = await context_manager.get_histories(context_id)
    assert len(histories) == 2
    assert "created_at" not in histories[0]

    # With timestamp
    histories_ts = await context_manager.get_histories(context_id, include_timestamp=True)
    assert len(histories_ts) == 2
    assert "created_at" in histories_ts[0]
    assert "created_at" in histories_ts[1]
    assert histories_ts[0]["message"] == "Hello"
    assert histories_ts[1]["message"] == "Hi there"


@pytest.mark.asyncio
async def test_merge_context(context_manager):
    ctx_from = f"ctx_from_{uuid4()}"
    ctx_to = f"ctx_to_{uuid4()}"
    await context_manager.add_histories(ctx_from, [{"message": "From message"}])
    await context_manager.add_histories(ctx_to, [{"message": "To message"}])

    await context_manager.merge_context(ctx_from, ctx_to)

    histories = await context_manager.get_histories(ctx_to)
    assert len(histories) == 2

    histories_from = await context_manager.get_histories(ctx_from)
    assert len(histories_from) == 0


@pytest.mark.asyncio
async def test_merge_context_with_hook(context_manager):
    hook_called = {}

    @context_manager.on_merge_context
    async def my_hook(from_id, to_id, cm):
        hook_called["from"] = from_id
        hook_called["to"] = to_id
        hook_called["histories"] = await cm.get_histories(from_id)

    ctx_src = f"ctx_src_{uuid4()}"
    ctx_dst = f"ctx_dst_{uuid4()}"
    await context_manager.add_histories(ctx_src, [{"message": "Source message"}])
    await context_manager.add_histories(ctx_dst, [{"message": "Dest message"}])

    await context_manager.merge_context(ctx_src, ctx_dst)

    assert hook_called["from"] == ctx_src
    assert hook_called["to"] == ctx_dst
    assert len(hook_called["histories"]) == 1
    assert hook_called["histories"][0]["message"] == "Source message"

    histories = await context_manager.get_histories(ctx_dst)
    assert len(histories) == 2
