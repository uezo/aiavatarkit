import os
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import pytest
import pytest_asyncio
from aiavatar.sts.llm.response_id_store.postgres import PostgreSQLResponseIdStore

AIAVATAR_DB_PORT = os.getenv("AIAVATAR_DB_PORT")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")


@pytest_asyncio.fixture
async def store():
    s = PostgreSQLResponseIdStore(
        port=AIAVATAR_DB_PORT,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
    )
    yield s
    if s._pool is not None:
        await s._pool.close()


@pytest.mark.asyncio
async def test_get_nonexistent(store):
    result = await store.get(f"non_existent_{uuid4()}")
    assert result is None


@pytest.mark.asyncio
async def test_set_and_get(store):
    ctx = f"ctx_{uuid4()}"
    await store.set(ctx, "resp_abc")
    result = await store.get(ctx)
    assert result == "resp_abc"


@pytest.mark.asyncio
async def test_set_overwrite(store):
    ctx = f"ctx_{uuid4()}"
    await store.set(ctx, "resp_old")
    await store.set(ctx, "resp_new")
    result = await store.get(ctx)
    assert result == "resp_new"


@pytest.mark.asyncio
async def test_delete(store):
    ctx = f"ctx_{uuid4()}"
    await store.set(ctx, "resp_abc")
    await store.delete(ctx)
    result = await store.get(ctx)
    assert result is None


@pytest.mark.asyncio
async def test_delete_nonexistent(store):
    await store.delete(f"non_existent_{uuid4()}")


@pytest.mark.asyncio
async def test_delete_older_than(store):
    ctx_new = f"ctx_new_{uuid4()}"
    ctx_old = f"ctx_old_{uuid4()}"

    await store.set(ctx_new, "resp_new")

    # Insert an old record directly
    old_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=7200)
    pool = await store.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO response_ids (context_id, response_id, updated_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (context_id) DO UPDATE
            SET response_id = EXCLUDED.response_id, updated_at = EXCLUDED.updated_at
            """,
            ctx_old, "resp_old", old_time,
        )

    await store.delete_older_than(3600)

    assert await store.get(ctx_old) is None
    assert await store.get(ctx_new) == "resp_new"
