import asyncio
import os
import pytest
import pytest_asyncio
from uuid import uuid4
from aiavatar.database.postgres import PostgreSQLPoolProvider

# Environment variables for PostgreSQL connection
AIAVATAR_DB_HOST = os.getenv("AIAVATAR_DB_HOST", "localhost")
AIAVATAR_DB_PORT = int(os.getenv("AIAVATAR_DB_PORT", 5432))
AIAVATAR_DB_NAME = os.getenv("AIAVATAR_DB_NAME", "aiavatar")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER", "postgres")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")
AIAVATAR_DB_CONNECTION_STR = os.getenv("AIAVATAR_DB_CONNECTION_STR")

# Skip tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not AIAVATAR_DB_PASSWORD and not AIAVATAR_DB_CONNECTION_STR,
    reason="PostgreSQL credentials not configured"
)


def get_connection_str():
    if AIAVATAR_DB_CONNECTION_STR:
        return AIAVATAR_DB_CONNECTION_STR
    return f"postgresql://{AIAVATAR_DB_USER}:{AIAVATAR_DB_PASSWORD}@{AIAVATAR_DB_HOST}:{AIAVATAR_DB_PORT}/{AIAVATAR_DB_NAME}"


@pytest_asyncio.fixture
async def pool_provider():
    provider = PostgreSQLPoolProvider(
        connection_str=get_connection_str(),
        min_size=2,
        max_size=5
    )
    yield provider
    await provider.close()


@pytest.mark.asyncio
async def test_db_type(pool_provider):
    assert pool_provider.db_type == "postgresql"


@pytest.mark.asyncio
async def test_get_stats_before_initialization():
    provider = PostgreSQLPoolProvider(
        connection_str="postgresql://dummy:dummy@localhost/dummy"
    )
    stats = provider.get_stats()
    assert stats == {"initialized": False}


@pytest.mark.asyncio
async def test_get_pool_creates_pool(pool_provider):
    pool = await pool_provider.get_pool()
    assert pool is not None
    assert pool_provider._pool is not None


@pytest.mark.asyncio
async def test_get_pool_returns_same_pool(pool_provider):
    pool1 = await pool_provider.get_pool()
    pool2 = await pool_provider.get_pool()
    assert pool1 is pool2


@pytest.mark.asyncio
async def test_get_pool_with_params():
    provider = PostgreSQLPoolProvider(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        dbname=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
        min_size=2,
        max_size=5
    )
    pool = await provider.get_pool()
    assert pool is not None
    await provider.close()


@pytest.mark.asyncio
async def test_concurrent_get_pool(pool_provider):
    results = await asyncio.gather(
        pool_provider.get_pool(),
        pool_provider.get_pool(),
        pool_provider.get_pool()
    )
    assert results[0] is results[1] is results[2]


@pytest.mark.asyncio
async def test_pool_can_execute_query(pool_provider):
    pool = await pool_provider.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1


@pytest.mark.asyncio
async def test_get_stats_after_initialization(pool_provider):
    await pool_provider.get_pool()
    stats = pool_provider.get_stats()

    assert stats["initialized"] is True
    assert "size" in stats
    assert "idle" in stats
    assert "in_use" in stats
    assert stats["min_size"] == 2
    assert stats["max_size"] == 5
    assert stats["in_use"] == stats["size"] - stats["idle"]


@pytest.mark.asyncio
async def test_stats_in_use_during_query(pool_provider):
    pool = await pool_provider.get_pool()

    async with pool.acquire() as conn:
        stats = pool_provider.get_stats()
        assert stats["in_use"] >= 1


@pytest.mark.asyncio
async def test_close_pool():
    provider = PostgreSQLPoolProvider(connection_str=get_connection_str())
    await provider.get_pool()
    assert provider._pool is not None

    await provider.close()
    assert provider._pool is None


@pytest.mark.asyncio
async def test_close_without_pool():
    provider = PostgreSQLPoolProvider(
        connection_str="postgresql://dummy:dummy@localhost/dummy"
    )
    await provider.close()
    assert provider._pool is None


@pytest.mark.asyncio
async def test_get_pool_after_close():
    provider = PostgreSQLPoolProvider(connection_str=get_connection_str())

    pool1 = await provider.get_pool()
    await provider.close()

    pool2 = await provider.get_pool()
    assert pool2 is not None
    assert pool1 is not pool2

    await provider.close()


@pytest.mark.asyncio
async def test_shared_pool_with_session_state_manager(pool_provider):
    from aiavatar.sts.session_state_manager.postgres import PostgreSQLSessionStateManager

    manager = PostgreSQLSessionStateManager(
        get_pool=pool_provider.get_pool
    )

    session_id = f"test_{uuid4()}"
    state = await manager.get_session_state(session_id)
    assert state is not None
    assert state.session_id == session_id

    stats = pool_provider.get_stats()
    assert stats["initialized"] is True


@pytest.mark.asyncio
async def test_shared_pool_with_context_manager(pool_provider):
    from aiavatar.sts.llm.context_manager.postgres import PostgreSQLContextManager

    manager = PostgreSQLContextManager(
        get_pool=pool_provider.get_pool
    )

    context_id = f"test_{uuid4()}"
    messages = await manager.get_histories(context_id)
    assert messages == []

    stats = pool_provider.get_stats()
    assert stats["initialized"] is True


@pytest.mark.asyncio
async def test_multiple_components_share_pool(pool_provider):
    from aiavatar.sts.session_state_manager.postgres import PostgreSQLSessionStateManager
    from aiavatar.sts.llm.context_manager.postgres import PostgreSQLContextManager

    session_manager = PostgreSQLSessionStateManager(
        get_pool=pool_provider.get_pool
    )
    context_manager = PostgreSQLContextManager(
        get_pool=pool_provider.get_pool
    )

    session_id = f"test_{uuid4()}"
    context_id = f"test_{uuid4()}"

    state = await session_manager.get_session_state(session_id)
    messages = await context_manager.get_histories(context_id)

    assert state is not None
    assert messages == []

    stats = pool_provider.get_stats()
    assert stats["initialized"] is True
    assert stats["size"] <= stats["max_size"]
