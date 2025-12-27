import os
from uuid import uuid4
import pytest
import asyncpg
from aiavatar.sts.performance_recorder import PerformanceRecord
from aiavatar.sts.performance_recorder.postgres import PostgreSQLPerformanceRecorder

# Environment variables for PostgreSQL connection
AIAVATAR_DB_HOST = os.getenv("AIAVATAR_DB_HOST", "localhost")
AIAVATAR_DB_PORT = os.getenv("AIAVATAR_DB_PORT", 5432)
AIAVATAR_DB_NAME = os.getenv("AIAVATAR_DB_NAME", "aiavatar")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER", "postgres")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")

# Skip tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not AIAVATAR_DB_PASSWORD,
    reason="PostgreSQL credentials not configured"
)


@pytest.fixture
def recorder():
    rec = PostgreSQLPerformanceRecorder(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        dbname=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
        db_pool_size=2
    )
    yield rec
    rec.close()


@pytest.fixture
def unique_transaction_id():
    return f"test_txn_{uuid4()}"


async def get_test_connection():
    """Create a separate connection for test verification"""
    return await asyncpg.connect(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        database=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD
    )


@pytest.mark.asyncio
async def test_record_single(recorder, unique_transaction_id):
    """Test recording a single performance record"""
    record = PerformanceRecord(
        transaction_id=unique_transaction_id,
        user_id="test_user",
        context_id="test_context",
        stt_name="test_stt",
        llm_name="test_llm",
        tts_name="test_tts",
        request_text="Hello, world!",
        response_text="Hi there!",
        voice_length=1.5,
        stt_time=0.1,
        llm_time=0.5,
        tts_time=0.3,
        total_time=0.9
    )

    recorder.record(record)
    recorder.record_queue.join()

    # Use separate connection for verification
    conn = await get_test_connection()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM performance_records WHERE transaction_id = $1",
            unique_transaction_id
        )
        assert row is not None
        assert row["user_id"] == "test_user"
        assert row["context_id"] == "test_context"
        assert row["request_text"] == "Hello, world!"
        assert row["response_text"] == "Hi there!"
        assert row["stt_name"] == "test_stt"
        assert row["llm_name"] == "test_llm"
        assert row["tts_name"] == "test_tts"
        assert abs(row["voice_length"] - 1.5) < 0.01
        assert abs(row["stt_time"] - 0.1) < 0.01
        assert abs(row["llm_time"] - 0.5) < 0.01
        assert abs(row["tts_time"] - 0.3) < 0.01
        assert abs(row["total_time"] - 0.9) < 0.01
        assert row["created_at"] is not None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_multiple(recorder):
    """Test recording multiple performance records"""
    transaction_ids = [f"test_txn_{uuid4()}" for _ in range(5)]

    for i, txn_id in enumerate(transaction_ids):
        record = PerformanceRecord(
            transaction_id=txn_id,
            user_id=f"user_{i}",
            context_id=f"context_{i}",
            request_text=f"Request {i}",
            response_text=f"Response {i}",
            total_time=i * 0.1
        )
        recorder.record(record)

    recorder.record_queue.join()

    conn = await get_test_connection()
    try:
        for i, txn_id in enumerate(transaction_ids):
            row = await conn.fetchrow(
                "SELECT * FROM performance_records WHERE transaction_id = $1",
                txn_id
            )
            assert row is not None
            assert row["user_id"] == f"user_{i}"
            assert row["context_id"] == f"context_{i}"
            assert row["request_text"] == f"Request {i}"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_with_none_values(recorder, unique_transaction_id):
    """Test recording with None values for optional fields"""
    record = PerformanceRecord(
        transaction_id=unique_transaction_id,
        user_id=None,
        context_id=None,
        request_text=None,
        response_text=None
    )

    recorder.record(record)
    recorder.record_queue.join()

    conn = await get_test_connection()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM performance_records WHERE transaction_id = $1",
            unique_transaction_id
        )
        assert row is not None
        assert row["user_id"] is None
        assert row["context_id"] is None
        assert row["request_text"] is None
        assert row["response_text"] is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_close_flushes_queue(unique_transaction_id):
    """Test that close() waits for all queued records to be processed"""
    recorder = PostgreSQLPerformanceRecorder(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        dbname=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD
    )

    record = PerformanceRecord(
        transaction_id=unique_transaction_id,
        request_text="Final record"
    )

    recorder.record(record)
    recorder.close()

    # After close, the record should be in the database
    conn = await get_test_connection()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM performance_records WHERE transaction_id = $1",
            unique_transaction_id
        )
        assert row is not None
        assert row["request_text"] == "Final record"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_init_db_creates_table(recorder):
    """Test that init_db creates the performance_records table"""
    # Wait for worker to initialize
    recorder.record_queue.join()

    conn = await get_test_connection()
    try:
        row = await conn.fetchrow(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'performance_records'
            """
        )
        assert row is not None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_init_db_creates_indexes(recorder):
    """Test that init_db creates the required indexes"""
    # Wait for worker to initialize
    recorder.record_queue.join()

    conn = await get_test_connection()
    try:
        rows = await conn.fetch(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'performance_records'
            """
        )
        index_names = {row["indexname"] for row in rows}
        assert "idx_created_at" in index_names
        assert "idx_transaction_id" in index_names
        assert "idx_user_id" in index_names
        assert "idx_context_id" in index_names
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_connection_string(unique_transaction_id):
    """Test using connection string instead of individual parameters"""
    connection_str = f"postgresql://{AIAVATAR_DB_USER}:{AIAVATAR_DB_PASSWORD}@{AIAVATAR_DB_HOST}:{AIAVATAR_DB_PORT}/{AIAVATAR_DB_NAME}"

    recorder = PostgreSQLPerformanceRecorder(
        connection_str=connection_str
    )

    try:
        record = PerformanceRecord(
            transaction_id=unique_transaction_id,
            request_text="Connection string test"
        )
        recorder.record(record)
        recorder.record_queue.join()

        conn = await get_test_connection()
        try:
            row = await conn.fetchrow(
                "SELECT * FROM performance_records WHERE transaction_id = $1",
                unique_transaction_id
            )
            assert row is not None
            assert row["request_text"] == "Connection string test"
        finally:
            await conn.close()
    finally:
        recorder.close()
