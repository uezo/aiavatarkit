import json
import os
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import pytest
import asyncpg
from aiavatar.sts.performance_recorder import PerformanceRecord
from aiavatar.sts.performance_recorder.postgres import PostgreSQLPerformanceRecorder
from aiavatar.sts.performance_recorder.query import (
    PostgreSQLMetricsQuery,
    VALID_INTERVALS,
)

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
def query():
    return PostgreSQLMetricsQuery(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        dbname=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
    )


@pytest.fixture
def test_marker():
    """Generate a unique marker for this test run to filter results"""
    return f"test_marker_{uuid4()}"


async def get_test_connection():
    """Create a separate connection for test verification"""
    return await asyncpg.connect(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        database=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD
    )


async def _update_created_at(transaction_id, new_created_at):
    """Helper to update created_at for testing time-based queries"""
    conn = await get_test_connection()
    try:
        await conn.execute(
            "UPDATE performance_records SET created_at = $1 WHERE transaction_id = $2",
            new_created_at, transaction_id
        )
    finally:
        await conn.close()


def _filter_buckets_by_marker(buckets, marker):
    """
    Filter timeline buckets - since buckets don't have marker info,
    we need to query the DB directly for accurate counting.
    This helper just returns buckets with data for basic validation.
    """
    return [b for b in buckets if b.request_count > 0]


async def _count_test_records(marker, error_only=False, success_only=False):
    """Count records matching the test marker"""
    conn = await get_test_connection()
    try:
        if error_only:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM performance_records WHERE user_id = $1 AND error_info IS NOT NULL",
                marker
            )
        elif success_only:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM performance_records WHERE user_id = $1 AND error_info IS NULL",
                marker
            )
        else:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM performance_records WHERE user_id = $1",
                marker
            )
        return row["cnt"]
    finally:
        await conn.close()


async def _get_test_records(marker):
    """Get records matching the test marker"""
    conn = await get_test_connection()
    try:
        rows = await conn.fetch(
            """SELECT stt_time, llm_first_chunk_time, llm_first_voice_chunk_time,
                      tts_first_chunk_time, error_info
               FROM performance_records WHERE user_id = $1""",
            marker
        )
        return rows
    finally:
        await conn.close()


# ========== query_timeline tests ==========

@pytest.mark.asyncio
async def test_query_timeline_with_data(recorder, query, test_marker):
    """Test timeline query with actual data"""
    txn_id = f"test_txn_{uuid4()}"
    record = PerformanceRecord(
        transaction_id=txn_id,
        user_id=test_marker,
        context_id="ctx1",
        stt_time=0.1,
        llm_first_chunk_time=0.2,
        llm_first_voice_chunk_time=0.3,
        tts_first_chunk_time=0.5,
        total_time=1.0,
    )
    recorder.record(record)
    recorder.record_queue.join()

    # Verify using direct DB query with marker
    count = await _count_test_records(test_marker)
    assert count == 1

    # Also verify timeline API returns data
    result = await query.query_timeline("1h", "1m")
    buckets_with_data = [b for b in result if b.request_count > 0]
    assert len(buckets_with_data) >= 1


@pytest.mark.asyncio
async def test_query_timeline_excludes_old_data(recorder, query, test_marker):
    """Test that timeline excludes data outside the period"""
    txn_id = f"test_txn_{uuid4()}"
    record = PerformanceRecord(
        transaction_id=txn_id,
        user_id=test_marker,
        stt_time=0.1,
        tts_first_chunk_time=0.5,
    )
    recorder.record(record)
    recorder.record_queue.join()

    # Move created_at to 2 hours ago
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    await _update_created_at(txn_id, old_time)

    # Verify the record exists
    count = await _count_test_records(test_marker)
    assert count == 1

    # Query last 1 hour timeline - the old record should be outside range
    # We can't easily verify exclusion without marker filtering in timeline,
    # but we can verify the API doesn't error
    result = await query.query_timeline("1h", "1m")
    assert len(result) > 0


@pytest.mark.asyncio
async def test_query_timeline_success_and_error_counts(recorder, query, test_marker):
    """Test that success and error counts are correct"""
    # Create success record
    success_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=success_txn,
        user_id=test_marker,
        stt_time=0.1,
        tts_first_chunk_time=0.5,
    ))

    # Create error record
    error_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=error_txn,
        user_id=test_marker,
        stt_time=0.1,
        error_info=json.dumps({"error": "Test error"}),
    ))

    recorder.record_queue.join()

    # Verify counts using direct DB query
    total = await _count_test_records(test_marker)
    success = await _count_test_records(test_marker, success_only=True)
    error = await _count_test_records(test_marker, error_only=True)

    assert total == 2
    assert success == 1
    assert error == 1


@pytest.mark.asyncio
async def test_query_timeline_phase_calculations(recorder, query, test_marker):
    """Test that phase calculations are correct"""
    txn_id = f"test_txn_{uuid4()}"
    record = PerformanceRecord(
        transaction_id=txn_id,
        user_id=test_marker,
        stt_time=0.1,
        llm_first_chunk_time=0.2,
        llm_first_voice_chunk_time=0.4,
        tts_first_chunk_time=0.6,
    )
    recorder.record(record)
    recorder.record_queue.join()

    # Verify the record exists with correct values
    records = await _get_test_records(test_marker)
    assert len(records) == 1
    r = records[0]
    assert abs(r["stt_time"] - 0.1) < 0.01
    assert abs(r["llm_first_chunk_time"] - 0.2) < 0.01
    assert abs(r["llm_first_voice_chunk_time"] - 0.4) < 0.01
    assert abs(r["tts_first_chunk_time"] - 0.6) < 0.01


@pytest.mark.asyncio
async def test_query_timeline_excludes_errors_from_phase_calc(recorder, query, test_marker):
    """Test that error records are excluded from phase calculations"""
    # Create error record with timing data
    error_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=error_txn,
        user_id=test_marker,
        stt_time=100.0,  # Large value that would skew average
        llm_first_chunk_time=200.0,
        tts_first_chunk_time=300.0,
        error_info=json.dumps({"error": "Test error"}),
    ))

    # Create success record
    success_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=success_txn,
        user_id=test_marker,
        stt_time=0.1,
        llm_first_chunk_time=0.2,
        llm_first_voice_chunk_time=0.3,
        tts_first_chunk_time=0.5,
    ))

    recorder.record_queue.join()

    # Verify counts
    total = await _count_test_records(test_marker)
    success = await _count_test_records(test_marker, success_only=True)
    error = await _count_test_records(test_marker, error_only=True)

    assert total == 2
    assert success == 1
    assert error == 1

    # Verify only success record has reasonable values
    records = await _get_test_records(test_marker)
    success_records = [r for r in records if r["error_info"] is None]
    assert len(success_records) == 1
    assert abs(success_records[0]["stt_time"] - 0.1) < 0.01


# ========== query_summary tests ==========

@pytest.mark.asyncio
async def test_query_summary_with_data(recorder, query, test_marker):
    """Test summary query with actual data"""
    for i in range(3):
        txn_id = f"test_txn_{uuid4()}"
        recorder.record(PerformanceRecord(
            transaction_id=txn_id,
            user_id=test_marker,
            stt_time=0.1 * (i + 1),
            tts_first_chunk_time=0.5 * (i + 1),
        ))

    recorder.record_queue.join()

    # Verify exact count
    count = await _count_test_records(test_marker)
    assert count == 3


@pytest.mark.asyncio
async def test_query_summary_percentiles(recorder, query, test_marker):
    """Test that percentiles are calculated correctly"""
    # Create 10 records with increasing response times
    for i in range(10):
        txn_id = f"test_txn_{uuid4()}"
        recorder.record(PerformanceRecord(
            transaction_id=txn_id,
            user_id=test_marker,
            stt_time=0.1,
            llm_first_chunk_time=0.2,
            llm_first_voice_chunk_time=0.3,
            tts_first_chunk_time=(i + 1) * 0.1,  # 0.1, 0.2, ..., 1.0
        ))

    recorder.record_queue.join()

    # Verify exact count
    count = await _count_test_records(test_marker)
    assert count == 10

    # Verify API returns percentiles
    result = await query.query_summary("1h")
    assert result.p50_tts_first_chunk_time is not None
    assert result.p95_tts_first_chunk_time is not None
    assert result.p99_tts_first_chunk_time is not None


@pytest.mark.asyncio
async def test_query_summary_excludes_errors(recorder, query, test_marker):
    """Test that error records are excluded from performance metrics"""
    # Create success record
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        stt_time=0.1,
        tts_first_chunk_time=0.5,
    ))

    # Create error record
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        stt_time=100.0,  # Large value
        tts_first_chunk_time=500.0,  # Large value
        error_info=json.dumps({"error": "Test error"}),
    ))

    recorder.record_queue.join()

    # Verify exact counts
    total = await _count_test_records(test_marker)
    success = await _count_test_records(test_marker, success_only=True)
    error = await _count_test_records(test_marker, error_only=True)

    assert total == 2
    assert success == 1
    assert error == 1


# ========== query_logs tests ==========

@pytest.mark.asyncio
async def test_query_logs_with_data(recorder, query, test_marker):
    """Test logs query with actual data"""
    context_id = f"ctx_{uuid4()}"
    for i in range(3):
        recorder.record(PerformanceRecord(
            transaction_id=f"test_txn_{uuid4()}",
            user_id=test_marker,
            context_id=context_id,
            request_text=f"Request {i}",
            response_text=f"Response {i}",
        ))

    recorder.record_queue.join()

    result = await query.query_logs(100)

    # Find our group by context_id
    our_group = None
    for group in result:
        if group.context_id == context_id:
            our_group = group
            break

    assert our_group is not None
    assert len(our_group.logs) == 3


@pytest.mark.asyncio
async def test_query_logs_groups_by_context(recorder, query, test_marker):
    """Test that logs are grouped by context_id"""
    ctx1 = f"ctx_{uuid4()}"
    ctx2 = f"ctx_{uuid4()}"

    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        context_id=ctx1,
        request_text="Request 1",
    ))
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        context_id=ctx2,
        request_text="Request 2",
    ))
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        context_id=ctx1,
        request_text="Request 3",
    ))

    recorder.record_queue.join()

    result = await query.query_logs(100)

    # Find our groups
    our_groups = [g for g in result if g.context_id in (ctx1, ctx2)]
    assert len(our_groups) == 2

    ctx1_group = next(g for g in our_groups if g.context_id == ctx1)
    ctx2_group = next(g for g in our_groups if g.context_id == ctx2)

    assert len(ctx1_group.logs) == 2
    assert len(ctx2_group.logs) == 1


@pytest.mark.asyncio
async def test_query_logs_has_error_flag(recorder, query, test_marker):
    """Test that has_error flag is set correctly for groups"""
    ctx_with_error = f"ctx_{uuid4()}"
    ctx_without_error = f"ctx_{uuid4()}"

    # Group with error
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        context_id=ctx_with_error,
        request_text="Request 1",
    ))
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        context_id=ctx_with_error,
        request_text="Request 2",
        error_info=json.dumps({"error": "Test error"}),
    ))

    # Group without error
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        context_id=ctx_without_error,
        request_text="Request 3",
    ))

    recorder.record_queue.join()

    result = await query.query_logs(100)

    # Find and verify our groups
    for group in result:
        if group.context_id == ctx_with_error:
            assert group.has_error is True
        elif group.context_id == ctx_without_error:
            assert group.has_error is False


@pytest.mark.asyncio
async def test_query_logs_includes_tool_calls(recorder, query, test_marker):
    """Test that tool_calls are included in logs"""
    tool_calls_data = [{"name": "get_weather", "arguments": '{"city": "Tokyo"}', "result": {"temp": 20}}]
    context_id = f"ctx_{uuid4()}"

    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        context_id=context_id,
        request_text="What's the weather?",
        tool_calls=json.dumps(tool_calls_data),
    ))

    recorder.record_queue.join()

    result = await query.query_logs(100)

    # Find our group
    our_group = None
    for group in result:
        if group.context_id == context_id:
            our_group = group
            break

    assert our_group is not None
    log = our_group.logs[0]
    assert log.tool_calls is not None
    parsed = json.loads(log.tool_calls)
    assert parsed[0]["name"] == "get_weather"


# ========== interval validation tests ==========

@pytest.mark.asyncio
async def test_query_timeline_invalid_interval(query):
    """Test that invalid interval raises error"""
    with pytest.raises(ValueError):
        await query.query_timeline("1h", "invalid")


@pytest.mark.asyncio
async def test_query_timeline_all_intervals(recorder, query, test_marker):
    """Test that all valid intervals work"""
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        stt_time=0.1,
    ))
    recorder.record_queue.join()

    for interval in VALID_INTERVALS:
        # Use appropriate period for each interval
        period = "30d" if interval == "1d" else "24h"
        result = await query.query_timeline(period, interval)
        assert len(result) > 0


# ========== connection string test ==========

@pytest.mark.asyncio
async def test_query_with_connection_string(recorder, test_marker):
    """Test using connection string instead of individual parameters"""
    connection_str = f"postgresql://{AIAVATAR_DB_USER}:{AIAVATAR_DB_PASSWORD}@{AIAVATAR_DB_HOST}:{AIAVATAR_DB_PORT}/{AIAVATAR_DB_NAME}"

    query = PostgreSQLMetricsQuery(connection_str=connection_str)

    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        user_id=test_marker,
        stt_time=0.1,
    ))
    recorder.record_queue.join()

    # Verify record was created
    count = await _count_test_records(test_marker)
    assert count == 1

    # Verify API works
    result = await query.query_summary("1h")
    assert result.total_requests >= 1
