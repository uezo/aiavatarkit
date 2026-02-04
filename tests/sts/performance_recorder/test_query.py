import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import pytest
from aiavatar.sts.performance_recorder import PerformanceRecord
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder
from aiavatar.sts.performance_recorder.query import (
    SQLiteMetricsQuery,
    parse_period,
    VALID_PERIODS,
    VALID_INTERVALS,
)


@pytest.fixture
def db_path():
    """Create a temporary database file for testing"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def recorder(db_path):
    rec = SQLitePerformanceRecorder(db_path=db_path)
    yield rec
    rec.close()


@pytest.fixture
def query(db_path, recorder):
    """Create query instance after recorder initializes the DB"""
    return SQLiteMetricsQuery(db_path=db_path)


def _update_created_at(db_path, transaction_id, new_created_at):
    """Helper to update created_at for testing time-based queries"""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE performance_records SET created_at = ? WHERE transaction_id = ?",
            (new_created_at, transaction_id)
        )
        conn.commit()
    finally:
        conn.close()


# ========== parse_period tests ==========

def test_parse_period_hours():
    assert parse_period("1h") == timedelta(hours=1)
    assert parse_period("6h") == timedelta(hours=6)
    assert parse_period("24h") == timedelta(hours=24)


def test_parse_period_days():
    assert parse_period("7d") == timedelta(days=7)
    assert parse_period("30d") == timedelta(days=30)


def test_parse_period_invalid():
    with pytest.raises(ValueError):
        parse_period("invalid")
    with pytest.raises(ValueError):
        parse_period("2h")  # Not in VALID_PERIODS


# ========== query_timeline tests ==========

@pytest.mark.asyncio
async def test_query_timeline_empty(query):
    """Test timeline query with no data"""
    result = await query.query_timeline("1h", "1m")
    assert len(result) > 0  # Should have empty buckets
    for bucket in result:
        assert bucket.request_count == 0
        assert bucket.success_count == 0
        assert bucket.error_count == 0


@pytest.mark.asyncio
async def test_query_timeline_with_data(recorder, db_path, query):
    """Test timeline query with actual data"""
    txn_id = f"test_txn_{uuid4()}"
    record = PerformanceRecord(
        transaction_id=txn_id,
        user_id="user1",
        context_id="ctx1",
        stt_time=0.1,
        llm_first_chunk_time=0.2,
        llm_first_voice_chunk_time=0.3,
        tts_first_chunk_time=0.5,
        total_time=1.0,
    )
    recorder.record(record)
    recorder.record_queue.join()

    result = await query.query_timeline("1h", "1m")

    # Find the bucket with data
    buckets_with_data = [b for b in result if b.request_count > 0]
    assert len(buckets_with_data) == 1
    bucket = buckets_with_data[0]
    assert bucket.request_count == 1
    assert bucket.success_count == 1
    assert bucket.error_count == 0


@pytest.mark.asyncio
async def test_query_timeline_excludes_old_data(recorder, db_path, query):
    """Test that timeline excludes data outside the period"""
    txn_id = f"test_txn_{uuid4()}"
    record = PerformanceRecord(
        transaction_id=txn_id,
        stt_time=0.1,
        tts_first_chunk_time=0.5,
    )
    recorder.record(record)
    recorder.record_queue.join()

    # Move created_at to 2 hours ago
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    _update_created_at(db_path, txn_id, old_time)

    # Query last 1 hour - should not include the old record
    result = await query.query_timeline("1h", "1m")
    buckets_with_data = [b for b in result if b.request_count > 0]
    assert len(buckets_with_data) == 0


@pytest.mark.asyncio
async def test_query_timeline_success_and_error_counts(recorder, db_path, query):
    """Test that success and error counts are correct"""
    # Create success record
    success_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=success_txn,
        stt_time=0.1,
        tts_first_chunk_time=0.5,
    ))

    # Create error record
    error_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=error_txn,
        stt_time=0.1,
        error_info=json.dumps({"error": "Test error"}),
    ))

    recorder.record_queue.join()

    result = await query.query_timeline("1h", "1m")
    buckets_with_data = [b for b in result if b.request_count > 0]
    assert len(buckets_with_data) == 1
    bucket = buckets_with_data[0]
    assert bucket.request_count == 2
    assert bucket.success_count == 1
    assert bucket.error_count == 1


@pytest.mark.asyncio
async def test_query_timeline_phase_calculations(recorder, db_path, query):
    """Test that phase calculations are correct"""
    txn_id = f"test_txn_{uuid4()}"
    record = PerformanceRecord(
        transaction_id=txn_id,
        stt_time=0.1,
        llm_first_chunk_time=0.2,
        llm_first_voice_chunk_time=0.4,
        tts_first_chunk_time=0.6,
    )
    recorder.record(record)
    recorder.record_queue.join()

    result = await query.query_timeline("1h", "1m")
    buckets_with_data = [b for b in result if b.request_count > 0]
    bucket = buckets_with_data[0]

    # STT phase = stt_time = 0.1
    assert abs(bucket.avg_stt_phase - 0.1) < 0.01
    # LLM phase = llm_first_chunk_time - stt_time = 0.2 - 0.1 = 0.1
    assert abs(bucket.avg_llm_phase - 0.1) < 0.01
    # Processing phase = llm_first_voice_chunk_time - llm_first_chunk_time = 0.4 - 0.2 = 0.2
    assert abs(bucket.avg_processing_phase - 0.2) < 0.01
    # TTS phase = tts_first_chunk_time - llm_first_voice_chunk_time = 0.6 - 0.4 = 0.2
    assert abs(bucket.avg_tts_phase - 0.2) < 0.01


@pytest.mark.asyncio
async def test_query_timeline_excludes_errors_from_phase_calc(recorder, db_path, query):
    """Test that error records are excluded from phase calculations"""
    # Create error record with timing data
    error_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=error_txn,
        stt_time=10.0,  # Large value that would skew average
        llm_first_chunk_time=20.0,
        tts_first_chunk_time=30.0,
        error_info=json.dumps({"error": "Test error"}),
    ))

    # Create success record
    success_txn = f"test_txn_{uuid4()}"
    recorder.record(PerformanceRecord(
        transaction_id=success_txn,
        stt_time=0.1,
        llm_first_chunk_time=0.2,
        llm_first_voice_chunk_time=0.3,
        tts_first_chunk_time=0.5,
    ))

    recorder.record_queue.join()

    result = await query.query_timeline("1h", "1m")
    buckets_with_data = [b for b in result if b.request_count > 0]
    bucket = buckets_with_data[0]

    # Phase calculations should only include success record
    assert abs(bucket.avg_stt_phase - 0.1) < 0.01
    assert abs(bucket.avg_tts_first_chunk_time - 0.5) < 0.01


# ========== query_summary tests ==========

@pytest.mark.asyncio
async def test_query_summary_empty(query):
    """Test summary query with no data"""
    result = await query.query_summary("1h")
    assert result.total_requests == 0
    assert result.success_count == 0
    assert result.error_count == 0


@pytest.mark.asyncio
async def test_query_summary_with_data(recorder, db_path, query):
    """Test summary query with actual data"""
    for i in range(3):
        txn_id = f"test_txn_{uuid4()}"
        recorder.record(PerformanceRecord(
            transaction_id=txn_id,
            stt_time=0.1 * (i + 1),
            tts_first_chunk_time=0.5 * (i + 1),
        ))

    recorder.record_queue.join()

    result = await query.query_summary("1h")
    assert result.total_requests == 3
    assert result.success_count == 3
    assert result.error_count == 0
    assert result.avg_tts_first_chunk_time is not None


@pytest.mark.asyncio
async def test_query_summary_percentiles(recorder, db_path, query):
    """Test that percentiles are calculated correctly"""
    # Create 10 records with increasing response times
    for i in range(10):
        txn_id = f"test_txn_{uuid4()}"
        recorder.record(PerformanceRecord(
            transaction_id=txn_id,
            stt_time=0.1,
            llm_first_chunk_time=0.2,
            llm_first_voice_chunk_time=0.3,
            tts_first_chunk_time=(i + 1) * 0.1,  # 0.1, 0.2, ..., 1.0
        ))

    recorder.record_queue.join()

    result = await query.query_summary("1h")
    assert result.total_requests == 10
    # p50 should be around 0.5-0.6
    assert 0.4 < result.p50_tts_first_chunk_time < 0.7
    # p95 should be around 0.9-1.0
    assert 0.8 < result.p95_tts_first_chunk_time < 1.1
    # p99 should be close to 1.0
    assert result.p99_tts_first_chunk_time >= 0.9


@pytest.mark.asyncio
async def test_query_summary_excludes_errors(recorder, db_path, query):
    """Test that error records are excluded from performance metrics"""
    # Create success record
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        stt_time=0.1,
        tts_first_chunk_time=0.5,
    ))

    # Create error record
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        stt_time=10.0,  # Large value
        tts_first_chunk_time=50.0,  # Large value
        error_info=json.dumps({"error": "Test error"}),
    ))

    recorder.record_queue.join()

    result = await query.query_summary("1h")
    assert result.total_requests == 2
    assert result.success_count == 1
    assert result.error_count == 1
    # Performance metrics should only reflect success record
    assert abs(result.avg_tts_first_chunk_time - 0.5) < 0.01


# ========== query_logs tests ==========

@pytest.mark.asyncio
async def test_query_logs_empty(query):
    """Test logs query with no data"""
    result = await query.query_logs(100)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_query_logs_with_data(recorder, db_path, query):
    """Test logs query with actual data"""
    context_id = f"ctx_{uuid4()}"
    for i in range(3):
        recorder.record(PerformanceRecord(
            transaction_id=f"test_txn_{uuid4()}",
            context_id=context_id,
            request_text=f"Request {i}",
            response_text=f"Response {i}",
        ))

    recorder.record_queue.join()

    result = await query.query_logs(100)
    assert len(result) == 1  # Grouped by context_id
    group = result[0]
    assert group.context_id == context_id
    assert len(group.logs) == 3


@pytest.mark.asyncio
async def test_query_logs_groups_by_context(recorder, db_path, query):
    """Test that logs are grouped by context_id"""
    ctx1 = f"ctx_{uuid4()}"
    ctx2 = f"ctx_{uuid4()}"

    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        context_id=ctx1,
        request_text="Request 1",
    ))
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        context_id=ctx2,
        request_text="Request 2",
    ))
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        context_id=ctx1,
        request_text="Request 3",
    ))

    recorder.record_queue.join()

    result = await query.query_logs(100)
    assert len(result) == 2

    ctx_ids = {g.context_id for g in result}
    assert ctx1 in ctx_ids
    assert ctx2 in ctx_ids


@pytest.mark.asyncio
async def test_query_logs_has_error_flag(recorder, db_path, query):
    """Test that has_error flag is set correctly for groups"""
    ctx_with_error = f"ctx_{uuid4()}"
    ctx_without_error = f"ctx_{uuid4()}"

    # Group with error
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        context_id=ctx_with_error,
        request_text="Request 1",
    ))
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        context_id=ctx_with_error,
        request_text="Request 2",
        error_info=json.dumps({"error": "Test error"}),
    ))

    # Group without error
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        context_id=ctx_without_error,
        request_text="Request 3",
    ))

    recorder.record_queue.join()

    result = await query.query_logs(100)

    for group in result:
        if group.context_id == ctx_with_error:
            assert group.has_error is True
        elif group.context_id == ctx_without_error:
            assert group.has_error is False


@pytest.mark.asyncio
async def test_query_logs_includes_tool_calls(recorder, db_path, query):
    """Test that tool_calls are included in logs"""
    tool_calls_data = [{"name": "get_weather", "arguments": '{"city": "Tokyo"}', "result": {"temp": 20}}]

    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        context_id=f"ctx_{uuid4()}",
        request_text="What's the weather?",
        tool_calls=json.dumps(tool_calls_data),
    ))

    recorder.record_queue.join()

    result = await query.query_logs(100)
    assert len(result) == 1
    log = result[0].logs[0]
    assert log.tool_calls is not None
    parsed = json.loads(log.tool_calls)
    assert parsed[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_query_logs_limit(recorder, db_path, query):
    """Test that limit parameter works correctly"""
    for i in range(10):
        recorder.record(PerformanceRecord(
            transaction_id=f"test_txn_{uuid4()}",
            context_id=f"ctx_{uuid4()}",  # Different context for each
            request_text=f"Request {i}",
        ))

    recorder.record_queue.join()

    result = await query.query_logs(5)
    total_logs = sum(len(g.logs) for g in result)
    assert total_logs == 5


# ========== interval validation tests ==========

@pytest.mark.asyncio
async def test_query_timeline_invalid_interval(query):
    """Test that invalid interval raises error"""
    with pytest.raises(ValueError):
        await query.query_timeline("1h", "invalid")


@pytest.mark.asyncio
async def test_query_timeline_all_intervals(recorder, db_path, query):
    """Test that all valid intervals work"""
    recorder.record(PerformanceRecord(
        transaction_id=f"test_txn_{uuid4()}",
        stt_time=0.1,
    ))
    recorder.record_queue.join()

    for interval in VALID_INTERVALS:
        # Use appropriate period for each interval
        period = "30d" if interval == "1d" else "24h"
        result = await query.query_timeline(period, interval)
        assert len(result) > 0
