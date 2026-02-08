import os
import sqlite3
import tempfile
from uuid import uuid4
import pytest
from aiavatar.sts.performance_recorder import PerformanceRecord
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder


@pytest.fixture
def db_path():
    """Create a temporary database file for testing"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def recorder(db_path):
    rec = SQLitePerformanceRecorder(db_path=db_path)
    yield rec
    rec.close()


@pytest.fixture
def unique_transaction_id():
    return f"test_txn_{uuid4()}"


def test_record_single(recorder, db_path, unique_transaction_id):
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

    # Verify the record was inserted
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM performance_records WHERE transaction_id = ?",
            (unique_transaction_id,)
        )
        row = cursor.fetchone()
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
        conn.close()


def test_record_multiple(recorder, db_path):
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

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for i, txn_id in enumerate(transaction_ids):
            cursor = conn.execute(
                "SELECT * FROM performance_records WHERE transaction_id = ?",
                (txn_id,)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["user_id"] == f"user_{i}"
            assert row["context_id"] == f"context_{i}"
            assert row["request_text"] == f"Request {i}"
    finally:
        conn.close()


def test_record_with_none_values(recorder, db_path, unique_transaction_id):
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

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM performance_records WHERE transaction_id = ?",
            (unique_transaction_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["user_id"] is None
        assert row["context_id"] is None
        assert row["request_text"] is None
        assert row["response_text"] is None
    finally:
        conn.close()


def test_close_flushes_queue(db_path, unique_transaction_id):
    """Test that close() waits for all queued records to be processed"""
    recorder = SQLitePerformanceRecorder(db_path=db_path)

    record = PerformanceRecord(
        transaction_id=unique_transaction_id,
        request_text="Final record"
    )

    recorder.record(record)
    recorder.close()

    # After close, the record should be in the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM performance_records WHERE transaction_id = ?",
            (unique_transaction_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["request_text"] == "Final record"
    finally:
        conn.close()


def test_init_db_creates_table(recorder, db_path):
    """Test that init_db creates the performance_records table"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='performance_records'
            """
        )
        row = cursor.fetchone()
        assert row is not None
    finally:
        conn.close()


def test_init_db_creates_indexes(recorder, db_path):
    """Test that init_db creates the required indexes"""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='performance_records'
            """
        )
        rows = cursor.fetchall()
        index_names = {row[0] for row in rows}
        assert "idx_created_at" in index_names
        assert "idx_transaction_id" in index_names
        assert "idx_user_id" in index_names
        assert "idx_context_id" in index_names
    finally:
        conn.close()


def test_default_db_path():
    """Test using default database path"""
    default_path = "aiavatar.db"
    recorder = SQLitePerformanceRecorder()

    try:
        assert recorder.db_path == default_path

        record = PerformanceRecord(
            transaction_id=f"test_txn_{uuid4()}",
            request_text="Default path test"
        )
        recorder.record(record)
        recorder.record_queue.join()

        # Verify the database file was created
        assert os.path.exists(default_path)
    finally:
        recorder.close()
        # Cleanup default db file
        if os.path.exists(default_path):
            os.remove(default_path)
