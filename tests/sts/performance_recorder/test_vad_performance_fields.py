import sqlite3
from datetime import datetime, timezone

import pytest

from aiavatar.sts.performance_recorder import PerformanceRecord
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder


def test_sqlite_fresh_schema_orders_timing_columns_from_speech_end(tmp_path):
    db_path = tmp_path / "fresh-performance.db"
    recorder = SQLitePerformanceRecorder(db_path=str(db_path))
    recorder.close()

    conn = sqlite3.connect(db_path)
    try:
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(performance_records)")
        ]
    finally:
        conn.close()

    assert columns == [
        "id",
        "created_at",
        "transaction_id",
        "user_id",
        "context_id",
        "voice_length",
        "speech_end_at",
        "silence_threshold_time",
        "stt_after_threshold_time",
        "turn_end_gate_time",
        "turn_end_gate_held",
        "stt_time",
        "stop_response_time",
        "before_llm_time",
        "llm_first_chunk_time",
        "llm_first_voice_chunk_time",
        "llm_time",
        "tts_first_chunk_time",
        "tts_time",
        "total_time",
        "stt_name",
        "llm_name",
        "tts_name",
        "request_text",
        "request_files",
        "response_text",
        "response_voice_text",
        "quick_response_text",
        "error_info",
        "tool_calls",
    ]


def test_sqlite_records_vad_performance_fields_and_nulls(tmp_path):
    db_path = tmp_path / "performance.db"
    recorder = SQLitePerformanceRecorder(db_path=str(db_path))
    speech_end_at = datetime.now(timezone.utc)
    try:
        recorder.record(PerformanceRecord(
            transaction_id="silero",
            speech_end_at=speech_end_at,
            silence_threshold_time=0.5,
            stt_after_threshold_time=0.12,
            turn_end_gate_time=0.3,
            turn_end_gate_held=True,
        ))
        recorder.record(PerformanceRecord(
            transaction_id="unsupported-vad",
            stt_time=0.25,
        ))
        recorder.record_queue.join()
    finally:
        recorder.close()

    conn = sqlite3.connect(db_path)
    try:
        silero = conn.execute(
            """
            SELECT speech_end_at, silence_threshold_time,
                   stt_after_threshold_time, turn_end_gate_time,
                   turn_end_gate_held, stt_time
            FROM performance_records
            WHERE transaction_id = ?
            """,
            ("silero",),
        ).fetchone()
        unsupported = conn.execute(
            """
            SELECT speech_end_at, silence_threshold_time,
                   stt_after_threshold_time, turn_end_gate_time,
                   turn_end_gate_held, stt_time
            FROM performance_records
            WHERE transaction_id = ?
            """,
            ("unsupported-vad",),
        ).fetchone()
    finally:
        conn.close()

    assert silero[0] == str(speech_end_at)
    assert silero[1:] == pytest.approx((0.5, 0.62, 0.92, 1, 0.0))
    assert unsupported == (None, None, None, None, None, 0.25)


@pytest.mark.parametrize(
    ("time_origin", "expected_offset"),
    [
        (None, 0.92),
        ("pipeline_start", 0.0),
    ],
)
def test_sqlite_time_origin_rebases_vad_and_pipeline_lap_times(
    tmp_path,
    time_origin,
    expected_offset,
):
    db_path = tmp_path / f"performance-{time_origin or 'default'}.db"
    recorder_kwargs = {"db_path": str(db_path)}
    if time_origin is not None:
        recorder_kwargs["time_origin"] = time_origin
    recorder = SQLitePerformanceRecorder(**recorder_kwargs)

    raw_lap_times = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    record = PerformanceRecord(
        transaction_id="time-origin",
        silence_threshold_time=0.5,
        stt_after_threshold_time=0.12,
        turn_end_gate_time=0.3,
        stt_time=raw_lap_times[0],
        stop_response_time=raw_lap_times[1],
        before_llm_time=raw_lap_times[2],
        llm_first_chunk_time=raw_lap_times[3],
        llm_first_voice_chunk_time=raw_lap_times[4],
        llm_time=raw_lap_times[5],
        tts_first_chunk_time=raw_lap_times[6],
        tts_time=raw_lap_times[7],
        total_time=raw_lap_times[8],
    )
    try:
        recorder.record(record)
        recorder.record_queue.join()
    finally:
        recorder.close()

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT silence_threshold_time, stt_after_threshold_time,
                   turn_end_gate_time, stt_time, stop_response_time,
                   before_llm_time, llm_first_chunk_time,
                   llm_first_voice_chunk_time, llm_time,
                   tts_first_chunk_time, tts_time, total_time
            FROM performance_records
            WHERE transaction_id = ?
            """,
            ("time-origin",),
        ).fetchone()
    finally:
        conn.close()

    expected_vad_laps = (
        (0.5, 0.62, 0.92)
        if time_origin is None
        else (0.5, 0.12, 0.3)
    )
    assert row[:3] == pytest.approx(expected_vad_laps)
    assert row[3:] == pytest.approx(tuple(
        value + expected_offset
        for value in raw_lap_times
    ))
    assert record.stt_time == 0.1
    assert record.stt_after_threshold_time == 0.12
    assert record.turn_end_gate_time == 0.3


def test_sqlite_user_speech_end_origin_keeps_unreached_laps_at_zero(tmp_path):
    db_path = tmp_path / "performance-zero-laps.db"
    recorder = SQLitePerformanceRecorder(db_path=str(db_path))
    try:
        recorder.record(PerformanceRecord(
            transaction_id="partial-record",
            silence_threshold_time=0.5,
            stt_after_threshold_time=0.1,
            turn_end_gate_time=0.2,
            stt_time=0.01,
        ))
        recorder.record_queue.join()
    finally:
        recorder.close()

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT stt_time, stop_response_time, before_llm_time, total_time
            FROM performance_records
            WHERE transaction_id = ?
            """,
            ("partial-record",),
        ).fetchone()
    finally:
        conn.close()

    assert row == pytest.approx((0.81, 0.0, 0.0, 0.0))


def test_sqlite_rejects_invalid_time_origin(tmp_path):
    with pytest.raises(ValueError, match="Invalid time_origin"):
        SQLitePerformanceRecorder(
            db_path=str(tmp_path / "invalid-origin.db"),
            time_origin="invalid",
        )


def test_sqlite_migrates_vad_performance_columns(tmp_path):
    db_path = tmp_path / "existing-performance.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE performance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP,
                transaction_id TEXT,
                user_id TEXT,
                context_id TEXT,
                request_files TEXT,
                before_llm_time REAL,
                quick_response_text TEXT,
                error_info TEXT,
                tool_calls TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    recorder = SQLitePerformanceRecorder(db_path=str(db_path))
    recorder.close()

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(performance_records)")
        }
    finally:
        conn.close()

    assert columns["speech_end_at"] == "TIMESTAMP"
    assert columns["silence_threshold_time"] == "REAL"
    assert columns["stt_after_threshold_time"] == "REAL"
    assert columns["turn_end_gate_time"] == "REAL"
    assert columns["turn_end_gate_held"] == "INTEGER"
