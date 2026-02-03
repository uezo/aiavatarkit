import asyncio
import math
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional


VALID_PERIODS = {"1h", "6h", "24h", "7d", "30d"}
VALID_INTERVALS = {"1m", "5m", "15m", "1h", "1d"}

INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "1d": 86400,
}


def parse_period(period: str) -> timedelta:
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period: {period}. Must be one of {VALID_PERIODS}")
    m = re.match(r"(\d+)([hd])", period)
    value, unit = int(m.group(1)), m.group(2)
    if unit == "h":
        return timedelta(hours=value)
    return timedelta(days=value)


def percentile(sorted_values: List[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _safe_avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _bucket_key(dt: datetime, interval_seconds: int) -> str:
    ts = dt.timestamp()
    bucketed_ts = (int(ts) // interval_seconds) * interval_seconds
    bucketed_dt = datetime.fromtimestamp(bucketed_ts, tz=timezone.utc)
    return bucketed_dt.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class TimelineBucket:
    timestamp: str
    request_count: int  # Total (for backward compatibility)
    success_count: int = 0
    error_count: int = 0
    avg_tts_first_chunk_time: Optional[float] = None
    p50_tts_first_chunk_time: Optional[float] = None
    p95_tts_first_chunk_time: Optional[float] = None
    avg_stt_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_tts_phase: Optional[float] = None


@dataclass
class MetricsSummary:
    total_requests: int  # Total (for backward compatibility)
    success_count: int = 0
    error_count: int = 0
    avg_tts_first_chunk_time: Optional[float] = None
    p50_tts_first_chunk_time: Optional[float] = None
    p95_tts_first_chunk_time: Optional[float] = None
    p99_tts_first_chunk_time: Optional[float] = None
    avg_stt_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_tts_phase: Optional[float] = None


def _compute_phases(rows):
    """Extract phase values from rows. Each row: (created_at, stt_time, llm_first_voice_chunk_time, tts_first_chunk_time, error_info)
    Only processes rows without errors for performance metrics."""
    stt_phases = []
    llm_phases = []
    tts_phases = []
    tts_first_chunks = []

    for created_at, stt_time, llm_fvc_time, tts_fc_time, error_info in rows:
        # Skip error records for performance metrics
        if error_info:
            continue
        if stt_time and stt_time > 0:
            stt_phases.append(stt_time)
        if llm_fvc_time and stt_time and llm_fvc_time > stt_time:
            llm_phases.append(llm_fvc_time - stt_time)
        if tts_fc_time and llm_fvc_time and tts_fc_time > llm_fvc_time:
            tts_phases.append(tts_fc_time - llm_fvc_time)
        if tts_fc_time and tts_fc_time > 0:
            tts_first_chunks.append(tts_fc_time)

    return stt_phases, llm_phases, tts_phases, tts_first_chunks


def _build_timeline(rows, interval_seconds: int, start_time: datetime) -> List[TimelineBucket]:
    # Group rows by bucket key
    data_buckets = {}
    for created_at, stt_time, llm_fvc_time, tts_fc_time, error_info in rows:
        key = _bucket_key(created_at, interval_seconds)
        if key not in data_buckets:
            data_buckets[key] = []
        data_buckets[key].append((created_at, stt_time, llm_fvc_time, tts_fc_time, error_info))

    # Generate all bucket keys for the full range (start_time -> now)
    now = datetime.now(timezone.utc)
    start_ts = (int(start_time.timestamp()) // interval_seconds) * interval_seconds
    end_ts = (int(now.timestamp()) // interval_seconds) * interval_seconds

    result = []
    current_ts = start_ts
    while current_ts <= end_ts:
        key = datetime.fromtimestamp(current_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        bucket_rows = data_buckets.get(key, [])
        if bucket_rows:
            # Count success and error
            error_count = sum(1 for r in bucket_rows if r[4])  # error_info is at index 4
            success_count = len(bucket_rows) - error_count
            # Compute phases (only from success records)
            stt_phases, llm_phases, tts_phases, tts_first_chunks = _compute_phases(bucket_rows)
            sorted_tts = sorted(tts_first_chunks)
            result.append(TimelineBucket(
                timestamp=key,
                request_count=len(bucket_rows),
                success_count=success_count,
                error_count=error_count,
                avg_tts_first_chunk_time=_safe_avg(tts_first_chunks),
                p50_tts_first_chunk_time=percentile(sorted_tts, 50),
                p95_tts_first_chunk_time=percentile(sorted_tts, 95),
                avg_stt_phase=_safe_avg(stt_phases),
                avg_llm_phase=_safe_avg(llm_phases),
                avg_tts_phase=_safe_avg(tts_phases),
            ))
        else:
            result.append(TimelineBucket(timestamp=key, request_count=0, success_count=0, error_count=0))
        current_ts += interval_seconds
    return result


def _build_summary(rows) -> MetricsSummary:
    if not rows:
        return MetricsSummary(total_requests=0, success_count=0, error_count=0)

    # Count success and error
    error_count = sum(1 for r in rows if r[4])  # error_info is at index 4
    success_count = len(rows) - error_count

    # Compute phases (only from success records)
    stt_phases, llm_phases, tts_phases, tts_first_chunks = _compute_phases(rows)
    sorted_tts = sorted(tts_first_chunks)
    return MetricsSummary(
        total_requests=len(rows),
        success_count=success_count,
        error_count=error_count,
        avg_tts_first_chunk_time=_safe_avg(tts_first_chunks),
        p50_tts_first_chunk_time=percentile(sorted_tts, 50),
        p95_tts_first_chunk_time=percentile(sorted_tts, 95),
        p99_tts_first_chunk_time=percentile(sorted_tts, 99),
        avg_stt_phase=_safe_avg(stt_phases),
        avg_llm_phase=_safe_avg(llm_phases),
        avg_tts_phase=_safe_avg(tts_phases),
    )


@dataclass
class ConversationLog:
    created_at: str
    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    tts_first_chunk_time: Optional[float] = None
    request_text: Optional[str] = None
    request_files: Optional[str] = None
    response_text: Optional[str] = None
    response_voice_text: Optional[str] = None
    error_info: Optional[str] = None


@dataclass
class ConversationGroup:
    context_id: Optional[str]
    logs: List[ConversationLog]
    has_error: bool = False


_QUERY_SQL = """
SELECT created_at, stt_time, llm_first_voice_chunk_time, tts_first_chunk_time, error_info
FROM performance_records
WHERE created_at >= ?
ORDER BY created_at
"""

_LOGS_SQL = """
SELECT created_at, transaction_id, user_id, context_id, tts_first_chunk_time,
       request_text, request_files, response_text, response_voice_text, error_info
FROM performance_records
ORDER BY created_at DESC
LIMIT ?
"""


class MetricsQuery(ABC):
    @abstractmethod
    async def query_timeline(self, period: str, interval: str) -> List[TimelineBucket]:
        pass

    @abstractmethod
    async def query_summary(self, period: str) -> MetricsSummary:
        pass

    @abstractmethod
    async def query_logs(self, limit: int) -> List[ConversationGroup]:
        pass


def _group_logs(rows) -> List[ConversationGroup]:
    logs = []
    for row in rows:
        created_at = row[0]
        if isinstance(created_at, datetime):
            created_at = created_at.strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(created_at, str):
            pass  # already string
        logs.append(ConversationLog(
            created_at=created_at,
            transaction_id=row[1],
            user_id=row[2],
            context_id=row[3],
            tts_first_chunk_time=row[4],
            request_text=row[5],
            request_files=row[6],
            response_text=row[7],
            response_voice_text=row[8],
            error_info=row[9],
        ))

    # Group by context_id, preserving order of first appearance
    groups_map = {}
    group_order = []
    for log in logs:
        key = log.context_id or ""
        if key not in groups_map:
            groups_map[key] = []
            group_order.append(key)
        groups_map[key].append(log)

    result = []
    for key in group_order:
        group_logs = groups_map[key]
        # Sort within group by created_at ascending
        group_logs.sort(key=lambda l: l.created_at)
        # Check if any log in the group has an error
        has_error = any(log.error_info for log in group_logs)
        result.append(ConversationGroup(
            context_id=key or None,
            logs=group_logs,
            has_error=has_error,
        ))
    return result


class SQLiteMetricsQuery(MetricsQuery):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _fetch_rows(self, start_time: datetime):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        try:
            rows = conn.execute(_QUERY_SQL, (start_time.isoformat(),)).fetchall()
            result = []
            for row in rows:
                created_at = row[0]
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                result.append((created_at, row[1], row[2], row[3], row[4]))
            return result
        finally:
            conn.close()

    def _fetch_logs(self, limit: int):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        try:
            return conn.execute(_LOGS_SQL, (limit,)).fetchall()
        finally:
            conn.close()

    async def query_timeline(self, period: str, interval: str) -> List[TimelineBucket]:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}")
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await asyncio.to_thread(self._fetch_rows, start_time)
        return _build_timeline(rows, INTERVAL_SECONDS[interval], start_time)

    async def query_summary(self, period: str) -> MetricsSummary:
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await asyncio.to_thread(self._fetch_rows, start_time)
        return _build_summary(rows)

    async def query_logs(self, limit: int) -> List[ConversationGroup]:
        rows = await asyncio.to_thread(self._fetch_logs, limit)
        return _group_logs(rows)


class PostgreSQLMetricsQuery(MetricsQuery):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        connection_str: str = None,
    ):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection_str = connection_str
        self._pool = None

    async def _get_pool(self):
        if self._pool is not None:
            return self._pool
        import asyncpg
        if self.connection_str:
            self._pool = await asyncpg.create_pool(dsn=self.connection_str, min_size=1, max_size=2)
        else:
            self._pool = await asyncpg.create_pool(
                host=self.host, port=self.port, database=self.dbname,
                user=self.user, password=self.password, min_size=1, max_size=2,
            )
        return self._pool

    async def _fetch_rows(self, start_time: datetime):
        pool = await self._get_pool()
        query = """
        SELECT created_at, stt_time, llm_first_voice_chunk_time, tts_first_chunk_time, error_info
        FROM performance_records
        WHERE created_at >= $1
        ORDER BY created_at
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, start_time)
        return [(r["created_at"], r["stt_time"], r["llm_first_voice_chunk_time"], r["tts_first_chunk_time"], r["error_info"]) for r in records]

    async def _fetch_logs(self, limit: int):
        pool = await self._get_pool()
        query = """
        SELECT created_at, transaction_id, user_id, context_id, tts_first_chunk_time,
               request_text, request_files, response_text, response_voice_text, error_info
        FROM performance_records
        ORDER BY created_at DESC
        LIMIT $1
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, limit)
        return [(r["created_at"], r["transaction_id"], r["user_id"], r["context_id"],
                 r["tts_first_chunk_time"], r["request_text"], r["request_files"],
                 r["response_text"], r["response_voice_text"], r["error_info"]) for r in records]

    async def query_timeline(self, period: str, interval: str) -> List[TimelineBucket]:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}")
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await self._fetch_rows(start_time)
        return _build_timeline(rows, INTERVAL_SECONDS[interval], start_time)

    async def query_summary(self, period: str) -> MetricsSummary:
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await self._fetch_rows(start_time)
        return _build_summary(rows)

    async def query_logs(self, limit: int) -> List[ConversationGroup]:
        rows = await self._fetch_logs(limit)
        return _group_logs(rows)


def create_metrics_query(recorder) -> MetricsQuery:
    from .sqlite import SQLitePerformanceRecorder
    if isinstance(recorder, SQLitePerformanceRecorder):
        return SQLiteMetricsQuery(recorder.db_path)

    from .postgres import PostgreSQLPerformanceRecorder
    if isinstance(recorder, PostgreSQLPerformanceRecorder):
        return PostgreSQLMetricsQuery(
            host=recorder.host,
            port=recorder.port,
            dbname=recorder.dbname,
            user=recorder.user,
            password=recorder.password,
            connection_str=recorder.connection_str,
        )

    raise ValueError(f"Unsupported recorder type: {type(recorder)}")
