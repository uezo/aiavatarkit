import asyncio
import math
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from .base import TIME_ORIGIN_PIPELINE_START, TIME_ORIGIN_USER_SPEECH_END


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
    avg_before_llm_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_processing_phase: Optional[float] = None
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
    avg_before_llm_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_processing_phase: Optional[float] = None
    avg_tts_phase: Optional[float] = None


@dataclass
class DetailedTimelineBucket:
    timestamp: str
    request_count: int
    success_count: int = 0
    error_count: int = 0
    measured_count: int = 0
    avg_first_response_time: Optional[float] = None
    avg_silence_detection_phase: Optional[float] = None
    avg_streaming_stt_finalization_phase: Optional[float] = None
    avg_turn_end_gate_phase: Optional[float] = None
    avg_stt_phase: Optional[float] = None
    avg_stop_response_phase: Optional[float] = None
    avg_before_llm_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_processing_phase: Optional[float] = None
    avg_tts_phase: Optional[float] = None


@dataclass
class DetailedMetricsSummary:
    total_requests: int
    success_count: int = 0
    error_count: int = 0
    measured_count: int = 0
    avg_first_response_time: Optional[float] = None
    p50_first_response_time: Optional[float] = None
    p95_first_response_time: Optional[float] = None
    p99_first_response_time: Optional[float] = None
    avg_silence_detection_phase: Optional[float] = None
    avg_streaming_stt_finalization_phase: Optional[float] = None
    avg_turn_end_gate_phase: Optional[float] = None
    avg_stt_phase: Optional[float] = None
    avg_stop_response_phase: Optional[float] = None
    avg_before_llm_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_processing_phase: Optional[float] = None
    avg_tts_phase: Optional[float] = None


_DETAILED_PHASE_FIELDS = (
    "avg_silence_detection_phase",
    "avg_streaming_stt_finalization_phase",
    "avg_turn_end_gate_phase",
    "avg_stt_phase",
    "avg_stop_response_phase",
    "avg_before_llm_phase",
    "avg_llm_phase",
    "avg_processing_phase",
    "avg_tts_phase",
)


def _compute_phases(rows):
    """Extract phase values from rows. Each row: (created_at, stt_time, before_llm_time, llm_first_chunk_time, llm_first_voice_chunk_time, tts_first_chunk_time, quick_response_text, error_info)
    Only processes rows without errors for performance metrics."""
    stt_phases = []
    before_llm_phases = []
    llm_phases = []
    processing_phases = []
    tts_phases = []
    first_response_times = []

    for created_at, stt_time, before_llm_time, llm_fc_time, llm_fvc_time, tts_fc_time, qr_text, error_info in rows:
        # Skip error records for performance metrics
        if error_info:
            continue
        if stt_time and stt_time > 0:
            stt_phases.append(stt_time)
        # Before LLM phase: stt_time -> before_llm_time (on_before_llm including QuickResponder)
        if before_llm_time and stt_time and before_llm_time > stt_time:
            before_llm_phases.append(before_llm_time - stt_time)
        # LLM phase: before_llm_time -> llm_first_chunk_time
        if llm_fc_time and before_llm_time and llm_fc_time > before_llm_time:
            llm_phases.append(llm_fc_time - before_llm_time)
        elif llm_fc_time and stt_time and llm_fc_time > stt_time and not before_llm_time:
            # Fallback for old records without before_llm_time
            llm_phases.append(llm_fc_time - stt_time)
        # Processing phase: llm_first_chunk_time -> llm_first_voice_chunk_time
        if llm_fvc_time and llm_fc_time and llm_fvc_time > llm_fc_time:
            processing_phases.append(llm_fvc_time - llm_fc_time)
        if tts_fc_time and llm_fvc_time and tts_fc_time > llm_fvc_time:
            tts_phases.append(tts_fc_time - llm_fvc_time)
        # First response time: before_llm_time if quick response, else tts_first_chunk_time
        if qr_text and before_llm_time and before_llm_time > 0:
            first_response_times.append(before_llm_time)
        elif tts_fc_time and tts_fc_time > 0:
            first_response_times.append(tts_fc_time)

    return stt_phases, before_llm_phases, llm_phases, processing_phases, tts_phases, first_response_times


def _build_timeline(rows, interval_seconds: int, start_time: datetime) -> List[TimelineBucket]:
    # Group rows by bucket key
    data_buckets = {}
    for row in rows:
        key = _bucket_key(row[0], interval_seconds)  # row[0] = created_at
        if key not in data_buckets:
            data_buckets[key] = []
        data_buckets[key].append(row)

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
            error_count = sum(1 for r in bucket_rows if r[7])  # error_info is at index 7
            success_count = len(bucket_rows) - error_count
            # Compute phases (only from success records)
            stt_phases, before_llm_phases, llm_phases, processing_phases, tts_phases, first_response_times = _compute_phases(bucket_rows)
            sorted_frt = sorted(first_response_times)
            result.append(TimelineBucket(
                timestamp=key,
                request_count=len(bucket_rows),
                success_count=success_count,
                error_count=error_count,
                avg_tts_first_chunk_time=_safe_avg(first_response_times),
                p50_tts_first_chunk_time=percentile(sorted_frt, 50),
                p95_tts_first_chunk_time=percentile(sorted_frt, 95),
                avg_stt_phase=_safe_avg(stt_phases),
                avg_before_llm_phase=_safe_avg(before_llm_phases),
                avg_llm_phase=_safe_avg(llm_phases),
                avg_processing_phase=_safe_avg(processing_phases),
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
    error_count = sum(1 for r in rows if r[7])  # error_info is at index 7
    success_count = len(rows) - error_count

    # Compute phases (only from success records)
    stt_phases, before_llm_phases, llm_phases, processing_phases, tts_phases, first_response_times = _compute_phases(rows)
    sorted_frt = sorted(first_response_times)
    return MetricsSummary(
        total_requests=len(rows),
        success_count=success_count,
        error_count=error_count,
        avg_tts_first_chunk_time=_safe_avg(first_response_times),
        p50_tts_first_chunk_time=percentile(sorted_frt, 50),
        p95_tts_first_chunk_time=percentile(sorted_frt, 95),
        p99_tts_first_chunk_time=percentile(sorted_frt, 99),
        avg_stt_phase=_safe_avg(stt_phases),
        avg_before_llm_phase=_safe_avg(before_llm_phases),
        avg_llm_phase=_safe_avg(llm_phases),
        avg_processing_phase=_safe_avg(processing_phases),
        avg_tts_phase=_safe_avg(tts_phases),
    )


def _raw_phase_vector(row, time_origin: str):
    """Return nine contiguous phases from speech end to first response.

    Detailed rows are ordered as defined by ``_DETAILED_QUERY_SQL``. Records
    without a speech-end timestamp, errors, or a first-response endpoint are
    intentionally excluded from the detailed cohort.
    """
    if row[12] or row[1] is None:
        return None

    def positive(value):
        return max(float(value or 0), 0.0)

    # Validate the actual stored endpoint before carrying missing intermediate
    # boundaries forward. Otherwise a silence timing alone can be mistaken for
    # a completed first-response measurement.
    endpoint_index = 7 if row[11] else 10
    if positive(row[endpoint_index]) <= 0:
        return None

    stored_origin = time_origin
    if time_origin == TIME_ORIGIN_USER_SPEECH_END:
        # Timing columns added before origin rebasing may coexist with newer
        # rows in the same table. A decreasing, explicitly stored VAD boundary
        # identifies an older pipeline-relative row without guessing from the
        # pipeline duration itself.
        previous = None
        for value in row[2:5]:
            if value is None:
                continue
            current = positive(value)
            if previous is not None and current < previous:
                stored_origin = TIME_ORIGIN_PIPELINE_START
                break
            previous = current

    if stored_origin == TIME_ORIGIN_PIPELINE_START:
        silence = positive(row[2])
        streaming_stt = positive(row[3])
        turn_end_gate = positive(row[4])
        pre_pipeline = silence + streaming_stt + turn_end_gate
        points = [
            0.0,
            silence,
            silence + streaming_stt,
            pre_pipeline,
            pre_pipeline + positive(row[5]),
            pre_pipeline + positive(row[6]),
            pre_pipeline + positive(row[7]),
            pre_pipeline + positive(row[8]),
            pre_pipeline + positive(row[9]),
            pre_pipeline + positive(row[10]),
        ]
    else:
        points = [
            0.0,
            positive(row[2]),
            positive(row[3]),
            positive(row[4]),
            positive(row[5]),
            positive(row[6]),
            positive(row[7]),
            positive(row[8]),
            positive(row[9]),
            positive(row[10]),
        ]

    # Missing timings are represented by zero in historical records. Carry the
    # previous boundary forward so a later known lap absorbs the unknown gap and
    # the stack remains equal to the measured first-response time.
    for index in range(1, len(points)):
        points[index] = max(points[index], points[index - 1])

    endpoint_index = 6 if row[11] else 9
    if points[endpoint_index] <= 0:
        return None
    points = points[:endpoint_index + 1]
    points.extend([points[-1]] * (10 - len(points)))
    return [points[index + 1] - points[index] for index in range(9)]


def _phase_vector(row, time_origin: str):
    """Return the nine display phases used by aggregate metrics."""
    return _raw_phase_vector(row, time_origin)


def _detailed_values(rows, time_origin: str):
    vectors = [vector for row in rows if (vector := _phase_vector(row, time_origin)) is not None]
    if not vectors:
        return 0, None, [None] * len(_DETAILED_PHASE_FIELDS), []
    phase_averages = [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(_DETAILED_PHASE_FIELDS))
    ]
    first_response_times = [sum(vector) for vector in vectors]
    return len(vectors), _safe_avg(first_response_times), phase_averages, first_response_times


def _detailed_fields(rows, time_origin: str):
    measured_count, average, phase_averages, first_response_times = _detailed_values(rows, time_origin)
    values = dict(zip(_DETAILED_PHASE_FIELDS, phase_averages))
    values.update(measured_count=measured_count, avg_first_response_time=average)
    return values, first_response_times


def _build_detailed_timeline(rows, interval_seconds: int, start_time: datetime, time_origin: str):
    data_buckets = {}
    for row in rows:
        data_buckets.setdefault(_bucket_key(row[0], interval_seconds), []).append(row)

    now = datetime.now(timezone.utc)
    current_ts = (int(start_time.timestamp()) // interval_seconds) * interval_seconds
    end_ts = (int(now.timestamp()) // interval_seconds) * interval_seconds
    result = []
    while current_ts <= end_ts:
        key = datetime.fromtimestamp(current_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        bucket_rows = data_buckets.get(key, [])
        error_count = sum(1 for row in bucket_rows if row[12])
        values, _ = _detailed_fields(bucket_rows, time_origin)
        result.append(DetailedTimelineBucket(
            timestamp=key,
            request_count=len(bucket_rows),
            success_count=len(bucket_rows) - error_count,
            error_count=error_count,
            **values,
        ))
        current_ts += interval_seconds
    return result


def _build_detailed_summary(rows, time_origin: str):
    error_count = sum(1 for row in rows if row[12])
    values, response_times = _detailed_fields(rows, time_origin)
    sorted_times = sorted(response_times)
    return DetailedMetricsSummary(
        total_requests=len(rows),
        success_count=len(rows) - error_count,
        error_count=error_count,
        p50_first_response_time=percentile(sorted_times, 50),
        p95_first_response_time=percentile(sorted_times, 95),
        p99_first_response_time=percentile(sorted_times, 99),
        **values,
    )


@dataclass
class ConversationLog:
    created_at: str
    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context_id: Optional[str] = None
    tts_first_chunk_time: Optional[float] = None
    before_llm_time: Optional[float] = None
    quick_response_text: Optional[str] = None
    request_text: Optional[str] = None
    request_files: Optional[str] = None
    response_text: Optional[str] = None
    response_voice_text: Optional[str] = None
    error_info: Optional[str] = None
    tool_calls: Optional[str] = None
    timing_breakdown: Optional["TurnTimingBreakdown"] = None


@dataclass
class TurnTimingBreakdown:
    total_first_response: float
    silence_detection: float
    streaming_stt_finalization: float
    turn_end_gate: float
    stt: float
    stop_response: float
    before_llm: float
    llm: float
    processing: float
    tts: float


@dataclass
class ConversationGroup:
    context_id: Optional[str]
    logs: List[ConversationLog]
    has_error: bool = False


_QUERY_SQL = """
SELECT COALESCE(speech_end_at, created_at) AS event_at,
       stt_time, before_llm_time, llm_first_chunk_time, llm_first_voice_chunk_time, tts_first_chunk_time, quick_response_text, error_info
FROM performance_records
WHERE COALESCE(speech_end_at, created_at) >= ?
ORDER BY event_at
"""

_DETAILED_QUERY_SQL = """
SELECT COALESCE(speech_end_at, created_at) AS event_at,
       speech_end_at, silence_threshold_time, stt_after_threshold_time, turn_end_gate_time,
       stt_time, stop_response_time, before_llm_time, llm_first_chunk_time, llm_first_voice_chunk_time,
       tts_first_chunk_time, quick_response_text, error_info
FROM performance_records
WHERE COALESCE(speech_end_at, created_at) >= ?
ORDER BY event_at
"""

_LOGS_SELECT_SQL = """
SELECT COALESCE(speech_end_at, created_at) AS event_at,
       transaction_id, user_id, session_id, context_id, tts_first_chunk_time, before_llm_time, quick_response_text,
       request_text, request_files, response_text, response_voice_text, error_info, tool_calls,
       speech_end_at, silence_threshold_time, stt_after_threshold_time, turn_end_gate_time,
       stt_time, stop_response_time, llm_first_chunk_time, llm_first_voice_chunk_time
FROM performance_records
"""


class MetricsQuery(ABC):
    @abstractmethod
    async def query_timeline(self, period: str, interval: str) -> List[TimelineBucket]:
        pass

    @abstractmethod
    async def query_summary(self, period: str) -> MetricsSummary:
        pass

    @abstractmethod
    async def query_detailed_timeline(self, period: str, interval: str) -> List[DetailedTimelineBucket]:
        pass

    @abstractmethod
    async def query_detailed_summary(self, period: str) -> DetailedMetricsSummary:
        pass

    @abstractmethod
    async def query_logs(
        self,
        limit: int,
        *,
        user_id: str = None,
        session_id: str = None,
        context_id: str = None,
        has_error: bool = None,
        keyword: str = None,
    ) -> List[ConversationGroup]:
        pass


def _group_logs(rows, time_origin: str) -> List[ConversationGroup]:
    logs = []
    for row in rows:
        created_at = row[0]
        if isinstance(created_at, datetime):
            created_at = created_at.strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(created_at, str):
            pass  # already string
        detailed_row = (
            row[0], row[14], row[15], row[16], row[17], row[18], row[19],
            row[6], row[20], row[21], row[5], row[7], row[12],
        )
        phases = _raw_phase_vector(detailed_row, time_origin)
        timing = TurnTimingBreakdown(
            total_first_response=sum(phases),
            silence_detection=phases[0],
            streaming_stt_finalization=phases[1],
            turn_end_gate=phases[2],
            stt=phases[3],
            stop_response=phases[4],
            before_llm=phases[5],
            llm=phases[6],
            processing=phases[7],
            tts=phases[8],
        ) if phases is not None else None
        logs.append(ConversationLog(
            created_at=created_at,
            transaction_id=row[1],
            user_id=row[2],
            session_id=row[3],
            context_id=row[4],
            tts_first_chunk_time=row[5],
            before_llm_time=row[6],
            quick_response_text=row[7],
            request_text=row[8],
            request_files=row[9],
            response_text=row[10],
            response_voice_text=row[11],
            error_info=row[12],
            tool_calls=row[13],
            timing_breakdown=timing,
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


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SQLiteMetricsQuery(MetricsQuery):
    def __init__(self, db_path: str, time_origin: str = TIME_ORIGIN_USER_SPEECH_END):
        self.db_path = db_path
        self.time_origin = time_origin

    def _fetch_rows(self, start_time: datetime):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        try:
            # Use datetime object directly for proper comparison
            rows = conn.execute(_QUERY_SQL, (start_time,)).fetchall()
            result = []
            for row in rows:
                created_at = row[0]
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                result.append((created_at, row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
            return result
        finally:
            conn.close()

    def _fetch_detailed_rows(self, start_time: datetime):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        try:
            rows = conn.execute(_DETAILED_QUERY_SQL, (start_time,)).fetchall()
            result = []
            for row in rows:
                created_at = row[0]
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                result.append((created_at, *row[1:]))
            return result
        finally:
            conn.close()

    def _fetch_logs(
        self,
        limit: int,
        user_id: str = None,
        session_id: str = None,
        context_id: str = None,
        has_error: bool = None,
        keyword: str = None,
    ):
        clauses = []
        params = []
        for column, value in (("user_id", user_id), ("session_id", session_id), ("context_id", context_id)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if has_error is True:
            clauses.append("error_info IS NOT NULL AND error_info <> ''")
        elif has_error is False:
            clauses.append("(error_info IS NULL OR error_info = '')")
        if keyword:
            searchable = (
                "request_text", "response_text", "response_voice_text",
                "quick_response_text", "error_info", "tool_calls",
            )
            clauses.append("(" + " OR ".join(f"COALESCE({column}, '') LIKE ? ESCAPE '\\'" for column in searchable) + ")")
            params.extend([f"%{_escape_like(keyword)}%"] * len(searchable))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"{_LOGS_SELECT_SQL}{where} ORDER BY event_at DESC LIMIT ?"
        params.append(limit)
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        try:
            return conn.execute(sql, params).fetchall()
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

    async def query_detailed_timeline(self, period: str, interval: str) -> List[DetailedTimelineBucket]:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}")
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await asyncio.to_thread(self._fetch_detailed_rows, start_time)
        return _build_detailed_timeline(rows, INTERVAL_SECONDS[interval], start_time, self.time_origin)

    async def query_detailed_summary(self, period: str) -> DetailedMetricsSummary:
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await asyncio.to_thread(self._fetch_detailed_rows, start_time)
        return _build_detailed_summary(rows, self.time_origin)

    async def query_logs(
        self,
        limit: int,
        *,
        user_id: str = None,
        session_id: str = None,
        context_id: str = None,
        has_error: bool = None,
        keyword: str = None,
    ) -> List[ConversationGroup]:
        rows = await asyncio.to_thread(
            self._fetch_logs, limit, user_id, session_id, context_id, has_error, keyword
        )
        return _group_logs(rows, self.time_origin)


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
        time_origin: str = TIME_ORIGIN_USER_SPEECH_END,
    ):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection_str = connection_str
        self.time_origin = time_origin
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
        SELECT COALESCE(speech_end_at, created_at) AS event_at,
               stt_time, before_llm_time, llm_first_chunk_time, llm_first_voice_chunk_time, tts_first_chunk_time, quick_response_text, error_info
        FROM performance_records
        WHERE COALESCE(speech_end_at, created_at) >= $1
        ORDER BY event_at
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, start_time)
        return [(r["event_at"], r["stt_time"], r["before_llm_time"], r["llm_first_chunk_time"], r["llm_first_voice_chunk_time"], r["tts_first_chunk_time"], r["quick_response_text"], r["error_info"]) for r in records]

    async def _fetch_logs(self, limit: int):
        return await self._fetch_filtered_logs(limit)

    async def _fetch_detailed_rows(self, start_time: datetime):
        pool = await self._get_pool()
        query = """
        SELECT COALESCE(speech_end_at, created_at) AS event_at,
               speech_end_at, silence_threshold_time, stt_after_threshold_time, turn_end_gate_time,
               stt_time, stop_response_time, before_llm_time, llm_first_chunk_time, llm_first_voice_chunk_time,
               tts_first_chunk_time, quick_response_text, error_info
        FROM performance_records
        WHERE COALESCE(speech_end_at, created_at) >= $1
        ORDER BY event_at
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, start_time)
        columns = (
            "event_at", "speech_end_at", "silence_threshold_time", "stt_after_threshold_time",
            "turn_end_gate_time", "stt_time", "stop_response_time", "before_llm_time",
            "llm_first_chunk_time", "llm_first_voice_chunk_time", "tts_first_chunk_time",
            "quick_response_text", "error_info",
        )
        return [tuple(record[column] for column in columns) for record in records]

    async def _fetch_filtered_logs(
        self,
        limit: int,
        user_id: str = None,
        session_id: str = None,
        context_id: str = None,
        has_error: bool = None,
        keyword: str = None,
    ):
        clauses = []
        params = []

        def bind(value):
            params.append(value)
            return f"${len(params)}"

        for column, value in (("user_id", user_id), ("session_id", session_id), ("context_id", context_id)):
            if value:
                clauses.append(f"{column} = {bind(value)}")
        if has_error is True:
            clauses.append("error_info IS NOT NULL AND error_info <> ''")
        elif has_error is False:
            clauses.append("(error_info IS NULL OR error_info = '')")
        if keyword:
            marker = bind(f"%{_escape_like(keyword)}%")
            searchable = (
                "request_text", "response_text", "response_voice_text",
                "quick_response_text", "error_info", "tool_calls",
            )
            clauses.append("(" + " OR ".join(
                f"COALESCE({column}, '') ILIKE {marker} ESCAPE '\\'" for column in searchable
            ) + ")")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_marker = bind(limit)
        query = f"{_LOGS_SELECT_SQL}{where} ORDER BY event_at DESC LIMIT {limit_marker}"
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            records = await conn.fetch(query, *params)
        columns = (
            "event_at", "transaction_id", "user_id", "session_id", "context_id",
            "tts_first_chunk_time", "before_llm_time", "quick_response_text", "request_text",
            "request_files", "response_text", "response_voice_text", "error_info", "tool_calls",
            "speech_end_at", "silence_threshold_time", "stt_after_threshold_time", "turn_end_gate_time",
            "stt_time", "stop_response_time", "llm_first_chunk_time", "llm_first_voice_chunk_time",
        )
        return [tuple(record[column] for column in columns) for record in records]

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

    async def query_detailed_timeline(self, period: str, interval: str) -> List[DetailedTimelineBucket]:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}")
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await self._fetch_detailed_rows(start_time)
        return _build_detailed_timeline(rows, INTERVAL_SECONDS[interval], start_time, self.time_origin)

    async def query_detailed_summary(self, period: str) -> DetailedMetricsSummary:
        start_time = datetime.now(timezone.utc) - parse_period(period)
        rows = await self._fetch_detailed_rows(start_time)
        return _build_detailed_summary(rows, self.time_origin)

    async def query_logs(
        self,
        limit: int,
        *,
        user_id: str = None,
        session_id: str = None,
        context_id: str = None,
        has_error: bool = None,
        keyword: str = None,
    ) -> List[ConversationGroup]:
        rows = await self._fetch_filtered_logs(
            limit, user_id, session_id, context_id, has_error, keyword
        )
        return _group_logs(rows, self.time_origin)


def create_metrics_query(recorder) -> MetricsQuery:
    from .sqlite import SQLitePerformanceRecorder
    if isinstance(recorder, SQLitePerformanceRecorder):
        return SQLiteMetricsQuery(recorder.db_path, time_origin=recorder.time_origin)

    from .postgres import PostgreSQLPerformanceRecorder
    if isinstance(recorder, PostgreSQLPerformanceRecorder):
        return PostgreSQLMetricsQuery(
            host=recorder.host,
            port=recorder.port,
            dbname=recorder.dbname,
            user=recorder.user,
            password=recorder.password,
            connection_str=recorder.connection_str,
            time_origin=recorder.time_origin,
        )

    raise ValueError(f"Unsupported recorder type: {type(recorder)}")
