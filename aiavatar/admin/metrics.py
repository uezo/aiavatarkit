import logging
from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from ..sts.performance_recorder.base import PerformanceRecorder
from ..sts.performance_recorder.query import create_metrics_query

logger = logging.getLogger(__name__)


class DetailedMetrics(BaseModel):
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


class TimelineBucketResponse(DetailedMetrics):
    timestamp: str
    request_count: int


class TimelineResponse(BaseModel):
    period: str
    interval: str
    buckets: List[TimelineBucketResponse]


class SummaryResponse(DetailedMetrics):
    period: str
    total_requests: int
    p50_first_response_time: Optional[float] = None
    p95_first_response_time: Optional[float] = None
    p99_first_response_time: Optional[float] = None


class MetricsAPI:
    def __init__(self, recorder: PerformanceRecorder):
        self.query = create_metrics_query(recorder)

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/metrics/timeline", response_model=TimelineResponse, tags=["Admin Metrics"])
        async def get_timeline(
            period: str = Query("24h"),
            interval: str = Query("1h"),
        ) -> TimelineResponse:
            try:
                buckets = await self.query.query_detailed_timeline(period, interval)
                return TimelineResponse(
                    period=period,
                    interval=interval,
                    buckets=[TimelineBucketResponse(**asdict(bucket)) for bucket in buckets],
                )
            except ValueError as ex:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))
            except Exception:
                logger.exception("Error querying detailed timeline metrics")
                raise HTTPException(status_code=500, detail="Could not query metrics")

        @router.get("/metrics/summary", response_model=SummaryResponse, tags=["Admin Metrics"])
        async def get_summary(period: str = Query("24h")) -> SummaryResponse:
            try:
                summary = await self.query.query_detailed_summary(period)
                return SummaryResponse(period=period, **asdict(summary))
            except ValueError as ex:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))
            except Exception:
                logger.exception("Error querying detailed metrics summary")
                raise HTTPException(status_code=500, detail="Could not query metrics")

        return router
