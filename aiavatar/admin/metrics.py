import logging
from dataclasses import asdict
from typing import List, Optional
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from ..sts.performance_recorder.base import PerformanceRecorder
from ..sts.performance_recorder.query import create_metrics_query, MetricsQuery
from .auth import create_api_key_dependency

logger = logging.getLogger(__name__)


class TimelineBucketResponse(BaseModel):
    timestamp: str
    request_count: int
    success_count: int = 0
    error_count: int = 0
    avg_tts_first_chunk_time: Optional[float] = None
    p50_tts_first_chunk_time: Optional[float] = None
    p95_tts_first_chunk_time: Optional[float] = None
    avg_stt_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_tts_phase: Optional[float] = None


class TimelineResponse(BaseModel):
    period: str
    interval: str
    buckets: List[TimelineBucketResponse]


class SummaryResponse(BaseModel):
    period: str
    total_requests: int
    success_count: int = 0
    error_count: int = 0
    avg_tts_first_chunk_time: Optional[float] = None
    p50_tts_first_chunk_time: Optional[float] = None
    p95_tts_first_chunk_time: Optional[float] = None
    p99_tts_first_chunk_time: Optional[float] = None
    avg_stt_phase: Optional[float] = None
    avg_llm_phase: Optional[float] = None
    avg_tts_phase: Optional[float] = None


class MetricsAPI:
    def __init__(self, recorder: PerformanceRecorder):
        self.query = create_metrics_query(recorder)

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.get(
            "/metrics/timeline",
            response_model=TimelineResponse,
            tags=["Metrics"],
            summary="Get time-series metrics",
            description="Returns time-bucketed metrics for request volume and response time breakdown.",
            responses={
                200: {"description": "Successfully returns timeline metrics"},
                400: {"description": "Invalid period or interval parameter"},
                500: {"description": "Internal server error"},
            },
        )
        async def get_timeline(
            period: str = Query("24h", description="Time period: 1h, 6h, 24h, 7d, 30d"),
            interval: str = Query("1h", description="Bucket interval: 1m, 5m, 15m, 1h, 1d"),
        ) -> TimelineResponse:
            try:
                buckets = await self.query.query_timeline(period, interval)
                return TimelineResponse(
                    period=period,
                    interval=interval,
                    buckets=[TimelineBucketResponse(**asdict(b)) for b in buckets],
                )
            except ValueError as ex:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))
            except Exception as ex:
                logger.error(f"Error querying timeline metrics: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while querying timeline metrics",
                )

        @router.get(
            "/metrics/summary",
            response_model=SummaryResponse,
            tags=["Metrics"],
            summary="Get aggregate metrics summary",
            description="Returns aggregate statistics (avg, p50, p95, p99) for a given time period.",
            responses={
                200: {"description": "Successfully returns metrics summary"},
                400: {"description": "Invalid period parameter"},
                500: {"description": "Internal server error"},
            },
        )
        async def get_summary(
            period: str = Query("24h", description="Time period: 1h, 6h, 24h, 7d, 30d"),
        ) -> SummaryResponse:
            try:
                summary = await self.query.query_summary(period)
                return SummaryResponse(period=period, **asdict(summary))
            except ValueError as ex:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ex))
            except Exception as ex:
                logger.error(f"Error querying metrics summary: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while querying metrics summary",
                )

        return router


def setup_metrics_api(
    app: FastAPI,
    *,
    recorder: PerformanceRecorder,
    api_key: str = None,
):
    deps = [Depends(create_api_key_dependency(api_key))] if api_key else []
    app.include_router(MetricsAPI(recorder=recorder).get_router(), dependencies=deps)
