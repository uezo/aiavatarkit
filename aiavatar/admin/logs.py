import logging
import re
from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from ..sts.performance_recorder.base import PerformanceRecorder
from ..sts.performance_recorder.query import create_metrics_query
from ..sts.voice_recorder import VoiceRecorder

logger = logging.getLogger(__name__)
_TRANSACTION_ID_PATTERN = re.compile(r"^[0-9a-f\-]+$", re.IGNORECASE)


class TurnTimingBreakdownResponse(BaseModel):
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


class ConversationLogResponse(BaseModel):
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
    timing_breakdown: Optional[TurnTimingBreakdownResponse] = None


class ConversationGroupResponse(BaseModel):
    context_id: Optional[str] = None
    logs: List[ConversationLogResponse]
    has_error: bool = False


class LogsResponse(BaseModel):
    limit: int
    voice_recorder_enabled: bool
    groups: List[ConversationGroupResponse]


class LogsAPI:
    def __init__(self, recorder: PerformanceRecorder, voice_recorder: VoiceRecorder = None):
        self.query = create_metrics_query(recorder)
        self.voice_recorder = voice_recorder

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/logs", response_model=LogsResponse, tags=["Admin Logs"])
        async def get_logs(
            limit: int = Query(200, ge=1, le=10000),
            user_id: Optional[str] = Query(None),
            session_id: Optional[str] = Query(None),
            context_id: Optional[str] = Query(None),
            has_error: Optional[bool] = Query(None),
            keyword: Optional[str] = Query(None, max_length=500),
        ) -> LogsResponse:
            try:
                groups = await self.query.query_logs(
                    limit,
                    user_id=user_id,
                    session_id=session_id,
                    context_id=context_id,
                    has_error=has_error,
                    keyword=keyword,
                )
                return LogsResponse(
                    limit=limit,
                    voice_recorder_enabled=self.voice_recorder is not None,
                    groups=[
                        ConversationGroupResponse(
                            context_id=group.context_id,
                            logs=[ConversationLogResponse(**asdict(log)) for log in group.logs],
                            has_error=group.has_error,
                        )
                        for group in groups
                    ],
                )
            except Exception:
                logger.exception("Error querying conversation logs")
                raise HTTPException(status_code=500, detail="Could not query logs")

        @router.get("/logs/voice/{transaction_id}/{voice_type}", tags=["Admin Logs"])
        async def get_voice(transaction_id: str, voice_type: str) -> Response:
            if self.voice_recorder is None:
                raise HTTPException(status_code=404, detail="Voice recording is not enabled")
            if not _TRANSACTION_ID_PATTERN.fullmatch(transaction_id):
                raise HTTPException(status_code=400, detail="Invalid transaction_id")
            try:
                if voice_type == "request":
                    data = await self.voice_recorder.get_request_voice(transaction_id)
                elif voice_type == "quick_response":
                    data = await self.voice_recorder.get_voice(f"{transaction_id}_qr_response_0")
                elif voice_type == "response":
                    voices = await self.voice_recorder.get_response_voices(transaction_id)
                    return {"count": len(voices)}
                elif voice_type.startswith("response_") and voice_type[9:].isdigit():
                    data = await self.voice_recorder.get_voice(f"{transaction_id}_{voice_type}")
                else:
                    raise HTTPException(status_code=400, detail="Invalid voice_type")
                if data is None:
                    raise HTTPException(status_code=404, detail="Voice file not found")
                return Response(content=data, media_type="audio/wav")
            except HTTPException:
                raise
            except Exception:
                logger.exception("Error retrieving voice")
                raise HTTPException(status_code=500, detail="Could not retrieve voice")

        return router
