import logging
import re
from dataclasses import asdict
from typing import List, Optional
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from ..sts.performance_recorder.base import PerformanceRecorder
from ..sts.performance_recorder.query import create_metrics_query, MetricsQuery
from ..sts.voice_recorder import VoiceRecorder
from .auth import create_api_key_dependency

logger = logging.getLogger(__name__)

_TRANSACTION_ID_PATTERN = re.compile(r"^[0-9a-f\-]+$", re.IGNORECASE)


class ConversationLogResponse(BaseModel):
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

        @router.get(
            "/logs",
            response_model=LogsResponse,
            tags=["Logs"],
            summary="Get conversation logs",
            description="Returns conversation logs grouped by context_id, sorted by created_at within each group.",
            responses={
                200: {"description": "Successfully returns conversation logs"},
                500: {"description": "Internal server error"},
            },
        )
        async def get_logs(
            limit: int = Query(200, description="Max number of records to fetch", ge=1, le=10000),
        ) -> LogsResponse:
            try:
                groups = await self.query.query_logs(limit)
                return LogsResponse(
                    limit=limit,
                    voice_recorder_enabled=self.voice_recorder is not None,
                    groups=[
                        ConversationGroupResponse(
                            context_id=g.context_id,
                            logs=[ConversationLogResponse(**asdict(l)) for l in g.logs],
                            has_error=g.has_error,
                        )
                        for g in groups
                    ],
                )
            except Exception as ex:
                logger.error(f"Error querying conversation logs: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while querying conversation logs",
                )

        @router.get(
            "/logs/voice/{transaction_id}/{voice_type}",
            tags=["Logs"],
            summary="Get recorded voice file",
            description="Returns a voice recording for the given transaction. Use 'request' for request voice, 'response' to get count of response voices, or 'response_N' for individual response voice.",
            responses={
                200: {"description": "Voice file"},
                400: {"description": "Invalid parameters"},
                404: {"description": "Voice file not found"},
                500: {"description": "Internal server error"},
            },
        )
        async def get_voice(
            transaction_id: str,
            voice_type: str,
        ) -> Response:
            if self.voice_recorder is None:
                raise HTTPException(status_code=404, detail="Voice recording is not enabled")

            if not _TRANSACTION_ID_PATTERN.match(transaction_id):
                raise HTTPException(status_code=400, detail="Invalid transaction_id")

            try:
                if voice_type == "request":
                    data = await self.voice_recorder.get_request_voice(transaction_id)
                    if data is None:
                        raise HTTPException(status_code=404, detail="Voice file not found")
                    return Response(content=data, media_type="audio/wav")

                elif voice_type == "response":
                    voices = await self.voice_recorder.get_response_voices(transaction_id)
                    return {"count": len(voices)}

                elif voice_type.startswith("response_"):
                    idx_str = voice_type[len("response_"):]
                    if not idx_str.isdigit():
                        raise HTTPException(status_code=400, detail="Invalid voice_type")
                    data = await self.voice_recorder.get_voice(f"{transaction_id}_response_{idx_str}")
                    if data is None:
                        raise HTTPException(status_code=404, detail="Voice file not found")
                    return Response(content=data, media_type="audio/wav")

                else:
                    raise HTTPException(status_code=400, detail="Invalid voice_type. Use 'request', 'response', or 'response_N'")

            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error retrieving voice: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while retrieving voice",
                )

        return router


def setup_logs_api(
    app: FastAPI,
    *,
    recorder: PerformanceRecorder,
    voice_recorder: VoiceRecorder = None,
    api_key: str = None,
):
    deps = [Depends(create_api_key_dependency(api_key))] if api_key else []
    app.include_router(
        LogsAPI(recorder=recorder, voice_recorder=voice_recorder).get_router(),
        dependencies=deps,
    )
