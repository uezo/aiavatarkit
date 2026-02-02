import logging
from calendar import monthrange
from datetime import date
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from ..character.service import CharacterService
from .auth import create_api_key_dependency

logger = logging.getLogger(__name__)


# --- Character / User / WeeklySchedule models ---

class CharacterResponse(BaseModel):
    id: str
    name: str
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


class CharacterUpdateRequest(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WeeklyScheduleResponse(BaseModel):
    content: Optional[str] = None
    exists: bool


class WeeklyScheduleUpdateRequest(BaseModel):
    content: str


class UserResponse(BaseModel):
    id: str
    name: str
    metadata: Optional[Dict[str, Any]] = None


class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UserListItem(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str


class UserListResponse(BaseModel):
    users: List[UserListItem]
    total: int


# --- Activity models ---

class DayActivity(BaseModel):
    date: str
    has_schedule: bool
    has_diary: bool
    schedule_content: Optional[str] = None
    diary_content: Optional[str] = None


class ActivitiesResponse(BaseModel):
    year: int
    month: int
    character_id: str
    days: List[DayActivity]


class ScheduleDetailResponse(BaseModel):
    date: str
    content: Optional[str] = None
    content_context: Optional[Dict[str, str]] = None
    exists: bool


class DiaryDetailResponse(BaseModel):
    date: str
    content: Optional[str] = None
    content_context: Optional[Dict[str, str]] = None
    exists: bool


class GenerateRequest(BaseModel):
    overwrite: bool = False


class GenerateRangeRequest(BaseModel):
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    overwrite: bool = False


class GenerateResponse(BaseModel):
    date: str
    content: str
    generated: bool


class GenerateRangeResponse(BaseModel):
    results: List[Dict]


class CharacterAPI:
    def __init__(self, character_service: CharacterService, character_id: str):
        self.service = character_service
        self.character_id = character_id

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.get(
            "/character/activities",
            response_model=ActivitiesResponse,
            tags=["Character"],
            summary="Get monthly activities for calendar",
            description="Returns daily schedules and diaries for a given month.",
        )
        async def get_activities(
            year: int = Query(..., description="Year"),
            month: int = Query(..., description="Month (1-12)", ge=1, le=12),
        ) -> ActivitiesResponse:
            try:
                _, last_day = monthrange(year, month)
                start = date(year, month, 1)
                end = date(year, month, last_day)

                schedules = await self.service.activity.list_daily_schedules(
                    character_id=self.character_id,
                    start_date=start,
                    end_date=end,
                )
                diaries = await self.service.activity.list_diaries(
                    character_id=self.character_id,
                    start_date=start,
                    end_date=end,
                )

                schedule_map = {s.schedule_date.isoformat(): s for s in schedules}
                diary_map = {d.diary_date.isoformat(): d for d in diaries}

                days = []
                for day_num in range(1, last_day + 1):
                    d = date(year, month, day_num)
                    d_str = d.isoformat()
                    s = schedule_map.get(d_str)
                    di = diary_map.get(d_str)
                    days.append(DayActivity(
                        date=d_str,
                        has_schedule=s is not None,
                        has_diary=di is not None,
                        schedule_content=s.content if s else None,
                        diary_content=di.content if di else None,
                    ))

                return ActivitiesResponse(
                    year=year,
                    month=month,
                    character_id=self.character_id,
                    days=days,
                )
            except Exception as ex:
                logger.error(f"Error querying activities: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while querying activities",
                )

        @router.get(
            "/character/schedule/{target_date}",
            response_model=ScheduleDetailResponse,
            tags=["Character"],
            summary="Get daily schedule detail",
        )
        async def get_schedule(target_date: str) -> ScheduleDetailResponse:
            try:
                d = date.fromisoformat(target_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

            try:
                schedule = await self.service.activity.get_daily_schedule(
                    character_id=self.character_id,
                    schedule_date=d,
                )
                if schedule:
                    return ScheduleDetailResponse(
                        date=target_date,
                        content=schedule.content,
                        content_context=schedule.content_context,
                        exists=True,
                    )
                return ScheduleDetailResponse(date=target_date, exists=False)
            except Exception as ex:
                logger.error(f"Error querying schedule: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                )

        @router.get(
            "/character/diary/{target_date}",
            response_model=DiaryDetailResponse,
            tags=["Character"],
            summary="Get diary detail",
        )
        async def get_diary(target_date: str) -> DiaryDetailResponse:
            try:
                d = date.fromisoformat(target_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

            try:
                diary = await self.service.activity.get_diary(
                    character_id=self.character_id,
                    diary_date=d,
                )
                if diary:
                    return DiaryDetailResponse(
                        date=target_date,
                        content=diary.content,
                        content_context=diary.content_context,
                        exists=True,
                    )
                return DiaryDetailResponse(date=target_date, exists=False)
            except Exception as ex:
                logger.error(f"Error querying diary: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error",
                )

        @router.post(
            "/character/schedule/{target_date}/generate",
            response_model=GenerateResponse,
            tags=["Character"],
            summary="Generate daily schedule",
        )
        async def generate_schedule(
            target_date: str,
            request: GenerateRequest = None,
        ) -> GenerateResponse:
            try:
                d = date.fromisoformat(target_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

            req = request or GenerateRequest()

            try:
                existing = await self.service.activity.get_daily_schedule(
                    character_id=self.character_id,
                    schedule_date=d,
                )
                if existing and not req.overwrite:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Schedule already exists for {target_date}. Set overwrite=true to regenerate.",
                    )
                if existing and req.overwrite:
                    await self.service.activity.delete_daily_schedule(
                        character_id=self.character_id,
                        schedule_date=d,
                    )

                schedule = await self.service.create_daily_schedule_with_generation(
                    character_id=self.character_id,
                    schedule_date=d,
                )
                return GenerateResponse(
                    date=target_date,
                    content=schedule.content,
                    generated=True,
                )
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error generating schedule: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error generating schedule: {str(ex)}",
                )

        @router.post(
            "/character/diary/{target_date}/generate",
            response_model=GenerateResponse,
            tags=["Character"],
            summary="Generate diary",
        )
        async def generate_diary(
            target_date: str,
            request: GenerateRequest = None,
        ) -> GenerateResponse:
            try:
                d = date.fromisoformat(target_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

            req = request or GenerateRequest()

            try:
                existing = await self.service.activity.get_diary(
                    character_id=self.character_id,
                    diary_date=d,
                )
                if existing and not req.overwrite:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Diary already exists for {target_date}. Set overwrite=true to regenerate.",
                    )
                if existing and req.overwrite:
                    await self.service.activity.delete_diary(
                        character_id=self.character_id,
                        diary_date=d,
                    )

                diary = await self.service.create_diary_with_generation(
                    character_id=self.character_id,
                    diary_date=d,
                )
                return GenerateResponse(
                    date=target_date,
                    content=diary.content,
                    generated=True,
                )
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error generating diary: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error generating diary: {str(ex)}",
                )

        @router.post(
            "/character/activities/generate",
            response_model=GenerateRangeResponse,
            tags=["Character"],
            summary="Batch generate schedules and diaries for a date range",
        )
        async def generate_range(
            request: GenerateRangeRequest,
        ) -> GenerateRangeResponse:
            try:
                start = date.fromisoformat(request.start_date)
                end = date.fromisoformat(request.end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

            if start > end:
                raise HTTPException(status_code=400, detail="start_date must be <= end_date")

            if (end - start).days > 31:
                raise HTTPException(status_code=400, detail="Date range must be 31 days or less")

            try:
                activity_results = await self.service.create_activity_range_with_generation(
                    character_id=self.character_id,
                    start_date=start,
                    end_date=end,
                    overwrite=request.overwrite,
                )
                results = []
                for r in activity_results:
                    results.append({
                        "date": r.target_date.isoformat(),
                        "is_schedule_generated": r.is_schedule_generated,
                        "is_diary_generated": r.is_diary_generated,
                    })
                return GenerateRangeResponse(results=results)
            except Exception as ex:
                logger.error(f"Error generating activity range: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error generating activities: {str(ex)}",
                )

        # --- Character info ---

        @router.get(
            "/character/info",
            response_model=CharacterResponse,
            tags=["Character"],
            summary="Get character info",
        )
        async def get_character_info() -> CharacterResponse:
            try:
                char = await self.service.character.get(character_id=self.character_id)
                if not char:
                    raise HTTPException(status_code=404, detail="Character not found")
                return CharacterResponse(
                    id=char.id,
                    name=char.name,
                    prompt=char.prompt,
                    metadata=char.metadata,
                )
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error getting character info: {ex}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @router.post(
            "/character/info",
            response_model=CharacterResponse,
            tags=["Character"],
            summary="Update character info",
        )
        async def update_character_info(request: CharacterUpdateRequest) -> CharacterResponse:
            try:
                updated = await self.service.character.update(
                    character_id=self.character_id,
                    name=request.name,
                    prompt=request.prompt,
                    metadata=request.metadata,
                )
                if not updated:
                    raise HTTPException(status_code=404, detail="Character not found")
                return CharacterResponse(
                    id=updated.id,
                    name=updated.name,
                    prompt=updated.prompt,
                    metadata=updated.metadata,
                )
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error updating character info: {ex}")
                raise HTTPException(status_code=500, detail="Internal server error")

        # --- Weekly Schedule ---

        @router.get(
            "/character/weekly-schedule",
            response_model=WeeklyScheduleResponse,
            tags=["Character"],
            summary="Get weekly schedule",
        )
        async def get_weekly_schedule() -> WeeklyScheduleResponse:
            try:
                ws = await self.service.activity.get_weekly_schedule(
                    character_id=self.character_id,
                )
                if ws:
                    return WeeklyScheduleResponse(content=ws.content, exists=True)
                return WeeklyScheduleResponse(exists=False)
            except Exception as ex:
                logger.error(f"Error getting weekly schedule: {ex}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @router.post(
            "/character/weekly-schedule",
            response_model=WeeklyScheduleResponse,
            tags=["Character"],
            summary="Update weekly schedule",
        )
        async def update_weekly_schedule(request: WeeklyScheduleUpdateRequest) -> WeeklyScheduleResponse:
            try:
                existing = await self.service.activity.get_weekly_schedule(
                    character_id=self.character_id,
                )
                if existing:
                    ws = await self.service.activity.update_weekly_schedule(
                        character_id=self.character_id,
                        content=request.content,
                    )
                else:
                    ws = await self.service.activity.create_weekly_schedule(
                        character_id=self.character_id,
                        content=request.content,
                    )
                return WeeklyScheduleResponse(content=ws.content, exists=True)
            except Exception as ex:
                logger.error(f"Error updating weekly schedule: {ex}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @router.post(
            "/character/weekly-schedule/generate",
            response_model=WeeklyScheduleResponse,
            tags=["Character"],
            summary="Generate weekly schedule",
        )
        async def generate_weekly_schedule() -> WeeklyScheduleResponse:
            try:
                existing = await self.service.activity.get_weekly_schedule(
                    character_id=self.character_id,
                )
                if existing:
                    await self.service.activity.delete_weekly_schedule(
                        character_id=self.character_id,
                    )
                ws = await self.service.create_weekly_schedule_with_generation(
                    character_id=self.character_id,
                )
                return WeeklyScheduleResponse(content=ws.content, exists=True)
            except Exception as ex:
                logger.error(f"Error generating weekly schedule: {ex}")
                raise HTTPException(status_code=500, detail=f"Error generating weekly schedule: {str(ex)}")

        # --- User ---

        @router.get(
            "/character/users",
            response_model=UserListResponse,
            tags=["Character"],
            summary="List users",
        )
        async def list_users(
            limit: int = Query(100, ge=1, le=500),
            offset: int = Query(0, ge=0),
        ) -> UserListResponse:
            try:
                users = await self.service.user.list(limit=limit, offset=offset)
                items = [
                    UserListItem(
                        id=u.id,
                        name=u.name,
                        created_at=u.created_at.isoformat(),
                        updated_at=u.updated_at.isoformat(),
                    )
                    for u in users
                ]
                return UserListResponse(users=items, total=len(items))
            except Exception as ex:
                logger.error(f"Error listing users: {ex}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @router.get(
            "/character/user/{user_id}",
            response_model=UserResponse,
            tags=["Character"],
            summary="Get user info",
        )
        async def get_user(user_id: str) -> UserResponse:
            try:
                user = await self.service.user.get(user_id=user_id)
                if not user:
                    raise HTTPException(status_code=404, detail="User not found")
                return UserResponse(
                    id=user.id,
                    name=user.name,
                    metadata=user.metadata,
                )
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error getting user: {ex}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @router.post(
            "/character/user/{user_id}",
            response_model=UserResponse,
            tags=["Character"],
            summary="Update user info",
        )
        async def update_user(user_id: str, request: UserUpdateRequest) -> UserResponse:
            try:
                updated = await self.service.user.update(
                    user_id=user_id,
                    name=request.name,
                    metadata=request.metadata,
                )
                if not updated:
                    raise HTTPException(status_code=404, detail="User not found")
                return UserResponse(
                    id=updated.id,
                    name=updated.name,
                    metadata=updated.metadata,
                )
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error updating user: {ex}")
                raise HTTPException(status_code=500, detail="Internal server error")

        return router


def setup_character_api(
    app: FastAPI,
    *,
    character_service: CharacterService,
    character_id: str,
    api_key: str = None,
):
    deps = [Depends(create_api_key_dependency(api_key))] if api_key else []
    app.include_router(
        CharacterAPI(character_service=character_service, character_id=character_id).get_router(),
        dependencies=deps,
    )
