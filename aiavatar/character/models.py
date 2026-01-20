from datetime import datetime, date
from typing import Dict, Any, Optional
from pydantic import BaseModel


class Character(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    name: str
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


class User(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    name: str
    metadata: Optional[Dict[str, Any]] = None


class WeeklySchedule(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    character_id: str
    content: str


class DailySchedule(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    character_id: str
    schedule_date: date
    content: str
    content_context: Optional[Dict[str, str]] = None


class Diary(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    character_id: str
    diary_date: date
    content: str
    content_context: Dict[str, str]


class MemorySearchResult(BaseModel):
    answer: Optional[str]
    retrieved_data: Optional[str]


class ActivityRangeResult(BaseModel):
    target_date: date
    daily_schedule: DailySchedule
    diary: Diary
    is_schedule_generated: bool
    is_diary_generated: bool
