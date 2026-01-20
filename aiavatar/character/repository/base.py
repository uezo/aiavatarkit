from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, Any, Optional
from ..models import Character, WeeklySchedule, DailySchedule, Diary, User


class CharacterRepositoryBase(ABC):
    @abstractmethod
    async def create(
        self,
        *,
        name: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Character:
        pass

    @abstractmethod
    async def get(self, *, character_id: str) -> Optional[Character]:
        pass

    @abstractmethod
    async def update(
        self,
        *,
        character_id: str,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Character]:
        pass

    @abstractmethod
    async def delete(self, *, character_id: str) -> bool:
        pass


class ActivityRepositoryBase(ABC):
    # WeeklySchedule operations

    @abstractmethod
    async def create_weekly_schedule(
        self,
        *,
        character_id: str,
        content: str
    ) -> WeeklySchedule:
        pass

    @abstractmethod
    async def get_weekly_schedule(self, *, character_id: str) -> Optional[WeeklySchedule]:
        pass

    @abstractmethod
    async def update_weekly_schedule(
        self,
        *,
        character_id: str,
        content: str
    ) -> Optional[WeeklySchedule]:
        pass

    @abstractmethod
    async def delete_weekly_schedule(self, *, character_id: str) -> bool:
        pass

    # DailySchedule operations

    @abstractmethod
    async def create_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date,
        content: str,
        content_context: Optional[Dict[str, str]] = None
    ) -> DailySchedule:
        pass

    @abstractmethod
    async def get_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date
    ) -> Optional[DailySchedule]:
        pass

    @abstractmethod
    async def update_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date,
        content: str
    ) -> Optional[DailySchedule]:
        pass

    @abstractmethod
    async def delete_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date
    ) -> bool:
        pass

    # Diary operations

    @abstractmethod
    async def create_diary(
        self,
        *,
        character_id: str,
        diary_date: date,
        content: str,
        content_context: Optional[Dict[str, str]] = None
    ) -> Diary:
        pass

    @abstractmethod
    async def get_diary(
        self,
        *,
        character_id: str,
        diary_date: date
    ) -> Optional[Diary]:
        pass

    @abstractmethod
    async def update_diary(
        self,
        *,
        character_id: str,
        diary_date: date,
        content: Optional[str] = None,
        content_context: Optional[Dict[str, str]] = None
    ) -> Optional[Diary]:
        pass

    @abstractmethod
    async def delete_diary(
        self,
        *,
        character_id: str,
        diary_date: date
    ) -> bool:
        pass


class UserRepository(ABC):
    @abstractmethod
    async def create(
        self,
        *,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        pass

    @abstractmethod
    async def get(self, *, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    async def update(
        self,
        *,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[User]:
        pass

    @abstractmethod
    async def delete(self, *, user_id: str) -> bool:
        pass
