import json
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional
from uuid import uuid4
import asyncpg
from .models import WeeklySchedule, DailySchedule, Diary


class ActivityRepository:
    WEEKLY_SCHEDULES_TABLE = "weekly_schedules"
    DAILY_SCHEDULES_TABLE = "daily_schedules"
    DIARIES_TABLE = "diaries"

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def init_tables(self):
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.WEEKLY_SCHEDULES_TABLE} (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    character_id TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.DAILY_SCHEDULES_TABLE} (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    character_id TEXT NOT NULL,
                    schedule_date DATE NOT NULL,
                    content TEXT NOT NULL,
                    content_context JSONB NOT NULL DEFAULT '{{}}',
                    UNIQUE (character_id, schedule_date)
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.DIARIES_TABLE} (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    character_id TEXT NOT NULL,
                    diary_date DATE NOT NULL,
                    content TEXT NOT NULL,
                    content_context JSONB NOT NULL DEFAULT '{{}}',
                    UNIQUE (character_id, diary_date)
                )
                """
            )

    # WeeklySchedule operations

    async def create_weekly_schedule(
        self,
        *,
        character_id: str,
        content: str
    ) -> WeeklySchedule:
        now = datetime.now(timezone.utc)
        schedule_id = str(uuid4())

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.WEEKLY_SCHEDULES_TABLE} (id, created_at, updated_at, character_id, content)
                VALUES ($1, $2, $3, $4, $5)
                """,
                schedule_id, now, now, character_id, content
            )

        return WeeklySchedule(
            id=schedule_id,
            created_at=now,
            updated_at=now,
            character_id=character_id,
            content=content
        )

    async def get_weekly_schedule(self, *, character_id: str) -> Optional[WeeklySchedule]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, created_at, updated_at, character_id, content
                FROM {self.WEEKLY_SCHEDULES_TABLE}
                WHERE character_id = $1
                """,
                character_id
            )

        if row is None:
            return None

        return WeeklySchedule(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            character_id=row["character_id"],
            content=row["content"]
        )

    async def update_weekly_schedule(
        self,
        *,
        character_id: str,
        content: str
    ) -> Optional[WeeklySchedule]:
        now = datetime.now(timezone.utc)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {self.WEEKLY_SCHEDULES_TABLE}
                SET content = $1, updated_at = $2
                WHERE character_id = $3
                RETURNING id, created_at, updated_at, character_id, content
                """,
                content, now, character_id
            )

        if row is None:
            return None

        return WeeklySchedule(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            character_id=row["character_id"],
            content=row["content"]
        )

    async def delete_weekly_schedule(self, *, character_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.WEEKLY_SCHEDULES_TABLE}
                WHERE character_id = $1
                """,
                character_id
            )

        return result == "DELETE 1"

    # DailySchedule operations

    async def create_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date,
        content: str,
        content_context: Optional[Dict[str, str]] = None
    ) -> DailySchedule:
        now = datetime.now(timezone.utc)
        schedule_id = str(uuid4())
        content_context_json = json.dumps(content_context or {})

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.DAILY_SCHEDULES_TABLE} (id, created_at, updated_at, character_id, schedule_date, content, content_context)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                schedule_id, now, now, character_id, schedule_date, content, content_context_json
            )

        return DailySchedule(
            id=schedule_id,
            created_at=now,
            updated_at=now,
            character_id=character_id,
            schedule_date=schedule_date,
            content=content,
            content_context=content_context
        )

    async def get_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date
    ) -> Optional[DailySchedule]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, created_at, updated_at, character_id, schedule_date, content, content_context
                FROM {self.DAILY_SCHEDULES_TABLE}
                WHERE character_id = $1 AND schedule_date = $2
                """,
                character_id, schedule_date
            )

        if row is None:
            return None

        content_context = row["content_context"]
        if isinstance(content_context, str):
            content_context = json.loads(content_context)

        return DailySchedule(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            character_id=row["character_id"],
            schedule_date=row["schedule_date"],
            content=row["content"],
            content_context=content_context
        )

    async def update_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date,
        content: str
    ) -> Optional[DailySchedule]:
        now = datetime.now(timezone.utc)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {self.DAILY_SCHEDULES_TABLE}
                SET content = $1, updated_at = $2
                WHERE character_id = $3 AND schedule_date = $4
                RETURNING id, created_at, updated_at, character_id, schedule_date, content, content_context
                """,
                content, now, character_id, schedule_date
            )

        if row is None:
            return None

        content_context = row["content_context"]
        if isinstance(content_context, str):
            content_context = json.loads(content_context)

        return DailySchedule(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            character_id=row["character_id"],
            schedule_date=row["schedule_date"],
            content=row["content"],
            content_context=content_context
        )

    async def delete_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.DAILY_SCHEDULES_TABLE}
                WHERE character_id = $1 AND schedule_date = $2
                """,
                character_id, schedule_date
            )

        return result == "DELETE 1"

    # Diary operations

    async def create_diary(
        self,
        *,
        character_id: str,
        diary_date: date,
        content: str,
        content_context: Optional[Dict[str, str]] = None
    ) -> Diary:
        now = datetime.now(timezone.utc)
        diary_id = str(uuid4())
        content_context_json = json.dumps(content_context or {})

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.DIARIES_TABLE} (id, created_at, updated_at, character_id, diary_date, content, content_context)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                diary_id, now, now, character_id, diary_date, content, content_context_json
            )

        return Diary(
            id=diary_id,
            created_at=now,
            updated_at=now,
            character_id=character_id,
            diary_date=diary_date,
            content=content,
            content_context=content_context or {}
        )

    async def get_diary(
        self,
        *,
        character_id: str,
        diary_date: date
    ) -> Optional[Diary]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, created_at, updated_at, character_id, diary_date, content, content_context
                FROM {self.DIARIES_TABLE}
                WHERE character_id = $1 AND diary_date = $2
                """,
                character_id, diary_date
            )

        if row is None:
            return None

        content_context = row["content_context"]
        if isinstance(content_context, str):
            content_context = json.loads(content_context)

        return Diary(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            character_id=row["character_id"],
            diary_date=row["diary_date"],
            content=row["content"],
            content_context=content_context
        )

    async def update_diary(
        self,
        *,
        character_id: str,
        diary_date: date,
        content: Optional[str] = None,
        content_context: Optional[Dict[str, str]] = None
    ) -> Optional[Diary]:
        now = datetime.now(timezone.utc)

        updates = ["updated_at = $1"]
        params: List[Any] = [now]
        param_idx = 2

        if content is not None:
            updates.append(f"content = ${param_idx}")
            params.append(content)
            param_idx += 1

        if content_context is not None:
            updates.append(f"content_context = ${param_idx}")
            params.append(json.dumps(content_context))
            param_idx += 1

        params.append(character_id)
        params.append(diary_date)

        query = f"""
            UPDATE {self.DIARIES_TABLE}
            SET {", ".join(updates)}
            WHERE character_id = ${param_idx} AND diary_date = ${param_idx + 1}
            RETURNING id, created_at, updated_at, character_id, diary_date, content, content_context
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row is None:
            return None

        row_content_context = row["content_context"]
        if isinstance(row_content_context, str):
            row_content_context = json.loads(row_content_context)

        return Diary(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            character_id=row["character_id"],
            diary_date=row["diary_date"],
            content=row["content"],
            content_context=row_content_context
        )

    async def delete_diary(
        self,
        *,
        character_id: str,
        diary_date: date
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.DIARIES_TABLE}
                WHERE character_id = $1 AND diary_date = $2
                """,
                character_id, diary_date
            )

        return result == "DELETE 1"
