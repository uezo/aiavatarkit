import asyncio
import json
import logging
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional, Callable, Awaitable
from uuid import uuid4
import asyncpg
from .base import CharacterRepositoryBase, ActivityRepositoryBase, UserRepository
from ..models import Character, WeeklySchedule, DailySchedule, Diary, User

logger = logging.getLogger(__name__)


class PostgreSQLCharacterRepository(CharacterRepositoryBase):
    TABLE_NAME = "characters"

    def __init__(
        self,
        *,
        get_pool: Callable[[], Awaitable[asyncpg.Pool]] = None,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        connection_str: str = None,
        db_pool_min_size: int = 1,
        db_pool_max_size: int = 5
    ):
        self._get_pool_func = get_pool
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection_str = connection_str
        self.db_pool_min_size = db_pool_min_size
        self.db_pool_max_size = db_pool_max_size
        self._pool: asyncpg.Pool = None
        self._pool_lock = asyncio.Lock()
        self._db_initialized = False

    async def get_pool(self) -> asyncpg.Pool:
        # Use shared pool if provided
        if self._get_pool_func is not None:
            pool = await self._get_pool_func()
            if not self._db_initialized:
                async with self._pool_lock:
                    if not self._db_initialized:
                        await self.init_db(pool)
            return pool

        # Otherwise, create own pool (backward compatible)
        if self._pool is not None and self._db_initialized:
            return self._pool

        async with self._pool_lock:
            if self._pool is not None and self._db_initialized:
                return self._pool

            if self._pool is None:
                if self.connection_str:
                    self._pool = await asyncpg.create_pool(
                        dsn=self.connection_str,
                        min_size=self.db_pool_min_size,
                        max_size=self.db_pool_max_size,
                    )
                else:
                    self._pool = await asyncpg.create_pool(
                        host=self.host,
                        port=self.port,
                        database=self.dbname,
                        user=self.user,
                        password=self.password,
                        min_size=self.db_pool_min_size,
                        max_size=self.db_pool_max_size,
                    )

            await self.init_db(self._pool)

        return self._pool

    async def init_db(self, pool: asyncpg.Pool):
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        name TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        episode TEXT,
                        attribute TEXT,
                        conversation_example TEXT,
                        metadata JSONB NOT NULL DEFAULT '{{}}'
                    )
                    """
                )
                # Migration: add columns if not present
                for col in ("episode", "attribute", "conversation_example"):
                    try:
                        await conn.execute(f"ALTER TABLE {self.TABLE_NAME} ADD COLUMN {col} TEXT")
                    except asyncpg.exceptions.DuplicateColumnError:
                        pass

                self._db_initialized = True

            except Exception:
                logger.exception("Error at init_db")

    def _row_to_character(self, row) -> Character:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Character(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            name=row["name"],
            prompt=row["prompt"],
            episode=row["episode"],
            attribute=row["attribute"],
            conversation_example=row["conversation_example"],
            metadata=metadata
        )

    async def create(
        self,
        *,
        name: str,
        prompt: str,
        episode: Optional[str] = None,
        attribute: Optional[str] = None,
        conversation_example: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Character:
        now = datetime.now(timezone.utc)
        character_id = str(uuid4())
        metadata_json = json.dumps(metadata or {})

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (id, created_at, updated_at, name, prompt, episode, attribute, conversation_example, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                character_id, now, now, name, prompt, episode, attribute, conversation_example, metadata_json
            )

        return Character(
            id=character_id,
            created_at=now,
            updated_at=now,
            name=name,
            prompt=prompt,
            episode=episode,
            attribute=attribute,
            conversation_example=conversation_example,
            metadata=metadata
        )

    async def get(self, *, character_id: str) -> Optional[Character]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, created_at, updated_at, name, prompt, episode, attribute, conversation_example, metadata
                FROM {self.TABLE_NAME}
                WHERE id = $1
                """,
                character_id
            )

        if row is None:
            return None

        return self._row_to_character(row)

    async def update(
        self,
        *,
        character_id: str,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        episode: Optional[str] = None,
        attribute: Optional[str] = None,
        conversation_example: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Character]:
        now = datetime.now(timezone.utc)

        updates = ["updated_at = $1"]
        params: List[Any] = [now]
        param_idx = 2

        if name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1

        if prompt is not None:
            updates.append(f"prompt = ${param_idx}")
            params.append(prompt)
            param_idx += 1

        if episode is not None:
            updates.append(f"episode = ${param_idx}")
            params.append(episode)
            param_idx += 1

        if attribute is not None:
            updates.append(f"attribute = ${param_idx}")
            params.append(attribute)
            param_idx += 1

        if conversation_example is not None:
            updates.append(f"conversation_example = ${param_idx}")
            params.append(conversation_example)
            param_idx += 1

        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(json.dumps(metadata))
            param_idx += 1

        params.append(character_id)

        query = f"""
            UPDATE {self.TABLE_NAME}
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, created_at, updated_at, name, prompt, episode, attribute, conversation_example, metadata
        """

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row is None:
            return None

        return self._row_to_character(row)

    async def delete(self, *, character_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                WHERE id = $1
                """,
                character_id
            )

        return result == "DELETE 1"


class PostgreSQLActivityRepository(ActivityRepositoryBase):
    WEEKLY_SCHEDULES_TABLE = "weekly_schedules"
    DAILY_SCHEDULES_TABLE = "daily_schedules"
    DIARIES_TABLE = "diaries"

    def __init__(
        self,
        *,
        get_pool: Callable[[], Awaitable[asyncpg.Pool]] = None,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        connection_str: str = None,
        db_pool_min_size: int = 1,
        db_pool_max_size: int = 5
    ):
        self._get_pool_func = get_pool
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection_str = connection_str
        self.db_pool_min_size = db_pool_min_size
        self.db_pool_max_size = db_pool_max_size
        self._pool: asyncpg.Pool = None
        self._pool_lock = asyncio.Lock()
        self._db_initialized = False

    async def get_pool(self) -> asyncpg.Pool:
        # Use shared pool if provided
        if self._get_pool_func is not None:
            pool = await self._get_pool_func()
            if not self._db_initialized:
                async with self._pool_lock:
                    if not self._db_initialized:
                        await self.init_db(pool)
            return pool

        # Otherwise, create own pool (backward compatible)
        if self._pool is not None and self._db_initialized:
            return self._pool

        async with self._pool_lock:
            if self._pool is not None and self._db_initialized:
                return self._pool

            if self._pool is None:
                if self.connection_str:
                    self._pool = await asyncpg.create_pool(
                        dsn=self.connection_str,
                        min_size=self.db_pool_min_size,
                        max_size=self.db_pool_max_size,
                    )
                else:
                    self._pool = await asyncpg.create_pool(
                        host=self.host,
                        port=self.port,
                        database=self.dbname,
                        user=self.user,
                        password=self.password,
                        min_size=self.db_pool_min_size,
                        max_size=self.db_pool_max_size,
                    )

            await self.init_db(self._pool)

        return self._pool

    async def init_db(self, pool: asyncpg.Pool):
        async with pool.acquire() as conn:
            try:
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
                self._db_initialized = True

            except Exception:
                logger.exception("Error at init_db")

    # WeeklySchedule operations

    async def create_weekly_schedule(
        self,
        *,
        character_id: str,
        content: str
    ) -> WeeklySchedule:
        now = datetime.now(timezone.utc)
        schedule_id = str(uuid4())

        pool = await self.get_pool()
        async with pool.acquire() as conn:
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
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

        pool = await self.get_pool()
        async with pool.acquire() as conn:
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
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

        pool = await self.get_pool()
        async with pool.acquire() as conn:
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
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

        pool = await self.get_pool()
        async with pool.acquire() as conn:
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

    async def list_daily_schedules(
        self,
        *,
        character_id: str,
        start_date: date,
        end_date: date
    ) -> List[DailySchedule]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, created_at, updated_at, character_id, schedule_date, content, content_context
                FROM {self.DAILY_SCHEDULES_TABLE}
                WHERE character_id = $1 AND schedule_date BETWEEN $2 AND $3
                ORDER BY schedule_date
                """,
                character_id, start_date, end_date
            )

        result = []
        for row in rows:
            content_context = row["content_context"]
            if isinstance(content_context, str):
                content_context = json.loads(content_context)
            result.append(DailySchedule(
                id=row["id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                character_id=row["character_id"],
                schedule_date=row["schedule_date"],
                content=row["content"],
                content_context=content_context
            ))
        return result

    async def delete_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date
    ) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
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

        pool = await self.get_pool()
        async with pool.acquire() as conn:
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
        pool = await self.get_pool()
        async with pool.acquire() as conn:
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

        pool = await self.get_pool()
        async with pool.acquire() as conn:
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

    async def list_diaries(
        self,
        *,
        character_id: str,
        start_date: date,
        end_date: date
    ) -> List[Diary]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, created_at, updated_at, character_id, diary_date, content, content_context
                FROM {self.DIARIES_TABLE}
                WHERE character_id = $1 AND diary_date BETWEEN $2 AND $3
                ORDER BY diary_date
                """,
                character_id, start_date, end_date
            )

        result = []
        for row in rows:
            content_context = row["content_context"]
            if isinstance(content_context, str):
                content_context = json.loads(content_context)
            result.append(Diary(
                id=row["id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                character_id=row["character_id"],
                diary_date=row["diary_date"],
                content=row["content"],
                content_context=content_context
            ))
        return result

    async def delete_diary(
        self,
        *,
        character_id: str,
        diary_date: date
    ) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.DIARIES_TABLE}
                WHERE character_id = $1 AND diary_date = $2
                """,
                character_id, diary_date
            )

        return result == "DELETE 1"


class PostgreSQLUserRepository(UserRepository):
    TABLE_NAME = "users"

    def __init__(
        self,
        *,
        get_pool: Callable[[], Awaitable[asyncpg.Pool]] = None,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        connection_str: str = None,
        db_pool_min_size: int = 1,
        db_pool_max_size: int = 5
    ):
        self._get_pool_func = get_pool
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection_str = connection_str
        self.db_pool_min_size = db_pool_min_size
        self.db_pool_max_size = db_pool_max_size
        self._pool: asyncpg.Pool = None
        self._pool_lock = asyncio.Lock()
        self._db_initialized = False

    async def get_pool(self) -> asyncpg.Pool:
        # Use shared pool if provided
        if self._get_pool_func is not None:
            pool = await self._get_pool_func()
            if not self._db_initialized:
                async with self._pool_lock:
                    if not self._db_initialized:
                        await self.init_db(pool)
            return pool

        # Otherwise, create own pool (backward compatible)
        if self._pool is not None and self._db_initialized:
            return self._pool

        async with self._pool_lock:
            if self._pool is not None and self._db_initialized:
                return self._pool

            if self._pool is None:
                if self.connection_str:
                    self._pool = await asyncpg.create_pool(
                        dsn=self.connection_str,
                        min_size=self.db_pool_min_size,
                        max_size=self.db_pool_max_size,
                    )
                else:
                    self._pool = await asyncpg.create_pool(
                        host=self.host,
                        port=self.port,
                        database=self.dbname,
                        user=self.user,
                        password=self.password,
                        min_size=self.db_pool_min_size,
                        max_size=self.db_pool_max_size,
                    )

            await self.init_db(self._pool)

        return self._pool

    async def init_db(self, pool: asyncpg.Pool):
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        name TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'
                    )
                    """
                )
                self._db_initialized = True

            except Exception:
                logger.exception("Error at init_db")

    async def create(
        self,
        *,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        now = datetime.now(timezone.utc)
        user_id = str(uuid4())
        metadata_json = json.dumps(metadata or {})

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (id, created_at, updated_at, name, metadata)
                VALUES ($1, $2, $3, $4, $5)
                """,
                user_id, now, now, name, metadata_json
            )

        return User(
            id=user_id,
            created_at=now,
            updated_at=now,
            name=name,
            metadata=metadata
        )

    async def get(self, *, user_id: str) -> Optional[User]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, created_at, updated_at, name, metadata
                FROM {self.TABLE_NAME}
                WHERE id = $1
                """,
                user_id
            )

        if row is None:
            return None

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return User(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            name=row["name"],
            metadata=metadata
        )

    async def update(
        self,
        *,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[User]:
        now = datetime.now(timezone.utc)

        updates = ["updated_at = $1"]
        params: List[Any] = [now]
        param_idx = 2

        if name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1

        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(json.dumps(metadata))
            param_idx += 1

        params.append(user_id)

        query = f"""
            UPDATE {self.TABLE_NAME}
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, created_at, updated_at, name, metadata
        """

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row is None:
            return None

        row_metadata = row["metadata"]
        if isinstance(row_metadata, str):
            row_metadata = json.loads(row_metadata)

        return User(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            name=row["name"],
            metadata=row_metadata
        )

    async def list(self, *, limit: int = 100, offset: int = 0) -> List[User]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, created_at, updated_at, name, metadata
                FROM {self.TABLE_NAME}
                ORDER BY updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset
            )

        result = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            result.append(User(
                id=row["id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                name=row["name"],
                metadata=metadata
            ))
        return result

    async def delete(self, *, user_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                WHERE id = $1
                """,
                user_id
            )

        return result == "DELETE 1"


