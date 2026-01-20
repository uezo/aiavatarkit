import json
import sqlite3
from datetime import datetime, timezone, date
from typing import Dict, Any, Optional, List
from uuid import uuid4
from .base import CharacterRepositoryBase, ActivityRepositoryBase, UserRepository
from ..models import Character, WeeklySchedule, DailySchedule, Diary, User


class SQLiteCharacterRepository(CharacterRepositoryBase):
    TABLE_NAME = "characters"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        name TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        metadata JSON NOT NULL DEFAULT '{{}}'
                    )
                    """
                )
        finally:
            conn.close()

    async def create(
        self,
        *,
        name: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Character:
        now = datetime.now(timezone.utc)
        character_id = str(uuid4())
        metadata_json = json.dumps(metadata or {})

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.TABLE_NAME} (id, created_at, updated_at, name, prompt, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (character_id, now.isoformat(), now.isoformat(), name, prompt, metadata_json)
                )
        finally:
            conn.close()

        return Character(
            id=character_id,
            created_at=now,
            updated_at=now,
            name=name,
            prompt=prompt,
            metadata=metadata
        )

    async def get(self, *, character_id: str) -> Optional[Character]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, created_at, updated_at, name, prompt, metadata
                FROM {self.TABLE_NAME}
                WHERE id = ?
                """,
                (character_id,)
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Character(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            name=row["name"],
            prompt=row["prompt"],
            metadata=metadata
        )

    async def update(
        self,
        *,
        character_id: str,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Character]:
        now = datetime.now(timezone.utc)

        updates = ["updated_at = ?"]
        params: List[Any] = [now.isoformat()]

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if prompt is not None:
            updates.append("prompt = ?")
            params.append(prompt)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        params.append(character_id)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET {", ".join(updates)}
                    WHERE id = ?
                    """,
                    tuple(params)
                )
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT id, created_at, updated_at, name, prompt, metadata
                    FROM {self.TABLE_NAME}
                    WHERE id = ?
                    """,
                    (character_id,)
                )
                row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        row_metadata = row["metadata"]
        if isinstance(row_metadata, str):
            row_metadata = json.loads(row_metadata)

        return Character(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            name=row["name"],
            prompt=row["prompt"],
            metadata=row_metadata
        )

    async def delete(self, *, character_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM {self.TABLE_NAME}
                    WHERE id = ?
                    """,
                    (character_id,)
                )
                return cursor.rowcount == 1
        finally:
            conn.close()


class SQLiteActivityRepository(ActivityRepositoryBase):
    WEEKLY_SCHEDULES_TABLE = "weekly_schedules"
    DAILY_SCHEDULES_TABLE = "daily_schedules"
    DIARIES_TABLE = "diaries"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.WEEKLY_SCHEDULES_TABLE} (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        character_id TEXT NOT NULL UNIQUE,
                        content TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.DAILY_SCHEDULES_TABLE} (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        character_id TEXT NOT NULL,
                        schedule_date DATE NOT NULL,
                        content TEXT NOT NULL,
                        content_context JSON NOT NULL DEFAULT '{{}}',
                        UNIQUE (character_id, schedule_date)
                    )
                    """
                )
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.DIARIES_TABLE} (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        character_id TEXT NOT NULL,
                        diary_date DATE NOT NULL,
                        content TEXT NOT NULL,
                        content_context JSON NOT NULL DEFAULT '{{}}',
                        UNIQUE (character_id, diary_date)
                    )
                    """
                )
        finally:
            conn.close()

    # WeeklySchedule operations

    async def create_weekly_schedule(
        self,
        *,
        character_id: str,
        content: str
    ) -> WeeklySchedule:
        now = datetime.now(timezone.utc)
        schedule_id = str(uuid4())

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.WEEKLY_SCHEDULES_TABLE} (id, created_at, updated_at, character_id, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (schedule_id, now.isoformat(), now.isoformat(), character_id, content)
                )
        finally:
            conn.close()

        return WeeklySchedule(
            id=schedule_id,
            created_at=now,
            updated_at=now,
            character_id=character_id,
            content=content
        )

    async def get_weekly_schedule(self, *, character_id: str) -> Optional[WeeklySchedule]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, created_at, updated_at, character_id, content
                FROM {self.WEEKLY_SCHEDULES_TABLE}
                WHERE character_id = ?
                """,
                (character_id,)
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return WeeklySchedule(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
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

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute(
                    f"""
                    UPDATE {self.WEEKLY_SCHEDULES_TABLE}
                    SET content = ?, updated_at = ?
                    WHERE character_id = ?
                    """,
                    (content, now.isoformat(), character_id)
                )
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT id, created_at, updated_at, character_id, content
                    FROM {self.WEEKLY_SCHEDULES_TABLE}
                    WHERE character_id = ?
                    """,
                    (character_id,)
                )
                row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        return WeeklySchedule(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            character_id=row["character_id"],
            content=row["content"]
        )

    async def delete_weekly_schedule(self, *, character_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM {self.WEEKLY_SCHEDULES_TABLE}
                    WHERE character_id = ?
                    """,
                    (character_id,)
                )
                return cursor.rowcount == 1
        finally:
            conn.close()

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

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.DAILY_SCHEDULES_TABLE} (id, created_at, updated_at, character_id, schedule_date, content, content_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (schedule_id, now.isoformat(), now.isoformat(), character_id, schedule_date.isoformat(), content, content_context_json)
                )
        finally:
            conn.close()

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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, created_at, updated_at, character_id, schedule_date, content, content_context
                FROM {self.DAILY_SCHEDULES_TABLE}
                WHERE character_id = ? AND schedule_date = ?
                """,
                (character_id, schedule_date.isoformat())
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        content_context = row["content_context"]
        if isinstance(content_context, str):
            content_context = json.loads(content_context)

        return DailySchedule(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            character_id=row["character_id"],
            schedule_date=date.fromisoformat(row["schedule_date"]),
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

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute(
                    f"""
                    UPDATE {self.DAILY_SCHEDULES_TABLE}
                    SET content = ?, updated_at = ?
                    WHERE character_id = ? AND schedule_date = ?
                    """,
                    (content, now.isoformat(), character_id, schedule_date.isoformat())
                )
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT id, created_at, updated_at, character_id, schedule_date, content, content_context
                    FROM {self.DAILY_SCHEDULES_TABLE}
                    WHERE character_id = ? AND schedule_date = ?
                    """,
                    (character_id, schedule_date.isoformat())
                )
                row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        content_context = row["content_context"]
        if isinstance(content_context, str):
            content_context = json.loads(content_context)

        return DailySchedule(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            character_id=row["character_id"],
            schedule_date=date.fromisoformat(row["schedule_date"]),
            content=row["content"],
            content_context=content_context
        )

    async def delete_daily_schedule(
        self,
        *,
        character_id: str,
        schedule_date: date
    ) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM {self.DAILY_SCHEDULES_TABLE}
                    WHERE character_id = ? AND schedule_date = ?
                    """,
                    (character_id, schedule_date.isoformat())
                )
                return cursor.rowcount == 1
        finally:
            conn.close()

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

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.DIARIES_TABLE} (id, created_at, updated_at, character_id, diary_date, content, content_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (diary_id, now.isoformat(), now.isoformat(), character_id, diary_date.isoformat(), content, content_context_json)
                )
        finally:
            conn.close()

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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, created_at, updated_at, character_id, diary_date, content, content_context
                FROM {self.DIARIES_TABLE}
                WHERE character_id = ? AND diary_date = ?
                """,
                (character_id, diary_date.isoformat())
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        content_context = row["content_context"]
        if isinstance(content_context, str):
            content_context = json.loads(content_context)

        return Diary(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            character_id=row["character_id"],
            diary_date=date.fromisoformat(row["diary_date"]),
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

        updates = ["updated_at = ?"]
        params: List[Any] = [now.isoformat()]

        if content is not None:
            updates.append("content = ?")
            params.append(content)

        if content_context is not None:
            updates.append("content_context = ?")
            params.append(json.dumps(content_context))

        params.append(character_id)
        params.append(diary_date.isoformat())

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute(
                    f"""
                    UPDATE {self.DIARIES_TABLE}
                    SET {", ".join(updates)}
                    WHERE character_id = ? AND diary_date = ?
                    """,
                    tuple(params)
                )
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT id, created_at, updated_at, character_id, diary_date, content, content_context
                    FROM {self.DIARIES_TABLE}
                    WHERE character_id = ? AND diary_date = ?
                    """,
                    (character_id, diary_date.isoformat())
                )
                row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        row_content_context = row["content_context"]
        if isinstance(row_content_context, str):
            row_content_context = json.loads(row_content_context)

        return Diary(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            character_id=row["character_id"],
            diary_date=date.fromisoformat(row["diary_date"]),
            content=row["content"],
            content_context=row_content_context
        )

    async def delete_diary(
        self,
        *,
        character_id: str,
        diary_date: date
    ) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM {self.DIARIES_TABLE}
                    WHERE character_id = ? AND diary_date = ?
                    """,
                    (character_id, diary_date.isoformat())
                )
                return cursor.rowcount == 1
        finally:
            conn.close()


class SQLiteUserRepository(UserRepository):
    TABLE_NAME = "users"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        name TEXT NOT NULL,
                        metadata JSON NOT NULL DEFAULT '{{}}'
                    )
                    """
                )
        finally:
            conn.close()

    async def create(
        self,
        *,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        now = datetime.now(timezone.utc)
        user_id = str(uuid4())
        metadata_json = json.dumps(metadata or {})

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.TABLE_NAME} (id, created_at, updated_at, name, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, now.isoformat(), now.isoformat(), name, metadata_json)
                )
        finally:
            conn.close()

        return User(
            id=user_id,
            created_at=now,
            updated_at=now,
            name=name,
            metadata=metadata
        )

    async def get(self, *, user_id: str) -> Optional[User]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, created_at, updated_at, name, metadata
                FROM {self.TABLE_NAME}
                WHERE id = ?
                """,
                (user_id,)
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return User(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
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

        updates = ["updated_at = ?"]
        params: List[Any] = [now.isoformat()]

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        params.append(user_id)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET {", ".join(updates)}
                    WHERE id = ?
                    """,
                    tuple(params)
                )
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT id, created_at, updated_at, name, metadata
                    FROM {self.TABLE_NAME}
                    WHERE id = ?
                    """,
                    (user_id,)
                )
                row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        row_metadata = row["metadata"]
        if isinstance(row_metadata, str):
            row_metadata = json.loads(row_metadata)

        return User(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            name=row["name"],
            metadata=row_metadata
        )

    async def delete(self, *, user_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    f"""
                    DELETE FROM {self.TABLE_NAME}
                    WHERE id = ?
                    """,
                    (user_id,)
                )
                return cursor.rowcount == 1
        finally:
            conn.close()
