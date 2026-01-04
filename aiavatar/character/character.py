import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4
import asyncpg
from .models import Character


class CharacterRepository:
    TABLE_NAME = "characters"

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def init_table(self):
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'
                )
                """
            )

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

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (id, created_at, updated_at, name, prompt, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                character_id, now, now, name, prompt, metadata_json
            )

        return Character(
            id=character_id,
            created_at=now,
            updated_at=now,
            name=name,
            prompt=prompt,
            metadata=metadata
        )

    async def get(self, *, character_id: str) -> Optional[Character]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, created_at, updated_at, name, prompt, metadata
                FROM {self.TABLE_NAME}
                WHERE id = $1
                """,
                character_id
            )

        if row is None:
            return None

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Character(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
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

        if metadata is not None:
            updates.append(f"metadata = ${param_idx}")
            params.append(json.dumps(metadata))
            param_idx += 1

        params.append(character_id)

        query = f"""
            UPDATE {self.TABLE_NAME}
            SET {", ".join(updates)}
            WHERE id = ${param_idx}
            RETURNING id, created_at, updated_at, name, prompt, metadata
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row is None:
            return None

        row_metadata = row["metadata"]
        if isinstance(row_metadata, str):
            row_metadata = json.loads(row_metadata)

        return Character(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            name=row["name"],
            prompt=row["prompt"],
            metadata=row_metadata
        )

    async def delete(self, *, character_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                WHERE id = $1
                """,
                character_id
            )

        return result == "DELETE 1"
