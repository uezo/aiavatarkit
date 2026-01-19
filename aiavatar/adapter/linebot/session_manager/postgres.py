import asyncio
from datetime import datetime, timezone
import logging
import json
from typing import Dict, Callable, Awaitable
import asyncpg
from .base import LineBotSession, LineBotSessionManager

logger = logging.getLogger(__name__)


class PostgreSQLLineBotSessionManager(LineBotSessionManager):
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
        timeout: float = 3600,
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
        self.timeout = timeout
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
                        self._db_initialized = True
            return pool

        # Otherwise, create own pool (backward compatible)
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            if self._pool is not None:
                return self._pool

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
            self._db_initialized = True

        return self._pool

    async def init_db(self, pool: asyncpg.Pool):
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS linebot_sessions (
                        linebot_user_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        context_id TEXT,
                        updated_at TIMESTAMP NOT NULL,
                        data JSON
                    )
                    """
                )
            except Exception as ex:
                logger.error(f"Error at init_db: {ex}")
                raise

    async def get_session(self, linebot_user_id: str) -> LineBotSession:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT linebot_user_id, session_id, user_id, context_id, updated_at, data
                    FROM linebot_sessions
                    WHERE linebot_user_id = $1
                    """,
                    linebot_user_id
                )

                if row:
                    updated_at = row["updated_at"] or datetime.min.replace(tzinfo=timezone.utc)
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    data: Dict = row["data"] or {}
                    if isinstance(data, str):
                        data = json.loads(data)
                    if (datetime.now(timezone.utc) - updated_at).total_seconds() <= self.timeout:
                        return LineBotSession(
                            id=row["session_id"],
                            linebot_user_id=row["linebot_user_id"],
                            user_id=row["user_id"],
                            context_id=row["context_id"],
                            updated_at=updated_at,
                            data=data,
                        )

                session = LineBotSession(linebot_user_id=linebot_user_id)
                await self.upsert_session(session)
                return session

            except Exception as ex:
                logger.error(f"Error at get_session: {ex}")
                raise

    async def upsert_session(self, linebot_session: LineBotSession):
        linebot_session.updated_at = linebot_session.updated_at or datetime.now(timezone.utc)
        updated_at_naive = linebot_session.updated_at.replace(tzinfo=None) if linebot_session.updated_at.tzinfo else linebot_session.updated_at

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                data_json = json.dumps(linebot_session.data, ensure_ascii=False) if linebot_session.data else None
                await conn.execute(
                    """
                    INSERT INTO linebot_sessions (linebot_user_id, session_id, user_id, context_id, updated_at, data)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT(linebot_user_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        user_id = EXCLUDED.user_id,
                        context_id = EXCLUDED.context_id,
                        updated_at = EXCLUDED.updated_at,
                        data = EXCLUDED.data
                    """,
                    linebot_session.linebot_user_id,
                    linebot_session.id,
                    linebot_session.user_id,
                    linebot_session.context_id,
                    updated_at_naive,
                    data_json
                )

            except Exception as ex:
                logger.error(f"Error at upsert_session: {ex}")
                raise

    async def delete_session(self, linebot_user_id: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "DELETE FROM linebot_sessions WHERE linebot_user_id = $1",
                    linebot_user_id
                )

            except Exception as ex:
                logger.error(f"Error at delete_session: {ex}")
                raise
