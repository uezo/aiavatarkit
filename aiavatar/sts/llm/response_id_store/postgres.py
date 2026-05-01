import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Awaitable, Optional
import asyncpg
from .base import ResponseIdStore

logger = logging.getLogger(__name__)


class PostgreSQLResponseIdStore(ResponseIdStore):
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
        db_pool_max_size: int = 5,
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
        if self._get_pool_func is not None:
            pool = await self._get_pool_func()
            if not self._db_initialized:
                async with self._pool_lock:
                    if not self._db_initialized:
                        await self.init_db(pool)
            return pool

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
                    """
                    CREATE TABLE IF NOT EXISTS response_ids (
                        context_id TEXT PRIMARY KEY,
                        response_id TEXT NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                    """
                )
                self._db_initialized = True
            except Exception:
                logger.exception("Error at init_db")

    async def get(self, context_id: str) -> Optional[str]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    "SELECT response_id FROM response_ids WHERE context_id = $1",
                    context_id,
                )
                return row["response_id"] if row else None
            except Exception as ex:
                logger.exception(f"Error at get: {ex}")
                return None

    async def set(self, context_id: str, response_id: str) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO response_ids (context_id, response_id, updated_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (context_id) DO UPDATE
                    SET response_id = EXCLUDED.response_id,
                        updated_at = EXCLUDED.updated_at
                    """,
                    context_id,
                    response_id,
                    datetime.now(timezone.utc).replace(tzinfo=None),
                )
            except Exception as ex:
                logger.exception(f"Error at set: {ex}")

    async def delete(self, context_id: str) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "DELETE FROM response_ids WHERE context_id = $1",
                    context_id,
                )
            except Exception as ex:
                logger.exception(f"Error at delete: {ex}")

    async def delete_older_than(self, seconds: int) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=seconds)
                await conn.execute(
                    "DELETE FROM response_ids WHERE updated_at < $1",
                    cutoff,
                )
            except Exception as ex:
                logger.exception(f"Error at delete_older_than: {ex}")
