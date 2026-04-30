import asyncio
from datetime import datetime, timezone
import logging
import json
from typing import Callable, Awaitable, Dict, List, Optional
import asyncpg
from .base import ChannelUser, UserContext, ChannelContextBridge

logger = logging.getLogger(__name__)


class PostgreSQLChannelContextBridge(ChannelContextBridge):
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
                    CREATE TABLE IF NOT EXISTS channel_users (
                        channel_id TEXT NOT NULL,
                        channel_user_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        data JSON,
                        PRIMARY KEY (channel_id, channel_user_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_channel_users_user_id
                    ON channel_users (user_id)
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_contexts (
                        user_id TEXT NOT NULL PRIMARY KEY,
                        context_id TEXT,
                        updated_at TIMESTAMP NOT NULL
                    )
                    """
                )
                self._db_initialized = True

            except Exception:
                logger.exception("Error at init_db")

    # Channel User operations
    async def get_channel_user(self, channel_id: str, channel_user_id: str, auto_create: bool = False) -> Optional[ChannelUser]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT channel_id, channel_user_id, user_id, data
                    FROM channel_users
                    WHERE channel_id = $1 AND channel_user_id = $2
                    """,
                    channel_id, channel_user_id
                )

                if row:
                    data: Dict = row["data"] or {}
                    if isinstance(data, str):
                        data = json.loads(data)
                    return ChannelUser(
                        channel_id=row["channel_id"],
                        channel_user_id=row["channel_user_id"],
                        user_id=row["user_id"],
                        data=data,
                    )

                if auto_create:
                    channel_user = ChannelUser(
                        channel_id=channel_id,
                        channel_user_id=channel_user_id,
                        user_id=channel_user_id,
                    )
                    await self.upsert_channel_user(channel_user)
                    return channel_user

                return None

            except Exception as ex:
                logger.error(f"Error at get_channel_user: {ex}")
                raise

    async def upsert_channel_user(self, channel_user: ChannelUser):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                data_json = json.dumps(channel_user.data, ensure_ascii=False) if channel_user.data else None
                await conn.execute(
                    """
                    INSERT INTO channel_users (channel_id, channel_user_id, user_id, data)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT(channel_id, channel_user_id) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        data = EXCLUDED.data
                    """,
                    channel_user.channel_id,
                    channel_user.channel_user_id,
                    channel_user.user_id,
                    data_json,
                )
            except Exception as ex:
                logger.error(f"Error at upsert_channel_user: {ex}")
                raise

    async def delete_channel_user(self, channel_id: str, channel_user_id: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "DELETE FROM channel_users WHERE channel_id = $1 AND channel_user_id = $2",
                    channel_id, channel_user_id
                )
            except Exception as ex:
                logger.error(f"Error at delete_channel_user: {ex}")
                raise

    async def find_channel_users(self, user_id: str) -> List[ChannelUser]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT channel_id, channel_user_id, user_id, data
                    FROM channel_users
                    WHERE user_id = $1
                    """,
                    user_id
                )
                result = []
                for row in rows:
                    data: Dict = row["data"] or {}
                    if isinstance(data, str):
                        data = json.loads(data)
                    result.append(ChannelUser(
                        channel_id=row["channel_id"],
                        channel_user_id=row["channel_user_id"],
                        user_id=row["user_id"],
                        data=data,
                    ))
                return result
            except Exception as ex:
                logger.error(f"Error at find_channel_users: {ex}")
                raise

    # User Context operations
    async def get_context(self, user_id: str) -> Optional[UserContext]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT user_id, context_id, updated_at
                    FROM user_contexts
                    WHERE user_id = $1
                    """,
                    user_id
                )

                if row:
                    updated_at = row["updated_at"] or datetime.min.replace(tzinfo=timezone.utc)
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - updated_at).total_seconds() <= self.timeout:
                        return UserContext(
                            user_id=row["user_id"],
                            context_id=row["context_id"],
                            updated_at=updated_at,
                        )
                return None

            except Exception as ex:
                logger.error(f"Error at get_context: {ex}")
                raise

    async def upsert_context(self, context: UserContext):
        context.updated_at = context.updated_at or datetime.now(timezone.utc)
        updated_at_naive = context.updated_at.replace(tzinfo=None) if context.updated_at.tzinfo else context.updated_at

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO user_contexts (user_id, context_id, updated_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT(user_id) DO UPDATE SET
                        context_id = EXCLUDED.context_id,
                        updated_at = EXCLUDED.updated_at
                    """,
                    context.user_id,
                    context.context_id,
                    updated_at_naive,
                )
            except Exception as ex:
                logger.error(f"Error at upsert_context: {ex}")
                raise

    async def delete_context(self, user_id: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "DELETE FROM user_contexts WHERE user_id = $1",
                    user_id
                )
            except Exception as ex:
                logger.error(f"Error at delete_context: {ex}")
                raise
