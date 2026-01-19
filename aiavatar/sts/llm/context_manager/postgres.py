import asyncio
from datetime import datetime, timezone, timedelta
import json
import logging
from typing import List, Dict, Union, Callable, Awaitable
import asyncpg
from ..base import ContextManager

logger = logging.getLogger(__name__)


class PostgreSQLContextManager(ContextManager):
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
        context_timeout: int = 3600,
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
        self.context_timeout = context_timeout
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
                # Create table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_histories (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        context_id TEXT NOT NULL,
                        serialized_data JSON NOT NULL,
                        context_schema TEXT
                    )
                    """
                )
                # Create index
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_histories_context_id_created_at
                    ON chat_histories (context_id, created_at)
                    """
                )
            except Exception as ex:
                logger.exception(f"Error at init_db: {ex}")

    async def get_histories(self, context_id: Union[str, List[str]], limit: int = 100) -> List[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                if not context_id:
                    raise ValueError("context_id is required")

                where_clauses = []
                params = []
                param_index = 1

                if isinstance(context_id, (list, tuple)):
                    placeholders = ",".join([f"${i}" for i in range(param_index, param_index + len(context_id))])
                    where_clauses.append(f"context_id IN ({placeholders})")
                    params.extend(context_id)
                    param_index += len(context_id)
                else:
                    where_clauses.append(f"context_id = ${param_index}")
                    params.append(context_id)
                    param_index += 1

                if self.context_timeout > 0:
                    # Cutoff time to exclude old records
                    where_clauses.append(f"created_at >= ${param_index}")
                    cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=self.context_timeout)
                    params.append(cutoff_time)
                    param_index += 1

                params.append(limit)

                sql = f"""
                SELECT serialized_data
                FROM chat_histories
                WHERE {' AND '.join(where_clauses)}
                ORDER BY id DESC
                LIMIT ${param_index}
                """

                rows = await conn.fetch(sql, *params)

                rows = list(reversed(rows))
                results = []
                for row in rows:
                    data = row["serialized_data"]
                    if isinstance(data, str):
                        data = json.loads(data)
                    results.append(data)
                return results

            except Exception as ex:
                logger.exception(f"Error at get_histories: {ex}")
                return []

    async def add_histories(self, context_id: str, data_list: List[Dict], context_schema: str = None):
        if not data_list:
            return

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                sql_query = """
                    INSERT INTO chat_histories (created_at, context_id, serialized_data, context_schema)
                    VALUES ($1, $2, $3, $4)
                """

                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                records = []
                for data_item in data_list:
                    record = (
                        now_utc,  # created_at
                        context_id,  # context_id
                        json.dumps(data_item, ensure_ascii=False),  # serialized_data
                        context_schema,  # context_schema
                    )
                    records.append(record)

                await conn.executemany(sql_query, records)

            except Exception as ex:
                logger.exception(f"Error at add_histories: {ex}")

    async def get_last_created_at(self, context_id: str) -> datetime:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                sql = """
                SELECT created_at
                FROM chat_histories
                WHERE context_id = $1
                ORDER BY id DESC
                LIMIT 1
                """
                row = await conn.fetchrow(sql, context_id)

                if row and row["created_at"]:
                    # Normalize DB timestamp to UTC (naive -> set UTC, aware -> convert to UTC)
                    last_created_at = row["created_at"]
                    if last_created_at.tzinfo is None:
                        last_created_at = last_created_at.replace(tzinfo=timezone.utc)
                    else:
                        last_created_at = last_created_at.astimezone(timezone.utc)
                else:
                    last_created_at = datetime.min.replace(tzinfo=timezone.utc)

                return last_created_at

            except Exception as ex:
                logger.exception(f"Error at get_last_created_at: {ex}")
                return datetime.min.replace(tzinfo=timezone.utc)
