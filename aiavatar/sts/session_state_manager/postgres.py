import asyncio
from datetime import datetime, timezone, timedelta
import json
import logging
from typing import Optional, Dict, Any, Callable, Awaitable
import asyncpg
from .base import SessionState, SessionStateManager

logger = logging.getLogger(__name__)


class PostgreSQLSessionStateManager(SessionStateManager):
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
        session_timeout: int = 3600,
        cache_ttl: int = 60,
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
        self.session_timeout = session_timeout
        self.cache_ttl = cache_ttl  # Cache TTL in seconds
        self.cache: Dict[str, SessionState] = {}  # In-memory cache
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
                    CREATE TABLE IF NOT EXISTS session_states (
                        session_id TEXT PRIMARY KEY,
                        active_transaction_id TEXT,
                        previous_request_timestamp TIMESTAMP,
                        previous_request_text TEXT,
                        previous_request_files JSON,
                        timestamp_inserted_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP NOT NULL
                    )
                    """
                )
                await conn.execute(
                    """
                    ALTER TABLE session_states
                    ADD COLUMN IF NOT EXISTS timestamp_inserted_at TIMESTAMP NOT NULL DEFAULT '0001-01-01 00:00:00'
                    """
                )
                # Create index for cleanup operations
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_states_updated_at
                    ON session_states (updated_at)
                    """
                )
            except Exception as ex:
                logger.error(f"Error at init_db: {ex}")
                raise

    async def get_session_state(self, session_id: str) -> SessionState:
        if not session_id:
            raise ValueError("Error at get_session_state: session_id cannot be None or empty")

        # Check cache first
        if session_id in self.cache:
            cached_state = self.cache[session_id]
            if cached_state.updated_at:
                cache_age = (datetime.now(timezone.utc) - cached_state.updated_at).total_seconds()
                if cache_age < self.cache_ttl:
                    return cached_state

        # Load from database if not in cache or cache expired
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT session_id, active_transaction_id, previous_request_timestamp,
                           previous_request_text, previous_request_files, timestamp_inserted_at, updated_at, created_at
                    FROM session_states
                    WHERE session_id = $1
                    """,
                    session_id
                )

                if row:
                    previous_request_files = row["previous_request_files"]
                    if isinstance(previous_request_files, str):
                        previous_request_files = json.loads(previous_request_files)

                    state = SessionState(
                        session_id=row["session_id"],
                        active_transaction_id=row["active_transaction_id"],
                        previous_request_timestamp=self._ensure_utc(row["previous_request_timestamp"]),
                        previous_request_text=row["previous_request_text"],
                        previous_request_files=previous_request_files,
                        timestamp_inserted_at=self._ensure_utc(row["timestamp_inserted_at"]),
                        updated_at=self._ensure_utc(row["updated_at"]),
                        created_at=self._ensure_utc(row["created_at"])
                    )
                    # Update cache
                    self.cache[session_id] = state
                    return state

                # Session doesn't exist - create new one (lazy initialization)
                now_utc = datetime.now(timezone.utc)
                now_naive = now_utc.replace(tzinfo=None)
                new_state = SessionState(
                    session_id=session_id,
                    updated_at=now_utc,
                    created_at=now_utc
                )

                # Save to database
                timestamp_inserted_at_naive = new_state.timestamp_inserted_at.replace(tzinfo=None) if new_state.timestamp_inserted_at else None
                await conn.execute(
                    """
                    INSERT INTO session_states (session_id, active_transaction_id, previous_request_timestamp,
                                               previous_request_text, previous_request_files, timestamp_inserted_at, updated_at, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    session_id, None, None, None, None, timestamp_inserted_at_naive, now_naive, now_naive
                )

                # Update cache
                self.cache[session_id] = new_state
                return new_state

            except Exception as ex:
                logger.error(f"Error at get_session_state: {ex}")
                raise

    async def update_transaction(self, session_id: str, transaction_id: str, timestamp_inserted_at: datetime) -> None:
        if not session_id:
            raise ValueError("Error at update_transaction: session_id cannot be None or empty")

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                now_utc = datetime.now(timezone.utc)
                now_naive = now_utc.replace(tzinfo=None)
                timestamp_inserted_at_naive = timestamp_inserted_at.replace(tzinfo=None) if timestamp_inserted_at and timestamp_inserted_at.tzinfo else timestamp_inserted_at

                await conn.execute(
                    """
                    INSERT INTO session_states (session_id, active_transaction_id, timestamp_inserted_at, updated_at, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT(session_id) DO UPDATE SET
                        active_transaction_id = EXCLUDED.active_transaction_id,
                        timestamp_inserted_at = COALESCE(EXCLUDED.timestamp_inserted_at, session_states.timestamp_inserted_at),
                        updated_at = EXCLUDED.updated_at
                    """,
                    session_id, transaction_id, timestamp_inserted_at_naive, now_naive, now_naive
                )

                # Update cache
                if session_id in self.cache:
                    self.cache[session_id].active_transaction_id = transaction_id
                    self.cache[session_id].timestamp_inserted_at = timestamp_inserted_at
                    self.cache[session_id].updated_at = now_utc
                else:
                    # Create new cache entry
                    self.cache[session_id] = SessionState(
                        session_id=session_id,
                        active_transaction_id=transaction_id,
                        timestamp_inserted_at=timestamp_inserted_at,
                        updated_at=now_utc,
                        created_at=now_utc
                    )

            except Exception as ex:
                logger.error(f"Error at update_transaction: {ex}")
                raise

    async def update_previous_request(
        self,
        session_id: str,
        timestamp: datetime,
        text: Optional[str],
        files: Optional[Dict[str, Any]]
    ) -> None:
        if not session_id:
            raise ValueError("Error at update_previous_request: session_id cannot be None or empty")

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                now_utc = datetime.now(timezone.utc)
                now_naive = now_utc.replace(tzinfo=None)
                files_json = json.dumps(files, ensure_ascii=False) if files else None
                timestamp_naive = timestamp.replace(tzinfo=None) if timestamp and timestamp.tzinfo else timestamp
                timestamp_inserted_at_naive = datetime.min

                await conn.execute(
                    """
                    INSERT INTO session_states (
                        session_id, active_transaction_id, previous_request_timestamp, previous_request_text,
                        previous_request_files, timestamp_inserted_at, updated_at, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT(session_id) DO UPDATE SET
                        previous_request_timestamp = EXCLUDED.previous_request_timestamp,
                        previous_request_text = EXCLUDED.previous_request_text,
                        previous_request_files = EXCLUDED.previous_request_files,
                        updated_at = EXCLUDED.updated_at
                    """,
                    session_id, None, timestamp_naive, text, files_json, timestamp_inserted_at_naive, now_naive, now_naive
                )

                # Update cache
                if session_id in self.cache:
                    self.cache[session_id].previous_request_timestamp = timestamp
                    self.cache[session_id].previous_request_text = text
                    self.cache[session_id].previous_request_files = files
                    self.cache[session_id].updated_at = now_utc
                else:
                    # Create new cache entry
                    self.cache[session_id] = SessionState(
                        session_id=session_id,
                        previous_request_timestamp=timestamp,
                        previous_request_text=text,
                        previous_request_files=files,
                        updated_at=now_utc,
                        created_at=now_utc
                    )

            except Exception as ex:
                logger.error(f"Error at update_previous_request: {ex}")
                raise

    async def clear_session(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("Error at clear_session: session_id cannot be None or empty")

        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "DELETE FROM session_states WHERE session_id = $1",
                    session_id
                )

                # Remove from cache
                if session_id in self.cache:
                    del self.cache[session_id]

            except Exception as ex:
                logger.error(f"Error at clear_session: {ex}")
                raise

    async def cleanup_old_sessions(self, timeout_seconds: int = 3600) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=timeout_seconds)

                # Get list of sessions to delete
                rows = await conn.fetch(
                    "SELECT session_id FROM session_states WHERE updated_at < $1",
                    cutoff_time
                )
                sessions_to_delete = [row["session_id"] for row in rows]

                # Delete from database
                result = await conn.execute(
                    "DELETE FROM session_states WHERE updated_at < $1",
                    cutoff_time
                )
                deleted_count = int(result.split()[-1]) if result else 0
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old sessions")

                # Remove from cache
                for session_id in sessions_to_delete:
                    if session_id in self.cache:
                        del self.cache[session_id]

            except Exception as ex:
                logger.error(f"Error at cleanup_old_sessions: {ex}")
                raise

    def _ensure_utc(self, dt: datetime) -> Optional[datetime]:
        """Ensure datetime has UTC timezone"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        else:
            return dt.astimezone(timezone.utc)
