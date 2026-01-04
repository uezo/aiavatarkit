from dataclasses import fields
from datetime import datetime, timezone
import logging
import queue
import threading
import asyncio
import asyncpg
from . import PerformanceRecorder, PerformanceRecord

logger = logging.getLogger(__name__)


class PostgreSQLPerformanceRecorder(PerformanceRecorder):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        connection_str: str = None,
        db_pool_size: int = 2  # Single worker only needs 1, but 2 provides failover when a connection becomes stale
    ):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.connection_str = connection_str
        self.db_pool_size = db_pool_size
        self.record_queue = queue.Queue()
        self.stop_event = threading.Event()
        self._pool: asyncpg.Pool = None
        self._pool_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop = None

        self.worker_thread = threading.Thread(target=self.start_worker, daemon=True)
        self.worker_thread.start()

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            if self._pool is not None:
                return self._pool

            if self.connection_str:
                self._pool = await asyncpg.create_pool(
                    dsn=self.connection_str,
                    min_size=self.db_pool_size,
                    max_size=self.db_pool_size,
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.dbname,
                    user=self.user,
                    password=self.password,
                    min_size=self.db_pool_size,
                    max_size=self.db_pool_size,
                )
            await self.init_db()

        return self._pool

    async def add_column_if_not_exist(self, conn, column_name):
        row = await conn.fetchrow(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name='performance_records' AND column_name=$1
            """,
            column_name
        )
        if not row:
            await conn.execute(
                f"ALTER TABLE performance_records ADD COLUMN {column_name} TEXT"
            )

    async def init_db(self):
        pool = self._pool
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_records (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ,
                        transaction_id TEXT,
                        user_id TEXT,
                        context_id TEXT,
                        voice_length REAL,
                        stt_time REAL,
                        stop_response_time REAL,
                        llm_first_chunk_time REAL,
                        llm_first_voice_chunk_time REAL,
                        llm_time REAL,
                        tts_first_chunk_time REAL,
                        tts_time REAL,
                        total_time REAL,
                        stt_name TEXT,
                        llm_name TEXT,
                        tts_name TEXT,
                        request_text TEXT,
                        request_files TEXT,
                        response_text TEXT,
                        response_voice_text TEXT
                    )
                    """
                )

                # Add request_files column if not exist (migration v0.3.0 -> 0.3.2)
                await self.add_column_if_not_exist(conn, "request_files")

                # Add user_id column if not exist (migration v0.3.2 -> 0.3.3)
                await self.add_column_if_not_exist(conn, "user_id")

                # Add transaction_id column if not exist (migration v0.3.3 -> 0.3.4)
                await self.add_column_if_not_exist(conn, "transaction_id")

                # Create index
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON performance_records (created_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_id ON performance_records (transaction_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON performance_records (user_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_context_id ON performance_records (context_id)")

            except Exception as ex:
                logger.error(f"Error at init_db: {ex}")
                raise

    def start_worker(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._worker())
        self._loop.close()

    async def _worker(self):
        pool = await self.get_pool()
        try:
            while not self.stop_event.is_set() or not self.record_queue.empty():
                try:
                    record = self.record_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    await self.insert_record(pool, record)
                except (asyncpg.InterfaceError, asyncpg.PostgresError) as e:
                    logger.warning(f"Error inserting record, retrying: {e}")
                    await asyncio.sleep(0.5)
                    try:
                        await self.insert_record(pool, record)
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")

                self.record_queue.task_done()
        finally:
            if self._pool is not None:
                await self._pool.close()

    async def insert_record(self, pool: asyncpg.Pool, record: PerformanceRecord):
        columns = [field.name for field in fields(PerformanceRecord)] + ["created_at"]
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = [getattr(record, field.name) for field in fields(PerformanceRecord)] + [datetime.now(timezone.utc)]
        sql = f"INSERT INTO performance_records ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        async with pool.acquire() as conn:
            await conn.execute(sql, *values)

    def record(self, record: PerformanceRecord):
        self.record_queue.put(record)

    def close(self):
        self.stop_event.set()
        self.record_queue.join()
        self.worker_thread.join()
