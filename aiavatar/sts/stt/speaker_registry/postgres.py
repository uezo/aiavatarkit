import asyncio
import json
from typing import Dict, Optional, Any, Tuple, List, AsyncIterable, Callable, Awaitable
import numpy as np
import asyncpg
from . import BaseSpeakerStore


def _to_pg_vector(arr: np.ndarray) -> str:
    """Convert numpy array to pgvector string format."""
    return "[" + ",".join(map(str, arr.astype(np.float32).tolist())) + "]"


def _from_pg_vector(s: str) -> np.ndarray:
    """Convert pgvector string format to numpy array."""
    return np.array(json.loads(s), dtype=np.float32)


class PGVectorStore(BaseSpeakerStore):
    """
    PostgreSQL + pgvector store (asyncpg version).
    Implements topk_similarity by DB-side ORDER BY distance LIMIT k.
    """
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
        table: str = "speakers",
        use_cosine: bool = False,
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
        self.table = table
        self._order_op = "<=>" if use_cosine else "<->"  # cosine vs L2 distance
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
                    max_size=self.db_pool_max_size
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.dbname,
                    user=self.user,
                    password=self.password,
                    min_size=self.db_pool_min_size,
                    max_size=self.db_pool_max_size
                )
            await self.init_db(self._pool)
            self._db_initialized = True

        return self._pool

    async def init_db(self, pool: asyncpg.Pool):
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    embedding vector(256),
                    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb
                );
            """)

    async def upsert(self, speaker_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec_str = _to_pg_vector(embedding)
        md_json = json.dumps(metadata or {})
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table} (id, embedding, metadata)
                VALUES ($1, $2::vector, $3::jsonb)
                ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata = {self.table}.metadata || EXCLUDED.metadata
                """,
                speaker_id, vec_str, md_json
            )

    async def get(self, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT embedding::text, metadata FROM {self.table} WHERE id = $1",
                speaker_id
            )
            if row is None:
                raise KeyError(f"Unknown speaker_id: {speaker_id}")
            emb = _from_pg_vector(row["embedding"])
            md = row["metadata"] or {}
            return emb, md

    async def set_metadata(self, speaker_id: str, key: str, value: Any) -> None:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE {self.table} SET metadata = metadata || $1::jsonb WHERE id = $2",
                json.dumps({key: value}), speaker_id
            )
            if result == "UPDATE 0":
                raise KeyError(f"Unknown speaker_id: {speaker_id}")

    async def get_metadata(self, speaker_id: str, key: str, default: Any = None) -> Any:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT metadata->$1 FROM {self.table} WHERE id = $2",
                key, speaker_id
            )
            if row is None:
                raise KeyError(f"Unknown speaker_id: {speaker_id}")
            return default if row[0] is None else row[0]

    async def all_items(self) -> AsyncIterable[Tuple[str, np.ndarray, Dict[str, Any]]]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT id, embedding::text, metadata FROM {self.table}")
            for row in rows:
                yield row["id"], _from_pg_vector(row["embedding"]), (row["metadata"] or {})

    async def count(self) -> int:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"SELECT COUNT(*) FROM {self.table}")
            return int(row[0])

    async def topk_similarity(self, q_norm: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """DB-side Top-K by distance; convert to cosine via dot."""
        k = max(1, k)
        vec_str = _to_pg_vector(q_norm)
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, embedding::text FROM {self.table} ORDER BY embedding {self._order_op} $1::vector LIMIT $2",
                vec_str, k
            )
        return [(row["id"], float(np.dot(_from_pg_vector(row["embedding"]), q_norm))) for row in rows]
