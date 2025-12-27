import json
from typing import Dict, Optional, Any, Tuple, List, AsyncIterable
import numpy as np
import asyncpg
# pip install pgvector
from pgvector.asyncpg import register_vector
from . import BaseSpeakerStore


class PGVectorStore(BaseSpeakerStore):
    """
    PostgreSQL + pgvector store (asyncpg version).
    Implements topk_similarity by DB-side ORDER BY distance LIMIT k.
    """
    def __init__(
        self,
        *,
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

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            if self.connection_str:
                self._pool = await asyncpg.create_pool(
                    dsn=self.connection_str,
                    min_size=self.db_pool_min_size,
                    max_size=self.db_pool_max_size,
                    init=register_vector
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
                    init=register_vector
                )
            await self.init_db()
        return self._pool

    async def init_db(self):
        pool = self._pool
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    embedding vector(256),
                    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb
                );
            """)

    async def upsert(self, speaker_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = embedding.astype(np.float32)
        md_json = json.dumps(metadata or {})
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table} (id, embedding, metadata)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata = {self.table}.metadata || EXCLUDED.metadata
                """,
                speaker_id, vec, md_json
            )

    async def get(self, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT embedding, metadata FROM {self.table} WHERE id = $1",
                speaker_id
            )
            if row is None:
                raise KeyError(f"Unknown speaker_id: {speaker_id}")
            emb = np.array(row["embedding"], dtype=np.float32)
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
            rows = await conn.fetch(f"SELECT id, embedding, metadata FROM {self.table}")
            for row in rows:
                yield row["id"], np.array(row["embedding"], dtype=np.float32), (row["metadata"] or {})

    async def count(self) -> int:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"SELECT COUNT(*) FROM {self.table}")
            return int(row[0])

    async def topk_similarity(self, q_norm: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """DB-side Top-K by distance; convert to cosine via dot."""
        k = max(1, k)
        vec = q_norm.astype(np.float32)
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id, embedding FROM {self.table} ORDER BY embedding {self._order_op} $1 LIMIT $2",
                vec, k
            )
        return [(row["id"], float(np.dot(np.array(row["embedding"], np.float32), q_norm))) for row in rows]
