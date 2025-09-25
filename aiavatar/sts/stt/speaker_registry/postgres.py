import json
from typing import Dict, Optional, Any, Tuple, Iterable, List
import numpy as np
import psycopg2
from . import BaseSpeakerStore


class PGVectorStore(BaseSpeakerStore):
    """
    PostgreSQL + pgvector store.
    Implements topk_similarity by DB-side ORDER BY distance LIMIT k.
    """
    def __init__(self, dsn: str, table: str = "speakers", use_cosine: bool = False):
        self.dsn = dsn
        self.table = table
        self._order_op = "<=>" if use_cosine else "<->"  # cosine vs L2 distance

        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    embedding vector(256),
                    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb
                );
            """)
            conn.commit()

    def upsert(self, speaker_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = embedding.astype(np.float32).tolist()
        md_json = json.dumps(metadata or {})
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table} (id, embedding, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata = {self.table}.metadata || EXCLUDED.metadata
                """,
                (speaker_id, vec, md_json),
            )
            conn.commit()

    def get(self, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT embedding, metadata FROM {self.table} WHERE id = %s", (speaker_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Unknown speaker_id: {speaker_id}")
            emb = np.array(row[0], dtype=np.float32)
            md = row[1] or {}
            return emb, md

    def set_metadata(self, speaker_id: str, key: str, value: Any) -> None:
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {self.table} SET metadata = metadata || %s::jsonb WHERE id = %s",
                (json.dumps({key: value}), speaker_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Unknown speaker_id: {speaker_id}")
            conn.commit()

    def get_metadata(self, speaker_id: str, key: str, default: Any = None) -> Any:
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT metadata->%s FROM {self.table} WHERE id = %s", (key, speaker_id))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Unknown speaker_id: {speaker_id}")
            return default if row[0] is None else row[0]

    def all_items(self) -> Iterable[Tuple[str, np.ndarray, Dict[str, Any]]]:
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, embedding, metadata FROM {self.table}")
            for sid, emb, md in cur.fetchall():
                yield sid, np.array(emb, dtype=np.float32), (md or {})

    def count(self) -> int:
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            (n,) = cur.fetchone()
            return int(n)

    def topk_similarity(self, q_norm: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """DB-side Top-K by distance; convert to cosine via dot."""
        k = max(1, k)
        with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, embedding FROM {self.table} ORDER BY embedding {self._order_op} %s LIMIT %s",
                (q_norm.astype(np.float32).tolist(), k),
            )
            rows = cur.fetchall()
        return [(sid, float(np.dot(np.array(emb, np.float32), q_norm))) for sid, emb in rows]
