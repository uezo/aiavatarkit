from datetime import datetime, timezone
import logging
import queue
import sqlite3
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class VisionResultRecorder:
    def __init__(self, db_path: str = "vision_stream.db"):
        self.db_path = db_path
        self.record_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.init_db()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vision_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TIMESTAMP,
                        context_id TEXT,
                        user_id TEXT,
                        attention_level INTEGER,
                        comment TEXT,
                        image_id TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_vision_records_context_id_created_at
                    ON vision_records (context_id, created_at)
                    """
                )
        finally:
            conn.close()

    def _worker(self):
        conn = sqlite3.connect(self.db_path)
        try:
            while not self.stop_event.is_set() or not self.record_queue.empty():
                try:
                    row = self.record_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                conn.execute(
                    "INSERT INTO vision_records (created_at, context_id, user_id, attention_level, comment, image_id) VALUES (?, ?, ?, ?, ?, ?)",
                    row,
                )
                conn.commit()
                self.record_queue.task_done()
        finally:
            conn.close()

    def record(self, context_id: str, attention_level: int, comment: str, image_id: Optional[str] = None, user_id: Optional[str] = None):
        self.record_queue.put((datetime.now(timezone.utc), context_id, user_id, attention_level, comment, image_id))

    def get_records(self, context_id: str, limit: int = 100, min_attention_level: Optional[int] = None, user_id: Optional[str] = None, since: Optional[datetime] = None, until: Optional[datetime] = None) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            where = ["context_id = ?"]
            params: list = [context_id]
            if user_id is not None:
                where.append("user_id = ?")
                params.append(user_id)
            if min_attention_level is not None:
                where.append("attention_level >= ?")
                params.append(min_attention_level)
            if since is not None:
                where.append("created_at >= ?")
                params.append(since)
            if until is not None:
                where.append("created_at <= ?")
                params.append(until)
            params.append(limit)
            cursor = conn.execute(
                f"""
                SELECT created_at, context_id, user_id, attention_level, comment, image_id
                FROM vision_records
                WHERE {' AND '.join(where)}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            )
            rows = cursor.fetchall()
            rows.reverse()
            return [
                {"created_at": r[0], "context_id": r[1], "user_id": r[2], "attention_level": r[3], "comment": r[4], "image_id": r[5]}
                for r in rows
            ]
        finally:
            conn.close()

    def close(self):
        self.stop_event.set()
        self.record_queue.join()
        self.worker_thread.join()
