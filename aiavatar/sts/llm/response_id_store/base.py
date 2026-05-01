import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseIdStore:
    """Duck-typing interface for response ID storage.

    Maps context_id to an external service's conversation identifier
    (e.g. OpenAI Responses API response_id, Dify conversation_id).
    """

    async def get(self, context_id: str) -> Optional[str]:
        raise NotImplementedError

    async def set(self, context_id: str, response_id: str) -> None:
        raise NotImplementedError

    async def delete(self, context_id: str) -> None:
        raise NotImplementedError

    async def delete_older_than(self, seconds: int) -> None:
        raise NotImplementedError


class SQLiteResponseIdStore(ResponseIdStore):
    def __init__(self, db_path: str = "aiavatar.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS response_ids (
                        context_id TEXT PRIMARY KEY,
                        response_id TEXT NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                    """
                )
        except Exception as ex:
            logger.exception(f"Error at init_db: {ex}")
        finally:
            conn.close()

    async def get(self, context_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_id FROM response_ids WHERE context_id = ?",
                (context_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as ex:
            logger.exception(f"Error at get: {ex}")
            return None
        finally:
            conn.close()

    async def set(self, context_id: str, response_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO response_ids (context_id, response_id, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (context_id, response_id, datetime.now(timezone.utc)),
                )
        except Exception as ex:
            logger.exception(f"Error at set: {ex}")
        finally:
            conn.close()

    async def delete(self, context_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM response_ids WHERE context_id = ?",
                    (context_id,),
                )
        except Exception as ex:
            logger.exception(f"Error at delete: {ex}")
        finally:
            conn.close()

    async def delete_older_than(self, seconds: int) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
            with conn:
                conn.execute(
                    "DELETE FROM response_ids WHERE updated_at < ?",
                    (cutoff,),
                )
        except Exception as ex:
            logger.exception(f"Error at delete_older_than: {ex}")
        finally:
            conn.close()
