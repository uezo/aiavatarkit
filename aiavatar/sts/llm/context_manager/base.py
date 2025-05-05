import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict

logger = logging.getLogger(__name__)


class ContextManager(ABC):
    @abstractmethod
    async def get_histories(self, context_id: str, limit: int = 100) -> List[Dict]:
        pass

    @abstractmethod
    async def add_histories(self, context_id: str, data_list: List[Dict], context_schema: str = None):
        pass

    @abstractmethod
    async def get_last_created_at(self, context_id: str) -> datetime:
        pass


class SQLiteContextManager(ContextManager):
    def __init__(self, db_path="context.db", context_timeout=3600):
        self.db_path = db_path
        self.context_timeout = context_timeout
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                # Create table if not exists
                # (Primary key 'id' automatically gets indexed by SQLite)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_histories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TIMESTAMP NOT NULL,
                        context_id TEXT NOT NULL,
                        serialized_data JSON NOT NULL,
                        context_schema TEXT
                    )
                    """
                )

                # Create an index to speed up filtering queries by context_id and created_at
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_histories_context_id_created_at
                    ON chat_histories (context_id, created_at)
                    """
                )

        except Exception as ex:
            logger.error(f"Error at init_db: {ex}")
        finally:
            conn.close()

    async def get_histories(self, context_id: str, limit: int = 100) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            sql = """
            SELECT serialized_data
            FROM chat_histories
            WHERE context_id = ?
            """
            params = [context_id]

            if self.context_timeout > 0:
                # Cutoff time to exclude old records
                sql += " AND created_at >= ?"
                cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.context_timeout)
                params.append(cutoff_time)

            sql += " ORDER BY id DESC"

            if limit > 0:
                sql += " LIMIT ?"
                params.append(limit)

            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

            # Reverse the list so that the newest item is at the end (larger index)
            rows.reverse()
            results = [json.loads(row[0]) for row in rows]
            return results

        except Exception as ex:
            logger.error(f"Error at get_histories: {ex}")
            return []

        finally:
            conn.close()

    async def add_histories(self, context_id: str, data_list: List[Dict], context_schema: str = None):
        if not data_list:
            # If the list is empty, do nothing
            return

        conn = sqlite3.connect(self.db_path)
        try:
            # Prepare INSERT statement
            columns = ["created_at", "context_id", "serialized_data", "context_schema"]
            placeholders = ["?"] * len(columns)
            sql = f"""
                INSERT INTO chat_histories ({', '.join(columns)}) 
                VALUES ({', '.join(placeholders)})
            """

            now_utc = datetime.now(timezone.utc)
            records = []
            for data_item in data_list:
                record = (
                    now_utc,                        # created_at
                    context_id,                     # context_id
                    json.dumps(data_item, ensure_ascii=True),  # serialized_data
                    context_schema,                 # context_schema
                )
                records.append(record)

            # Execute many inserts in a single statement
            conn.executemany(sql, records)
            conn.commit()

        except Exception as ex:
            logger.error(f"Error at add_histories: {ex}")
            conn.rollback()

        finally:
            conn.close()

    async def get_last_created_at(self, context_id: str) -> datetime:
        conn = sqlite3.connect(self.db_path)
        try:
            sql = """
            SELECT created_at
            FROM chat_histories
            WHERE context_id = ?
            ORDER BY id DESC
            LIMIT 1
            """
            cursor = conn.cursor()
            cursor.execute(sql, (context_id,))
            row = cursor.fetchone()
            if row:
                last_created_at = datetime.fromisoformat(row[0])
            else:
                last_created_at = datetime.min

            return last_created_at.replace(tzinfo=timezone.utc)

        except Exception as ex:
            logger.error(f"Error at get_last_created_at: {ex}")
            return datetime.min.replace(tzinfo=timezone.utc)

        finally:
            conn.close()
