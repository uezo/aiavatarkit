from datetime import datetime, timezone, timedelta
import json
import logging
from typing import List, Dict
import psycopg2
import psycopg2.extras
from ..base import ContextManager

logger = logging.getLogger(__name__)


class PostgreSQLContextManager(ContextManager):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        context_timeout: int = 3600
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
        self.context_timeout = context_timeout
        self.init_db()

    def connect_db(self):
        return psycopg2.connect(**self.connection_params)

    def init_db(self):
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                # Create table
                cur.execute(
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
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_histories_context_id_created_at
                    ON chat_histories (context_id, created_at)
                    """
                )
            conn.commit()
        except Exception as ex:
            logger.error(f"Error at init_db: {ex}")
            conn.rollback()
        finally:
            conn.close()

    async def get_histories(self, context_id: str, limit: int = 100) -> List[Dict]:
        conn = self.connect_db()
        try:
            sql_query = """
            SELECT serialized_data
            FROM chat_histories
            WHERE context_id = %s
            """
            params = [context_id]

            if self.context_timeout > 0:
                sql_query += " AND created_at >= %s"
                cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.context_timeout)
                params.append(cutoff_time)

            sql_query += " ORDER BY id DESC"

            if limit > 0:
                sql_query += " LIMIT %s"
                params.append(limit)

            with conn.cursor() as cur:
                cur.execute(sql_query, tuple(params))
                rows = cur.fetchall()

            rows.reverse()
            results = [row[0] for row in rows]
            return results

        except Exception as ex:
            logger.error(f"Error at get_histories: {ex}")
            return []

        finally:
            conn.close()

    async def add_histories(self, context_id: str, data_list: List[Dict], context_schema: str = None):
        if not data_list:
            return

        conn = self.connect_db()
        try:
            columns = ["created_at", "context_id", "serialized_data", "context_schema"]
            placeholders = ["%s"] * len(columns)
            sql_query = f"""
                INSERT INTO chat_histories ({', '.join(columns)}) 
                VALUES ({', '.join(placeholders)})
            """

            now_utc = datetime.now(timezone.utc)
            records = []
            for data_item in data_list:
                record = (
                    now_utc,  # created_at
                    context_id,  # context_id
                    json.dumps(data_item, ensure_ascii=False),  # serialized_data
                    context_schema,  # context_schema
                )
                records.append(record)

            with conn.cursor() as cur:
                cur.executemany(sql_query, records)
            conn.commit()

        except Exception as ex:
            logger.error(f"Error at add_histories: {ex}")
            conn.rollback()

        finally:
            conn.close()

    async def get_last_created_at(self, context_id: str) -> datetime:
        conn = self.connect_db()
        try:
            sql = """
            SELECT created_at
            FROM chat_histories
            WHERE context_id = %s
            ORDER BY id DESC
            LIMIT 1
            """
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(sql, (context_id,))
                row = cursor.fetchone()

                if row and row["created_at"]:
                    last_created_at = row["created_at"]
                    if last_created_at.tzinfo is None:
                        last_created_at = last_created_at.replace(tzinfo=timezone.utc)
                    else:
                        last_created_at = last_created_at.astimezone(timezone.utc)
                else:
                    last_created_at = datetime.min.replace(tzinfo=timezone.utc)

                return last_created_at

        except Exception as ex:
            logger.error(f"Error at get_last_created_at: {ex}")
            return datetime.min.replace(tzinfo=timezone.utc)

        finally:
            conn.close()
