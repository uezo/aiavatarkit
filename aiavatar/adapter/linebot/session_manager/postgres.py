from datetime import datetime, timezone
import logging
import json
from typing import Dict
import psycopg2
import psycopg2.extras
from .base import LineBotSession, LineBotSessionManager

logger = logging.getLogger(__name__)


class PostgreSQLLineBotSessionManager(LineBotSessionManager):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        connection_str: str = None,
        timeout: float = 3600
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
        self.connection_str = connection_str
        self.timeout = timeout
        self.init_db()

    def connect_db(self):
        if self.connection_str:
            return psycopg2.connect(self.connection_str)
        else:
            return psycopg2.connect(**self.connection_params)

    def init_db(self):
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                # Create table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS linebot_sessions (
                        linebot_user_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        context_id TEXT,
                        updated_at TIMESTAMP NOT NULL,
                        data JSON
                    )
                    """
                )
            conn.commit()

        except:
            logger.exception("Error at init_db.")
            conn.rollback()

        finally:
            conn.close()

    async def get_session(self, linebot_user_id: str) -> LineBotSession:
        conn = self.connect_db()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT linebot_user_id, session_id, user_id, context_id, updated_at, data
                    FROM linebot_sessions
                    WHERE linebot_user_id = %s
                    """,
                    (linebot_user_id,)
                )
                row = cur.fetchone()

            if row:
                updated_at = row["updated_at"] or datetime.min.replace(tzinfo=timezone.utc)
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                data: Dict = row["data"] or {}
                if isinstance(data, str):
                    data = json.loads(data)
                if (datetime.now(timezone.utc) - updated_at).total_seconds() <= self.timeout:
                    return LineBotSession(
                        id=row["session_id"],
                        linebot_user_id=row["linebot_user_id"],
                        user_id=row["user_id"],
                        context_id=row["context_id"],
                        updated_at=updated_at,
                        data=data,
                    )

            session = LineBotSession(linebot_user_id=linebot_user_id)
            await self.upsert_session(session)
            return session

        except Exception as ex:
            logger.error(f"Error at get_session: {ex}")
            raise

        finally:
            conn.close()

    async def upsert_session(self, linebot_session: LineBotSession):
        linebot_session.updated_at = linebot_session.updated_at or datetime.now(timezone.utc)
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO linebot_sessions (linebot_user_id, session_id, user_id, context_id, updated_at, data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(linebot_user_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        user_id = EXCLUDED.user_id,
                        context_id = EXCLUDED.context_id,
                        updated_at = EXCLUDED.updated_at,
                        data = EXCLUDED.data
                    """,
                    (
                        linebot_session.linebot_user_id,
                        linebot_session.id,
                        linebot_session.user_id,
                        linebot_session.context_id,
                        linebot_session.updated_at,
                        psycopg2.extras.Json(linebot_session.data) if linebot_session.data else None
                    )
                )
            conn.commit()

        except:
            logger.exception("Error at upsert_session.")
            conn.rollback()
            raise

        finally:
            conn.close()

    async def delete_session(self, linebot_user_id: str):
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM linebot_sessions WHERE linebot_user_id = %s",
                    (linebot_user_id,)
                )
            conn.commit()

        except:
            logger.exception("Error at delete_session.")
            conn.rollback()
            raise

        finally:
            conn.close()
