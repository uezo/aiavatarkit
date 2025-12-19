from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
import json
import sqlite3
from typing import Dict
from uuid import uuid4

logger = logging.getLogger(__name__)


class LineBotSession:
    def __init__(
        self,
        *,
        id: str = None,
        linebot_user_id: str,
        user_id: str = None,
        context_id: str = None,
        updated_at: datetime = None,
        data: Dict = None
    ):
        self.id = id or f"linebot_sess_{uuid4()}"
        self.linebot_user_id = linebot_user_id
        self.user_id = user_id or linebot_user_id
        self.context_id = context_id
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.data = data or {}


class LineBotSessionManager(ABC):
    @abstractmethod
    async def get_session(self, linebot_user_id: str) -> LineBotSession:
        pass

    @abstractmethod
    async def upsert_session(self, linebot_session: LineBotSession):
        pass

    @abstractmethod
    async def delete_session(self, linebot_user_id: str):
        pass


class SQLiteLineBotSessionManager(LineBotSessionManager):
    def __init__(self, db_path: str = "linebot_sessions.db", timeout: float = 3600):
        self.db_path = db_path
        self.timeout = timeout
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS linebot_sessions (
                        linebot_user_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        context_id TEXT,
                        updated_at TEXT NOT NULL,
                        data TEXT
                    )
                    """
                )
        except:
            logger.exception(f"Error at init_db.")
            raise
        finally:
            conn.close()

    async def get_session(self, linebot_user_id: str) -> LineBotSession:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT linebot_user_id, session_id, user_id, context_id, updated_at, data
                FROM linebot_sessions
                WHERE linebot_user_id = ?
                """,
                (linebot_user_id,)
            )
            row = cursor.fetchone()
            if row:
                updated_at = datetime.fromisoformat(row[4])
                if (datetime.now(timezone.utc) - updated_at).total_seconds() <= self.timeout:
                    return LineBotSession(
                        id = row[1],
                        linebot_user_id=row[0],
                        user_id=row[2],
                        context_id = row[3],
                        updated_at = datetime.fromisoformat(row[4]),
                        data = json.loads(row[5]) if row[5] else {},
                    )

            # Create new session when not found or expired
            session = LineBotSession(
                linebot_user_id=linebot_user_id
            )
            await self.upsert_session(session)
            return session

        except Exception as ex:
            logger.error(f"Error at get_session: {ex}")
            raise

        finally:
            conn.close()

    async def upsert_session(self, linebot_session: LineBotSession):
        linebot_session.updated_at = linebot_session.updated_at or datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO linebot_sessions (linebot_user_id, session_id, user_id, context_id, updated_at, data)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(linebot_user_id) DO UPDATE SET
                        session_id = excluded.session_id,
                        user_id = excluded.user_id,
                        context_id = excluded.context_id,
                        updated_at = excluded.updated_at,
                        data = excluded.data
                    """,
                    (
                        linebot_session.linebot_user_id,
                        linebot_session.id,
                        linebot_session.user_id,
                        linebot_session.context_id,
                        linebot_session.updated_at.isoformat(),
                        json.dumps(linebot_session.data) if linebot_session.data else None
                    )
                )

        except:
            logger.exception(f"Error at upsert_session.")
            raise

        finally:
            conn.close()

    async def delete_session(self, linebot_user_id: str):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM linebot_sessions WHERE linebot_user_id = ?",
                    (linebot_user_id,)
                )

        except:
            logger.exception(f"Error at delete_session.")
            raise

        finally:
            conn.close()
