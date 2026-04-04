from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
import json
import sqlite3
from typing import Dict, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChannelSession(BaseModel):
    channel_id: str
    channel_user_id: str
    session_id: str = Field(default_factory=lambda: f"ch_sess_{uuid4()}")
    user_id: str = ""
    context_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict = Field(default_factory=dict)


class ChannelSessionManager(ABC):
    @abstractmethod
    async def get_session(self, channel_id: str, channel_user_id: str) -> ChannelSession:
        pass

    @abstractmethod
    async def upsert_session(self, session: ChannelSession):
        pass

    @abstractmethod
    async def update_context_id(self, channel_id: str, channel_user_id: str, context_id: str):
        pass

    @abstractmethod
    async def delete_session(self, channel_id: str, channel_user_id: str):
        pass


class SQLiteChannelSessionManager(ChannelSessionManager):
    def __init__(self, db_path: str = "channel_sessions.db", timeout: float = 3600):
        self.db_path = db_path
        self.timeout = timeout
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS channel_sessions (
                        channel_id TEXT NOT NULL,
                        channel_user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        context_id TEXT,
                        updated_at TEXT NOT NULL,
                        data TEXT,
                        PRIMARY KEY (channel_id, channel_user_id)
                    )
                    """
                )
        except:
            logger.exception("Error at init_db.")
            raise
        finally:
            conn.close()

    async def get_session(self, channel_id: str, channel_user_id: str) -> ChannelSession:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT channel_id, channel_user_id, session_id, user_id, context_id, updated_at, data
                FROM channel_sessions
                WHERE channel_id = ? AND channel_user_id = ?
                """,
                (channel_id, channel_user_id)
            )
            row = cursor.fetchone()
            if row:
                updated_at = datetime.fromisoformat(row[5])
                if (datetime.now(timezone.utc) - updated_at).total_seconds() <= self.timeout:
                    return ChannelSession(
                        channel_id=row[0],
                        channel_user_id=row[1],
                        session_id=row[2],
                        user_id=row[3],
                        context_id=row[4],
                        updated_at=updated_at,
                        data=json.loads(row[6]) if row[6] else {},
                    )

            # Create new session when not found or expired
            session = ChannelSession(
                channel_id=channel_id,
                channel_user_id=channel_user_id,
                user_id=channel_user_id,
            )
            await self.upsert_session(session)
            return session

        except Exception as ex:
            logger.error(f"Error at get_session: {ex}")
            raise

        finally:
            conn.close()

    async def upsert_session(self, session: ChannelSession):
        session.updated_at = session.updated_at or datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO channel_sessions (channel_id, channel_user_id, session_id, user_id, context_id, updated_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(channel_id, channel_user_id) DO UPDATE SET
                        session_id = excluded.session_id,
                        user_id = excluded.user_id,
                        context_id = excluded.context_id,
                        updated_at = excluded.updated_at,
                        data = excluded.data
                    """,
                    (
                        session.channel_id,
                        session.channel_user_id,
                        session.session_id,
                        session.user_id,
                        session.context_id,
                        session.updated_at.isoformat(),
                        json.dumps(session.data) if session.data else None,
                    )
                )

        except:
            logger.exception("Error at upsert_session.")
            raise

        finally:
            conn.close()

    async def update_context_id(self, channel_id: str, channel_user_id: str, context_id: str):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE channel_sessions
                    SET context_id = ?, updated_at = ?
                    WHERE channel_id = ? AND channel_user_id = ?
                    """,
                    (context_id, datetime.now(timezone.utc).isoformat(), channel_id, channel_user_id)
                )
        except:
            logger.exception("Error at update_context_id.")
            raise
        finally:
            conn.close()

    async def delete_session(self, channel_id: str, channel_user_id: str):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM channel_sessions WHERE channel_id = ? AND channel_user_id = ?",
                    (channel_id, channel_user_id)
                )

        except:
            logger.exception("Error at delete_session.")
            raise

        finally:
            conn.close()
