from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
import json
import sqlite3
from typing import Callable, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChannelUser(BaseModel):
    channel_id: str
    channel_user_id: str
    user_id: str = ""
    data: Dict = Field(default_factory=dict)


class UserContext(BaseModel):
    user_id: str
    context_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChannelContextBridge(ABC):
    _create_user_id: Optional[Callable[[str, str], str]] = None

    def create_user_id(self, func: Callable[[str, str], str]):
        """Decorator to register a custom user_id generator for auto_create.

        The function receives (channel_id, channel_user_id) and returns a user_id.
        If not set, channel_user_id is used as user_id (default behavior).

        Usage::

            @bridge.create_user_id
            def create_user_id(channel_id, channel_user_id):
                return str(uuid4())
        """
        self._create_user_id = func
        return func

    # Channel User operations

    @abstractmethod
    async def get_channel_user(self, channel_id: str, channel_user_id: str, auto_create: bool = False) -> Optional[ChannelUser]:
        pass

    @abstractmethod
    async def upsert_channel_user(self, channel_user: ChannelUser):
        pass

    @abstractmethod
    async def delete_channel_user(self, channel_id: str, channel_user_id: str):
        pass

    @abstractmethod
    async def find_channel_users(self, user_id: str) -> List[ChannelUser]:
        pass

    # User Context operations

    @abstractmethod
    async def get_context(self, user_id: str) -> Optional[UserContext]:
        pass

    @abstractmethod
    async def upsert_context(self, context: UserContext):
        pass

    @abstractmethod
    async def delete_context(self, user_id: str):
        pass

    # Composite operations
    async def link_channel_user(self, channel_id: str, channel_user_id: str, user_id: str) -> Optional[UserContext]:
        """Link a channel user to an app user and return their context."""
        cu = await self.get_channel_user(channel_id, channel_user_id, auto_create=True)
        cu.user_id = user_id
        await self.upsert_channel_user(cu)
        return await self.get_context(user_id)

    # Hook registration
    def bind(self, adapter, channel_id: str, auto_create_channel_user: bool = True):
        """Register on_session_start and on_response hooks to sync context_id
        between the adapter and user_contexts automatically."""

        @adapter.on_session_start
        async def on_session_start(request, session_data):
            if not request.user_id:
                return

            # Resolve channel_user_id → user_id
            channel_user = await self.get_channel_user(
                channel_id, request.user_id,
                auto_create=auto_create_channel_user
            )
            if channel_user and channel_user.user_id != request.user_id:
                request.user_id = channel_user.user_id

            # Get context_id for user

            ctx = await self.get_context(request.user_id)
            if ctx and ctx.context_id:
                request.context_id = ctx.context_id

        @adapter.on_response
        async def on_response(response, _):
            if response.type == "start" and response.user_id and response.context_id:
                await self.upsert_context(UserContext(
                    user_id=response.user_id,
                    context_id=response.context_id,
                ))


class SQLiteChannelContextBridge(ChannelContextBridge):
    def __init__(self, db_path: str = "channel_context_bridge.db", timeout: float = 3600):
        self.db_path = db_path
        self.timeout = timeout
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS channel_users (
                        channel_id TEXT NOT NULL,
                        channel_user_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        data TEXT,
                        PRIMARY KEY (channel_id, channel_user_id)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_channel_users_user_id
                    ON channel_users (user_id)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_contexts (
                        user_id TEXT NOT NULL PRIMARY KEY,
                        context_id TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
        except:
            logger.exception("Error at init_db.")
            raise
        finally:
            conn.close()

    # Channel User operations
    async def get_channel_user(self, channel_id: str, channel_user_id: str, auto_create: bool = False) -> Optional[ChannelUser]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT channel_id, channel_user_id, user_id, data
                FROM channel_users
                WHERE channel_id = ? AND channel_user_id = ?
                """,
                (channel_id, channel_user_id)
            )
            row = cursor.fetchone()
            if row:
                return ChannelUser(
                    channel_id=row[0],
                    channel_user_id=row[1],
                    user_id=row[2],
                    data=json.loads(row[3]) if row[3] else {},
                )

            if auto_create:
                user_id = self._create_user_id(channel_id, channel_user_id) if self._create_user_id else channel_user_id
                channel_user = ChannelUser(
                    channel_id=channel_id,
                    channel_user_id=channel_user_id,
                    user_id=user_id,
                )
                await self.upsert_channel_user(channel_user)
                return channel_user

            return None

        except Exception as ex:
            logger.error(f"Error at get_channel_user: {ex}")
            raise
        finally:
            conn.close()

    async def upsert_channel_user(self, channel_user: ChannelUser):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO channel_users (channel_id, channel_user_id, user_id, data)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(channel_id, channel_user_id) DO UPDATE SET
                        user_id = excluded.user_id,
                        data = excluded.data
                    """,
                    (
                        channel_user.channel_id,
                        channel_user.channel_user_id,
                        channel_user.user_id,
                        json.dumps(channel_user.data) if channel_user.data else None,
                    )
                )
        except:
            logger.exception("Error at upsert_channel_user.")
            raise
        finally:
            conn.close()

    async def delete_channel_user(self, channel_id: str, channel_user_id: str):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM channel_users WHERE channel_id = ? AND channel_user_id = ?",
                    (channel_id, channel_user_id)
                )
        except:
            logger.exception("Error at delete_channel_user.")
            raise
        finally:
            conn.close()

    async def find_channel_users(self, user_id: str) -> List[ChannelUser]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT channel_id, channel_user_id, user_id, data
                FROM channel_users
                WHERE user_id = ?
                """,
                (user_id,)
            )
            return [
                ChannelUser(
                    channel_id=row[0],
                    channel_user_id=row[1],
                    user_id=row[2],
                    data=json.loads(row[3]) if row[3] else {},
                )
                for row in cursor.fetchall()
            ]
        except Exception as ex:
            logger.error(f"Error at find_channel_users: {ex}")
            raise
        finally:
            conn.close()

    # User Context operations
    async def get_context(self, user_id: str) -> Optional[UserContext]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT user_id, context_id, updated_at
                FROM user_contexts
                WHERE user_id = ?
                """,
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                updated_at = datetime.fromisoformat(row[2])
                if (datetime.now(timezone.utc) - updated_at).total_seconds() <= self.timeout:
                    return UserContext(
                        user_id=row[0],
                        context_id=row[1],
                        updated_at=updated_at,
                    )
            return None

        except Exception as ex:
            logger.error(f"Error at get_context: {ex}")
            raise
        finally:
            conn.close()

    async def upsert_context(self, context: UserContext):
        context.updated_at = context.updated_at or datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_contexts (user_id, context_id, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        context_id = excluded.context_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        context.user_id,
                        context.context_id,
                        context.updated_at.isoformat(),
                    )
                )
        except:
            logger.exception("Error at upsert_context.")
            raise
        finally:
            conn.close()

    async def delete_context(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM user_contexts WHERE user_id = ?",
                    (user_id,)
                )
        except:
            logger.exception("Error at delete_context.")
            raise
        finally:
            conn.close()
