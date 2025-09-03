from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import logging
import sqlite3
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    active_transaction_id: Optional[str] = None
    previous_request_timestamp: Optional[datetime] = None
    previous_request_text: Optional[str] = None
    previous_request_files: Optional[Dict[str, Any]] = None
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class SessionStateManager(ABC):
    @abstractmethod
    async def get_session_state(self, session_id: str) -> SessionState:
        pass

    @abstractmethod
    async def update_transaction(self, session_id: str, transaction_id: str) -> None:
        pass

    @abstractmethod
    async def update_previous_request(
        self, 
        session_id: str, 
        timestamp: datetime, 
        text: Optional[str], 
        files: Optional[Dict[str, Any]]
    ) -> None:
        pass

    @abstractmethod
    async def clear_session(self, session_id: str) -> None:
        pass

    @abstractmethod
    async def cleanup_old_sessions(self, timeout_seconds: int = 3600) -> None:
        pass


class SQLiteSessionStateManager(SessionStateManager):
    def __init__(self, db_path="session_state.db", session_timeout=3600, cache_ttl=60):
        self.db_path = db_path
        self.session_timeout = session_timeout
        self.cache_ttl = cache_ttl  # Cache TTL in seconds
        self.cache: Dict[str, SessionState] = {}  # In-memory cache
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_states (
                        session_id TEXT PRIMARY KEY,
                        active_transaction_id TEXT,
                        previous_request_timestamp TIMESTAMP,
                        previous_request_text TEXT,
                        previous_request_files JSON,
                        updated_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP NOT NULL
                    )
                    """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_states_updated_at
                    ON session_states (updated_at)
                    """
                )

        except Exception as ex:
            logger.error(f"Error at init_db: {ex}")
            raise
        finally:
            conn.close()

    async def get_session_state(self, session_id: str) -> SessionState:
        if not session_id:
            raise ValueError("Error at get_session_state: session_id cannot be None or empty")

        # Check cache first
        if session_id in self.cache:
            cached_state = self.cache[session_id]
            if cached_state.updated_at:
                cache_age = (datetime.now(timezone.utc) - cached_state.updated_at).total_seconds()
                if cache_age < self.cache_ttl:
                    return cached_state
        
        # Load from database if not in cache or cache expired
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, active_transaction_id, previous_request_timestamp,
                       previous_request_text, previous_request_files, updated_at, created_at
                FROM session_states
                WHERE session_id = ?
                """,
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                state = SessionState(
                    session_id=row[0],
                    active_transaction_id=row[1],
                    previous_request_timestamp=datetime.fromisoformat(row[2]) if row[2] else None,
                    previous_request_text=row[3],
                    previous_request_files=json.loads(row[4]) if row[4] else None,
                    updated_at=datetime.fromisoformat(row[5]),
                    created_at=datetime.fromisoformat(row[6])
                )
                # Update cache
                self.cache[session_id] = state
                return state
            
            # Session doesn't exist - create new one (lazy initialization)
            now_utc = datetime.now(timezone.utc)
            new_state = SessionState(
                session_id=session_id,
                updated_at=now_utc,
                created_at=now_utc
            )
            
            # Save to database
            with conn:
                conn.execute(
                    """
                    INSERT INTO session_states (session_id, active_transaction_id, previous_request_timestamp,
                                               previous_request_text, previous_request_files, updated_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, None, None, None, None, now_utc, now_utc)
                )
            
            # Update cache
            self.cache[session_id] = new_state
            return new_state
            
        except Exception as ex:
            logger.error(f"Error at get_session_state: {ex}")
            raise
        finally:
            conn.close()

    async def update_transaction(self, session_id: str, transaction_id: str) -> None:
        if not session_id:
            raise ValueError("Error at update_transaction: session_id cannot be None or empty")
        
        conn = sqlite3.connect(self.db_path)
        try:
            now_utc = datetime.now(timezone.utc)
            with conn:
                conn.execute(
                    """
                    INSERT INTO session_states (session_id, active_transaction_id, updated_at, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        active_transaction_id = excluded.active_transaction_id,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, transaction_id, now_utc, now_utc)
                )
            
            # Update cache
            if session_id in self.cache:
                self.cache[session_id].active_transaction_id = transaction_id
                self.cache[session_id].updated_at = now_utc
            else:
                # Create new cache entry
                self.cache[session_id] = SessionState(
                    session_id=session_id,
                    active_transaction_id=transaction_id,
                    updated_at=now_utc,
                    created_at=now_utc
                )
                
        except Exception as ex:
            logger.error(f"Error at update_transaction: {ex}")
            raise
        finally:
            conn.close()

    async def update_previous_request(
        self, 
        session_id: str, 
        timestamp: datetime, 
        text: Optional[str], 
        files: Optional[Dict[str, Any]]
    ) -> None:
        if not session_id:
            raise ValueError("Error at update_previous_request: session_id cannot be None or empty")
        
        conn = sqlite3.connect(self.db_path)
        try:
            now_utc = datetime.now(timezone.utc)
            files_json = json.dumps(files, ensure_ascii=True) if files else None
            
            with conn:
                conn.execute(
                    """
                    INSERT INTO session_states (
                        session_id, active_transaction_id, previous_request_timestamp, previous_request_text, 
                        previous_request_files, updated_at, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        previous_request_timestamp = excluded.previous_request_timestamp,
                        previous_request_text = excluded.previous_request_text,
                        previous_request_files = excluded.previous_request_files,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, None, timestamp, text, files_json, now_utc, now_utc)
                )
            
            # Update cache
            if session_id in self.cache:
                self.cache[session_id].previous_request_timestamp = timestamp
                self.cache[session_id].previous_request_text = text
                self.cache[session_id].previous_request_files = files
                self.cache[session_id].updated_at = now_utc
            else:
                # Create new cache entry
                self.cache[session_id] = SessionState(
                    session_id=session_id,
                    previous_request_timestamp=timestamp,
                    previous_request_text=text,
                    previous_request_files=files,
                    updated_at=now_utc,
                    created_at=now_utc
                )
                
        except Exception as ex:
            logger.error(f"Error at update_previous_request: {ex}")
            raise
        finally:
            conn.close()

    async def clear_session(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("Error at clear_session: session_id cannot be None or empty")
        
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM session_states WHERE session_id = ?",
                    (session_id,)
                )
            
            # Remove from cache
            if session_id in self.cache:
                del self.cache[session_id]
                
        except Exception as ex:
            logger.error(f"Error at clear_session: {ex}")
            raise
        finally:
            conn.close()

    async def cleanup_old_sessions(self, timeout_seconds: int = 3600) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
            
            # Get list of sessions to delete
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id FROM session_states WHERE updated_at < ?",
                (cutoff_time,)
            )
            sessions_to_delete = [row[0] for row in cursor.fetchall()]
            
            # Delete from database
            with conn:
                cursor = conn.execute(
                    "DELETE FROM session_states WHERE updated_at < ?",
                    (cutoff_time,)
                )
                if cursor.rowcount > 0:
                    logger.info(f"Cleaned up {cursor.rowcount} old sessions")
            
            # Remove from cache
            for session_id in sessions_to_delete:
                if session_id in self.cache:
                    del self.cache[session_id]
                    
        except Exception as ex:
            logger.error(f"Error at cleanup_old_sessions: {ex}")
            raise
        finally:
            conn.close()
