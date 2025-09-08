from datetime import datetime, timezone, timedelta
import json
import logging
from typing import Optional, Dict, Any
import psycopg2
import psycopg2.extras
from .base import SessionState, SessionStateManager

logger = logging.getLogger(__name__)


class PostgreSQLSessionStateManager(SessionStateManager):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "aiavatar",
        user: str = "postgres",
        password: str = None,
        connection_str: str = None,
        session_timeout: int = 3600,
        cache_ttl: int = 60
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
        self.connection_str = connection_str
        self.session_timeout = session_timeout
        self.cache_ttl = cache_ttl  # Cache TTL in seconds
        self.cache: Dict[str, SessionState] = {}  # In-memory cache
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
                # Create index for cleanup operations
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_states_updated_at
                    ON session_states (updated_at)
                    """
                )
            conn.commit()
        except Exception as ex:
            logger.error(f"Error at init_db: {ex}")
            conn.rollback()
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
        conn = self.connect_db()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT session_id, active_transaction_id, previous_request_timestamp,
                           previous_request_text, previous_request_files, updated_at, created_at
                    FROM session_states
                    WHERE session_id = %s
                    """,
                    (session_id,)
                )
                row = cur.fetchone()
            
            if row:
                state = SessionState(
                    session_id=row["session_id"],
                    active_transaction_id=row["active_transaction_id"],
                    previous_request_timestamp=self._ensure_utc(row["previous_request_timestamp"]),
                    previous_request_text=row["previous_request_text"],
                    previous_request_files=row["previous_request_files"],
                    updated_at=self._ensure_utc(row["updated_at"]),
                    created_at=self._ensure_utc(row["created_at"])
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_states (session_id, active_transaction_id, previous_request_timestamp,
                                               previous_request_text, previous_request_files, updated_at, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, None, None, None, None, now_utc, now_utc)
                )
            conn.commit()
            
            # Update cache
            self.cache[session_id] = new_state
            return new_state
            
        except Exception as ex:
            logger.error(f"Error at get_session_state: {ex}")
            conn.rollback()
            raise
        finally:
            conn.close()

    async def update_transaction(self, session_id: str, transaction_id: str) -> None:
        if not session_id:
            raise ValueError("Error at update_transaction: session_id cannot be None or empty")
        
        conn = self.connect_db()
        try:
            now_utc = datetime.now(timezone.utc)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_states (session_id, active_transaction_id, updated_at, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(session_id) DO UPDATE SET
                        active_transaction_id = EXCLUDED.active_transaction_id,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (session_id, transaction_id, now_utc, now_utc)
                )
            conn.commit()
            
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
            conn.rollback()
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
        
        conn = self.connect_db()
        try:
            now_utc = datetime.now(timezone.utc)
            files_json = json.dumps(files, ensure_ascii=False) if files else None
            
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_states (
                        session_id, active_transaction_id, previous_request_timestamp, previous_request_text, 
                        previous_request_files, updated_at, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(session_id) DO UPDATE SET
                        previous_request_timestamp = EXCLUDED.previous_request_timestamp,
                        previous_request_text = EXCLUDED.previous_request_text,
                        previous_request_files = EXCLUDED.previous_request_files,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (session_id, None, timestamp, text, files_json, now_utc, now_utc)
                )
            conn.commit()
            
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
            conn.rollback()
            raise
        finally:
            conn.close()

    async def clear_session(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("Error at clear_session: session_id cannot be None or empty")
        
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM session_states WHERE session_id = %s",
                    (session_id,)
                )
            conn.commit()
            
            # Remove from cache
            if session_id in self.cache:
                del self.cache[session_id]
                
        except Exception as ex:
            logger.error(f"Error at clear_session: {ex}")
            conn.rollback()
            raise
        finally:
            conn.close()

    async def cleanup_old_sessions(self, timeout_seconds: int = 3600) -> None:
        conn = self.connect_db()
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
            
            # Get list of sessions to delete
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id FROM session_states WHERE updated_at < %s",
                    (cutoff_time,)
                )
                sessions_to_delete = [row[0] for row in cur.fetchall()]
            
            # Delete from database
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM session_states WHERE updated_at < %s",
                    (cutoff_time,)
                )
                if cur.rowcount > 0:
                    logger.info(f"Cleaned up {cur.rowcount} old sessions")
            conn.commit()
            
            # Remove from cache
            for session_id in sessions_to_delete:
                if session_id in self.cache:
                    del self.cache[session_id]
                    
        except Exception as ex:
            logger.error(f"Error at cleanup_old_sessions: {ex}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_utc(self, dt: datetime) -> Optional[datetime]:
        """Ensure datetime has UTC timezone"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        else:
            return dt.astimezone(timezone.utc)
