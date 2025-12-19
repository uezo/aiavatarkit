from datetime import datetime, timezone, timedelta
import asyncio
import os
import pytest
import sqlite3
from aiavatar.sts.session_state_manager import SessionState, SQLiteSessionStateManager


@pytest.fixture
def db_path(tmp_path):
    return os.path.join(tmp_path, "test_session_state.db")


@pytest.fixture
def session_manager(db_path) -> SQLiteSessionStateManager:
    return SQLiteSessionStateManager(db_path=db_path, session_timeout=3600, cache_ttl=2)


@pytest.mark.asyncio
async def test_get_session_state_empty_creates_new(session_manager):
    session_id = "non_existent_session"
    state = await session_manager.get_session_state(session_id)
    
    # Should create new session state with lazy initialization
    assert state is not None
    assert state.session_id == session_id
    assert state.active_transaction_id is None
    assert state.previous_request_timestamp is None
    assert state.previous_request_text is None
    assert state.previous_request_files is None
    assert state.updated_at is not None
    assert state.created_at is not None
    
    # Should be saved to database and cached
    assert session_id in session_manager.cache


@pytest.mark.asyncio
async def test_update_and_get_transaction(session_manager):
    session_id = "test_session_1"
    transaction_id = "txn_123"
    timestamp_inserted_at = datetime.now(timezone.utc)
    
    await session_manager.update_transaction(session_id, transaction_id, timestamp_inserted_at)
    
    state = await session_manager.get_session_state(session_id)
    assert state is not None
    assert state.session_id == session_id
    assert state.active_transaction_id == transaction_id
    assert state.timestamp_inserted_at == timestamp_inserted_at
    assert state.previous_request_timestamp is None
    assert state.previous_request_text is None
    assert state.previous_request_files is None


@pytest.mark.asyncio
async def test_update_and_get_previous_request(session_manager):
    session_id = "test_session_2"
    timestamp = datetime.now(timezone.utc)
    text = "Hello, world!"
    files = {"file1": "content1", "file2": "content2"}
    
    await session_manager.update_previous_request(session_id, timestamp, text, files)
    
    state = await session_manager.get_session_state(session_id)
    assert state is not None
    assert state.session_id == session_id
    assert state.active_transaction_id is None
    assert state.previous_request_timestamp is not None
    assert state.previous_request_text == text
    assert state.previous_request_files == files


@pytest.mark.asyncio
async def test_update_both_transaction_and_request(session_manager):
    session_id = "test_session_3"
    transaction_id = "txn_456"
    timestamp = datetime.now(timezone.utc)
    text = "Test request"
    files = {"file": "data"}
    
    # Update transaction first
    await session_manager.update_transaction(session_id, transaction_id, timestamp)
    
    # Then update previous request
    await session_manager.update_previous_request(session_id, timestamp, text, files)
    
    state = await session_manager.get_session_state(session_id)
    assert state is not None
    assert state.active_transaction_id == transaction_id
    assert state.timestamp_inserted_at == timestamp
    assert state.previous_request_text == text
    assert state.previous_request_files == files


@pytest.mark.asyncio
async def test_cache_functionality(session_manager):
    session_id = "test_cache"
    transaction_id = "txn_cache"
    timestamp_inserted_at = datetime.now(timezone.utc)
    
    # First update - will be written to DB and cache
    await session_manager.update_transaction(session_id, transaction_id, timestamp_inserted_at)
    
    # First get - should come from cache
    state1 = await session_manager.get_session_state(session_id)
    assert state1.active_transaction_id == transaction_id
    
    # Verify it's from cache by checking cache directly
    assert session_id in session_manager.cache
    assert session_manager.cache[session_id].active_transaction_id == transaction_id
    
    # Second get - should still come from cache (within TTL)
    state2 = await session_manager.get_session_state(session_id)
    assert state2.active_transaction_id == transaction_id
    
    # Wait for cache to expire (cache_ttl is 2 seconds in fixture)
    await asyncio.sleep(2.5)
    
    # Third get - should reload from DB
    state3 = await session_manager.get_session_state(session_id)
    assert state3.active_transaction_id == transaction_id


@pytest.mark.asyncio
async def test_cache_update_consistency(session_manager):
    session_id = "test_consistency"
    timestamp_inserted_at = datetime.now(timezone.utc)

    # Initial transaction
    await session_manager.update_transaction(session_id, "txn_1", timestamp_inserted_at)
    state1 = await session_manager.get_session_state(session_id)
    assert state1.active_transaction_id == "txn_1"
    
    # Update transaction - cache should be updated too
    await session_manager.update_transaction(session_id, "txn_2", timestamp_inserted_at)
    
    # Get from cache (should reflect the update)
    state2 = await session_manager.get_session_state(session_id)
    assert state2.active_transaction_id == "txn_2"
    
    # Update previous request
    timestamp = datetime.now(timezone.utc)
    await session_manager.update_previous_request(session_id, timestamp, "request text", None)
    
    # Cache should have both updates
    state3 = await session_manager.get_session_state(session_id)
    assert state3.active_transaction_id == "txn_2"
    assert state3.previous_request_text == "request text"


@pytest.mark.asyncio
async def test_clear_session(session_manager):
    session_id = "test_clear"
    timestamp_inserted_at = datetime.now(timezone.utc)
    
    await session_manager.update_transaction(session_id, "txn_clear", timestamp_inserted_at)
    
    # Verify session exists with transaction
    state = await session_manager.get_session_state(session_id)
    assert state is not None
    assert state.active_transaction_id == "txn_clear"
    
    # Clear session
    await session_manager.clear_session(session_id)
    
    # Verify cache is cleared
    assert session_id not in session_manager.cache
    
    # After clear, get_session_state should create new empty session (lazy initialization)
    state = await session_manager.get_session_state(session_id)
    assert state is not None
    assert state.active_transaction_id is None  # Should be empty/new session
    assert state.session_id == session_id


@pytest.mark.asyncio
async def test_cleanup_old_sessions(session_manager):
    timestamp_inserted_at = datetime.now(timezone.utc)

    # Create sessions with transactions
    await session_manager.update_transaction("session_1", "txn_1", timestamp_inserted_at)
    await session_manager.update_transaction("session_2", "txn_2", timestamp_inserted_at)
    await session_manager.update_transaction("session_3", "txn_3", timestamp_inserted_at)
    
    # Verify all sessions exist with transactions
    state1 = await session_manager.get_session_state("session_1")
    state2 = await session_manager.get_session_state("session_2")
    state3 = await session_manager.get_session_state("session_3")
    assert state1.active_transaction_id == "txn_1"
    assert state2.active_transaction_id == "txn_2"
    assert state3.active_transaction_id == "txn_3"
    
    # Clean up sessions older than 0 seconds (should delete all)
    await session_manager.cleanup_old_sessions(timeout_seconds=0)
    
    # Verify cache is cleared
    assert "session_1" not in session_manager.cache
    assert "session_2" not in session_manager.cache
    assert "session_3" not in session_manager.cache
    
    # After cleanup, get_session_state should create new empty sessions (lazy initialization)
    state1_new = await session_manager.get_session_state("session_1")
    state2_new = await session_manager.get_session_state("session_2")
    state3_new = await session_manager.get_session_state("session_3")
    
    # These should be newly created empty sessions
    assert state1_new.active_transaction_id is None
    assert state2_new.active_transaction_id is None
    assert state3_new.active_transaction_id is None


@pytest.mark.asyncio
async def test_multiple_sessions(session_manager):
    sessions = [
        ("session_a", "txn_a", "Request A"),
        ("session_b", "txn_b", "Request B"),
        ("session_c", "txn_c", "Request C"),
    ]
    timestamp_inserted_at = datetime.now(timezone.utc)
    
    # Create multiple sessions
    for session_id, transaction_id, text in sessions:
        await session_manager.update_transaction(session_id, transaction_id, timestamp_inserted_at)
        await session_manager.update_previous_request(
            session_id, 
            datetime.now(timezone.utc), 
            text, 
            None
        )
    
    # Verify each session independently
    for session_id, transaction_id, text in sessions:
        state = await session_manager.get_session_state(session_id)
        assert state is not None
        assert state.active_transaction_id == transaction_id
        assert state.previous_request_text == text


@pytest.mark.asyncio
async def test_overwrite_transaction(session_manager):
    session_id = "test_overwrite"
    timestamp_inserted_at = datetime.now(timezone.utc)
    
    # Set initial transaction
    await session_manager.update_transaction(session_id, "txn_old", timestamp_inserted_at)
    state1 = await session_manager.get_session_state(session_id)
    assert state1.active_transaction_id == "txn_old"
    
    # Overwrite with new transaction
    await session_manager.update_transaction(session_id, "txn_new", timestamp_inserted_at)
    state2 = await session_manager.get_session_state(session_id)
    assert state2.active_transaction_id == "txn_new"


@pytest.mark.asyncio
async def test_none_values_handling(session_manager):
    session_id = "test_none"
    
    # Update with None values
    await session_manager.update_previous_request(
        session_id,
        datetime.now(timezone.utc),
        None,  # text is None
        None   # files is None
    )
    
    state = await session_manager.get_session_state(session_id)
    assert state is not None
    assert state.previous_request_text is None
    assert state.previous_request_files is None


@pytest.mark.asyncio
async def test_init_db_adds_missing_timestamp_column(db_path):
    # Create legacy table without timestamp_inserted_at
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE session_states (
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
    conn.close()

    # init_db should add the missing column
    manager = SQLiteSessionStateManager(db_path=db_path, session_timeout=3600, cache_ttl=2)

    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(session_states)")}
    finally:
        conn.close()

    assert "timestamp_inserted_at" in columns

    state = await manager.get_session_state("legacy_session")
    assert state.timestamp_inserted_at == datetime.min.replace(tzinfo=timezone.utc)
