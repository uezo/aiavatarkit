from datetime import datetime, timezone
import asyncio
import os
from uuid import uuid4
import pytest
from aiavatar.sts.session_state_manager.postgres import PostgreSQLSessionStateManager

# Environment variables for PostgreSQL connection
AIAVATAR_DB_HOST = os.getenv("AIAVATAR_DB_HOST", "localhost")
AIAVATAR_DB_PORT = os.getenv("AIAVATAR_DB_PORT", 5432)
AIAVATAR_DB_NAME = os.getenv("AIAVATAR_DB_NAME", "aiavatar")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER", "postgres")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")

# Skip tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not AIAVATAR_DB_PASSWORD,
    reason="PostgreSQL credentials not configured"
)


@pytest.fixture
def session_manager() -> PostgreSQLSessionStateManager:
    return PostgreSQLSessionStateManager(
        host=AIAVATAR_DB_HOST,
        port=AIAVATAR_DB_PORT,
        dbname=AIAVATAR_DB_NAME,
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
        session_timeout=3600,
        cache_ttl=2
    )


@pytest.fixture
def unique_session_id():
    """Generate unique session ID for each test to avoid conflicts"""
    return f"test_session_{uuid4()}"


@pytest.mark.asyncio
async def test_get_session_state_empty_creates_new(session_manager, unique_session_id):
    state = await session_manager.get_session_state(unique_session_id)
    
    # Should create new session state with lazy initialization
    assert state is not None
    assert state.session_id == unique_session_id
    assert state.active_transaction_id is None
    assert state.previous_request_timestamp is None
    assert state.previous_request_text is None
    assert state.previous_request_files is None
    assert state.updated_at is not None
    assert state.created_at is not None
    
    # Should be saved to database and cached
    assert unique_session_id in session_manager.cache


@pytest.mark.asyncio
async def test_update_and_get_transaction(session_manager, unique_session_id):
    transaction_id = f"txn_{uuid4()}"
    
    await session_manager.update_transaction(unique_session_id, transaction_id)
    
    state = await session_manager.get_session_state(unique_session_id)
    assert state is not None
    assert state.session_id == unique_session_id
    assert state.active_transaction_id == transaction_id
    assert state.previous_request_timestamp is None
    assert state.previous_request_text is None
    assert state.previous_request_files is None


@pytest.mark.asyncio
async def test_update_and_get_previous_request(session_manager, unique_session_id):
    timestamp = datetime.now(timezone.utc)
    text = "Hello, world!"
    files = {"file1": "content1", "file2": "content2"}
    
    await session_manager.update_previous_request(unique_session_id, timestamp, text, files)
    
    state = await session_manager.get_session_state(unique_session_id)
    assert state is not None
    assert state.session_id == unique_session_id
    assert state.active_transaction_id is None
    assert state.previous_request_timestamp is not None
    # Compare timestamps with small tolerance for PostgreSQL timestamp precision
    assert abs((state.previous_request_timestamp - timestamp).total_seconds()) < 1
    assert state.previous_request_text == text
    assert state.previous_request_files == files


@pytest.mark.asyncio
async def test_update_both_transaction_and_request(session_manager, unique_session_id):
    transaction_id = f"txn_{uuid4()}"
    timestamp = datetime.now(timezone.utc)
    text = "Test request"
    files = {"file": "data"}
    
    # Update transaction first
    await session_manager.update_transaction(unique_session_id, transaction_id)
    
    # Then update previous request
    await session_manager.update_previous_request(unique_session_id, timestamp, text, files)
    
    state = await session_manager.get_session_state(unique_session_id)
    assert state is not None
    assert state.active_transaction_id == transaction_id
    assert state.previous_request_text == text
    assert state.previous_request_files == files


@pytest.mark.asyncio
async def test_cache_functionality(session_manager, unique_session_id):
    transaction_id = f"txn_cache_{uuid4()}"
    
    # First update - will be written to DB and cache
    await session_manager.update_transaction(unique_session_id, transaction_id)
    
    # First get - should come from cache
    state1 = await session_manager.get_session_state(unique_session_id)
    assert state1.active_transaction_id == transaction_id
    
    # Verify it's from cache by checking cache directly
    assert unique_session_id in session_manager.cache
    assert session_manager.cache[unique_session_id].active_transaction_id == transaction_id
    
    # Second get - should still come from cache (within TTL)
    state2 = await session_manager.get_session_state(unique_session_id)
    assert state2.active_transaction_id == transaction_id
    
    # Wait for cache to expire (cache_ttl is 2 seconds in fixture)
    await asyncio.sleep(2.5)
    
    # Third get - should reload from DB
    state3 = await session_manager.get_session_state(unique_session_id)
    assert state3.active_transaction_id == transaction_id


@pytest.mark.asyncio
async def test_cache_update_consistency(session_manager, unique_session_id):
    # Initial transaction
    await session_manager.update_transaction(unique_session_id, f"txn_1_{uuid4()}")
    state1 = await session_manager.get_session_state(unique_session_id)
    assert state1.active_transaction_id.startswith("txn_1_")
    
    # Update transaction - cache should be updated too
    await session_manager.update_transaction(unique_session_id, f"txn_2_{uuid4()}")
    
    # Get from cache (should reflect the update)
    state2 = await session_manager.get_session_state(unique_session_id)
    assert state2.active_transaction_id.startswith("txn_2_")
    
    # Update previous request
    timestamp = datetime.now(timezone.utc)
    await session_manager.update_previous_request(unique_session_id, timestamp, "request text", None)
    
    # Cache should have both updates
    state3 = await session_manager.get_session_state(unique_session_id)
    assert state3.active_transaction_id.startswith("txn_2_")
    assert state3.previous_request_text == "request text"


@pytest.mark.asyncio
async def test_clear_session(session_manager, unique_session_id):
    transaction_id = f"txn_clear_{uuid4()}"
    
    await session_manager.update_transaction(unique_session_id, transaction_id)
    
    # Verify session exists with transaction
    state = await session_manager.get_session_state(unique_session_id)
    assert state is not None
    assert state.active_transaction_id == transaction_id
    
    # Clear session
    await session_manager.clear_session(unique_session_id)
    
    # Verify cache is cleared
    assert unique_session_id not in session_manager.cache
    
    # After clear, get_session_state should create new empty session (lazy initialization)
    state = await session_manager.get_session_state(unique_session_id)
    assert state is not None
    assert state.active_transaction_id is None  # Should be empty/new session
    assert state.session_id == unique_session_id


@pytest.mark.asyncio
async def test_cleanup_old_sessions(session_manager):
    # Create sessions with unique IDs
    session_ids = [f"cleanup_test_{uuid4()}" for _ in range(3)]
    
    # Create sessions with transactions
    for i, session_id in enumerate(session_ids):
        await session_manager.update_transaction(session_id, f"txn_{i}")
    
    # Verify all sessions exist with transactions
    for i, session_id in enumerate(session_ids):
        state = await session_manager.get_session_state(session_id)
        assert state.active_transaction_id == f"txn_{i}"
    
    # Clean up sessions older than 0 seconds (should delete all)
    await session_manager.cleanup_old_sessions(timeout_seconds=0)
    
    # Verify cache is cleared
    for session_id in session_ids:
        assert session_id not in session_manager.cache
    
    # After cleanup, get_session_state should create new empty sessions (lazy initialization)
    for session_id in session_ids:
        state = await session_manager.get_session_state(session_id)
        assert state.active_transaction_id is None  # Should be newly created empty sessions


@pytest.mark.asyncio
async def test_multiple_sessions(session_manager):
    sessions = [
        (f"session_a_{uuid4()}", f"txn_a_{uuid4()}", "Request A"),
        (f"session_b_{uuid4()}", f"txn_b_{uuid4()}", "Request B"),
        (f"session_c_{uuid4()}", f"txn_c_{uuid4()}", "Request C"),
    ]
    
    # Create multiple sessions
    for session_id, transaction_id, text in sessions:
        await session_manager.update_transaction(session_id, transaction_id)
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
async def test_overwrite_transaction(session_manager, unique_session_id):
    # Set initial transaction
    await session_manager.update_transaction(unique_session_id, f"txn_old_{uuid4()}")
    state1 = await session_manager.get_session_state(unique_session_id)
    assert state1.active_transaction_id.startswith("txn_old_")
    
    # Overwrite with new transaction
    await session_manager.update_transaction(unique_session_id, f"txn_new_{uuid4()}")
    state2 = await session_manager.get_session_state(unique_session_id)
    assert state2.active_transaction_id.startswith("txn_new_")


@pytest.mark.asyncio
async def test_none_values_handling(session_manager, unique_session_id):
    # Update with None values
    await session_manager.update_previous_request(
        unique_session_id,
        datetime.now(timezone.utc),
        None,  # text is None
        None   # files is None
    )
    
    state = await session_manager.get_session_state(unique_session_id)
    assert state is not None
    assert state.previous_request_text is None
    assert state.previous_request_files is None


@pytest.mark.asyncio
async def test_timezone_handling(session_manager, unique_session_id):
    """Test that timestamps are properly handled with timezone information"""
    # Create timestamp with timezone
    timestamp_with_tz = datetime.now(timezone.utc)
    
    await session_manager.update_previous_request(
        unique_session_id,
        timestamp_with_tz,
        "Test with timezone",
        None
    )
    
    state = await session_manager.get_session_state(unique_session_id)
    assert state.previous_request_timestamp is not None
    assert state.previous_request_timestamp.tzinfo is not None
    assert state.previous_request_timestamp.tzinfo == timezone.utc
    
    # Verify updated_at and created_at also have timezone
    assert state.updated_at.tzinfo == timezone.utc
    assert state.created_at.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_error_handling_invalid_session_id(session_manager):
    """Test that proper error is raised for invalid session_id"""
    with pytest.raises(ValueError) as exc_info:
        await session_manager.get_session_state(None)
    assert "session_id cannot be None or empty" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        await session_manager.update_transaction(None, "txn_123")
    assert "session_id cannot be None or empty" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        await session_manager.update_previous_request(None, datetime.now(timezone.utc), "text", None)
    assert "session_id cannot be None or empty" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        await session_manager.clear_session(None)
    assert "session_id cannot be None or empty" in str(exc_info.value)