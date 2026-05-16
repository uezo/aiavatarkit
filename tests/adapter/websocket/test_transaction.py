import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiavatar.sts.models import STSResponse
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData
from aiavatar.adapter.models import AIAvatarResponse


class MockWebSocket:
    def __init__(self):
        self.sent_messages = []

    async def send_text(self, message: str):
        self.sent_messages.append(message)


def create_server_with_session(session_id: str, **kwargs):
    """Create a minimal AIAvatarWebSocketServer with a session set up for testing."""
    with patch.object(AIAvatarWebSocketServer, "__init__", lambda self, **kw: None):
        server = AIAvatarWebSocketServer()

    server.sessions = {}
    server.websockets = {}
    server.debug = kwargs.get("debug", False)
    server.response_audio_chunk_size = kwargs.get("response_audio_chunk_size", 512)
    server._on_response_handlers = []
    server.control_tag_pattern = r'\[{tag}:(\w+)\]|<{tag}\s[^>]*{attr}=["\'](\w+)["\']'

    # Set up session
    session_data = WebSocketSessionData()
    session_data.id = session_id
    server.sessions[session_id] = session_data

    # Set up websocket
    ws = MockWebSocket()
    server.websockets[session_id] = ws

    return server, session_data, ws


@pytest.mark.asyncio
async def test_accepted_sets_active_transaction_id():
    """Test that accepted response sets active_transaction_id on session"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    assert session_data.active_transaction_id is None

    await server.handle_response(STSResponse(
        type="accepted",
        session_id=session_id,
        transaction_id="txn_1",
        metadata={"block_barge_in": False}
    ))

    assert session_data.active_transaction_id == "txn_1"


@pytest.mark.asyncio
async def test_new_accepted_sends_synthetic_final_for_old_transaction():
    """Test that when a new accepted arrives, a synthetic final is sent for the old transaction"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    # Set initial transaction
    session_data.active_transaction_id = "txn_old"

    # New accepted arrives
    await server.handle_response(STSResponse(
        type="accepted",
        session_id=session_id,
        user_id="user_1",
        context_id="ctx_1",
        transaction_id="txn_new",
        metadata={"block_barge_in": False}
    ))

    # Should have sent synthetic final + accepted
    assert len(ws.sent_messages) == 2

    # First message: synthetic final for old transaction
    import json
    final_msg = json.loads(ws.sent_messages[0])
    assert final_msg["type"] == "final"
    assert final_msg["session_id"] == session_id
    assert final_msg["user_id"] == "user_1"
    assert final_msg["context_id"] == "ctx_1"
    assert final_msg["text"] == ""
    assert final_msg["voice_text"] == ""
    assert final_msg["metadata"]["interrupted"] is True

    # Second message: accepted itself
    accepted_msg = json.loads(ws.sent_messages[1])
    assert accepted_msg["type"] == "accepted"

    # active_transaction_id should be updated
    assert session_data.active_transaction_id == "txn_new"


@pytest.mark.asyncio
async def test_no_synthetic_final_when_same_transaction():
    """Test that no synthetic final is sent when accepted has the same transaction_id"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    session_data.active_transaction_id = "txn_1"

    await server.handle_response(STSResponse(
        type="accepted",
        session_id=session_id,
        transaction_id="txn_1",
        metadata={"block_barge_in": False}
    ))

    # Should only send accepted, no synthetic final
    assert len(ws.sent_messages) == 1
    import json
    msg = json.loads(ws.sent_messages[0])
    assert msg["type"] == "accepted"


@pytest.mark.asyncio
async def test_stale_response_skipped():
    """Test that responses with non-matching transaction_id are skipped"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    session_data.active_transaction_id = "txn_2"

    # Send a chunk response with old transaction_id
    await server.handle_response(STSResponse(
        type="chunk",
        session_id=session_id,
        transaction_id="txn_1",
        text="stale text",
        voice_text="stale voice",
        metadata={}
    ))

    # Nothing should be sent
    assert len(ws.sent_messages) == 0


@pytest.mark.asyncio
async def test_stale_final_skipped():
    """Test that even final responses with non-matching transaction_id are skipped"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    session_data.active_transaction_id = "txn_2"

    await server.handle_response(STSResponse(
        type="final",
        session_id=session_id,
        transaction_id="txn_1",
        text="old final",
        voice_text="old voice",
    ))

    # Stale final should be skipped
    assert len(ws.sent_messages) == 0


@pytest.mark.asyncio
async def test_response_without_transaction_id_passes_through():
    """Test that responses without transaction_id (e.g. connected) are not skipped"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    session_data.active_transaction_id = "txn_1"

    # connected response has no transaction_id
    await server.handle_response(STSResponse(
        type="connected",
        session_id=session_id,
        user_id="user_1",
        context_id="ctx_1"
    ))

    # Should pass through
    assert len(ws.sent_messages) == 1
    import json
    msg = json.loads(ws.sent_messages[0])
    assert msg["type"] == "connected"


@pytest.mark.asyncio
async def test_active_response_passes_through():
    """Test that responses with matching transaction_id are sent normally"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    session_data.active_transaction_id = "txn_1"

    await server.handle_response(STSResponse(
        type="chunk",
        session_id=session_id,
        transaction_id="txn_1",
        text="hello",
        voice_text="hello",
        metadata={}
    ))

    assert len(ws.sent_messages) == 1
    import json
    msg = json.loads(ws.sent_messages[0])
    assert msg["type"] == "chunk"
    assert msg["text"] == "hello"


@pytest.mark.asyncio
async def test_send_lock_serializes_writes():
    """Test that send_lock prevents concurrent writes"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id)

    write_order = []
    original_send_text = ws.send_text

    async def slow_send_text(message: str):
        import json
        msg = json.loads(message)
        write_order.append(f"start_{msg['type']}")
        await asyncio.sleep(0.05)
        write_order.append(f"end_{msg['type']}")
        await original_send_text(message)

    ws.send_text = slow_send_text

    # Launch two sends concurrently
    resp1 = AIAvatarResponse(type="chunk", session_id=session_id, text="first")
    resp2 = AIAvatarResponse(type="final", session_id=session_id, text="second")

    await asyncio.gather(
        server.send_response(resp1),
        server.send_response(resp2),
    )

    # With lock, writes should be serialized (no interleaving)
    assert write_order[0] == "start_chunk" or write_order[0] == "start_final"
    if write_order[0] == "start_chunk":
        assert write_order[1] == "end_chunk"
        assert write_order[2] == "start_final"
        assert write_order[3] == "end_final"
    else:
        assert write_order[1] == "end_final"
        assert write_order[2] == "start_chunk"
        assert write_order[3] == "end_chunk"


@pytest.mark.asyncio
async def test_audio_chunk_loop_breaks_on_transaction_change():
    """Test that audio chunk sending breaks when transaction_id changes mid-loop"""
    session_id = "test_session"
    server, session_data, ws = create_server_with_session(session_id, response_audio_chunk_size=100)

    session_data.active_transaction_id = "txn_1"

    # Create WAV audio data (512 bytes of PCM = 5+ chunks at 100 bytes each)
    import io
    import wave
    pcm_data = b"\x00\x01" * 256  # 512 bytes of PCM data
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_data)
    wav_data = wav_buffer.getvalue()

    # Track sends and change transaction after 2 chunks
    send_count = 0
    original_send_text = ws.send_text

    async def tracking_send_text(message: str):
        nonlocal send_count
        send_count += 1
        # After sending initial response + 2 audio chunks, simulate new transaction
        if send_count == 3:
            session_data.active_transaction_id = "txn_2"
        await original_send_text(message)

    ws.send_text = tracking_send_text

    await server.handle_response(STSResponse(
        type="chunk",
        session_id=session_id,
        transaction_id="txn_1",
        text="audio chunk",
        voice_text="audio chunk",
        audio_data=wav_data,
        metadata={}
    ))

    # Should have stopped early: 1 initial response + 2 audio chunks = 3
    # (would be 1 + 6 without break since 512/100 = 6 chunks)
    assert send_count < 7, f"Expected early break but sent {send_count} messages"
    assert send_count == 3, f"Expected 3 sends (1 initial + 2 chunks before break), got {send_count}"
