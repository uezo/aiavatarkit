import base64
from collections import deque
import json
from unittest.mock import AsyncMock, Mock, call

import pytest

from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.sts.vad.filters import AudioFilter, RecoverableAudioFilterError
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector


class TemporaryFailureFilter(AudioFilter):
    def __init__(self, error):
        self.error = error
        self.reset_session = Mock()

    def process(self, samples, session_id):
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return samples


class FakePipeline:
    def __init__(self, vad):
        self.vad = vad
        self.finalize = AsyncMock()

    def add_response_handler(self, handler):
        pass

    def on_accepted(self, *, channels):
        return lambda callback: callback


class FakeWebSocket:
    def __init__(self, *requests, before_disconnect=None):
        self.requests = deque(json.dumps(request) for request in requests)
        self.before_disconnect = before_disconnect
        self.accepted = False
        self.closed = False
        self.sent_messages = []

    async def accept(self, subprotocol=None):
        self.accepted = True

    async def receive_text(self):
        if self.requests:
            return self.requests.popleft()
        if self.before_disconnect:
            self.before_disconnect()
        raise RuntimeError("WebSocket is not connected")

    async def send_text(self, message):
        self.sent_messages.append(json.loads(message))

    async def close(self):
        self.closed = True


def make_server(monkeypatch, error):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", lambda *args: None)
    monkeypatch.setattr(SileroSpeechDetector, "_create_vad_iterator", lambda *args: Mock())
    vad = SileroStreamSpeechDetector(
        speech_recognizer=Mock(),
        audio_filters=[TemporaryFailureFilter(error)],
    )
    vad.process_samples = AsyncMock(wraps=vad.process_samples)
    vad.reset_session_audio_state = Mock()
    vad.reset_session = Mock()
    vad.delete_session = Mock()
    pipeline = FakePipeline(vad)
    server = AIAvatarWebSocketServer(sts=pipeline)
    endpoint = next(route.endpoint for route in server.get_websocket_router().routes
                    if route.path == "/ws")
    return server, pipeline, endpoint


def data_request(samples):
    return {
        "type": "data",
        "session_id": "session-1",
        "audio_data": base64.b64encode(samples).decode("ascii"),
    }


@pytest.mark.asyncio
async def test_recoverable_filter_error_keeps_websocket_receiving(monkeypatch):
    server, pipeline, endpoint = make_server(
        monkeypatch, RecoverableAudioFilterError("temporary overload")
    )
    created_sessions = []

    @server.on_session_start
    async def remember_session(request, session):
        created_sessions.append(session)

    def check_before_disconnect():
        assert websocket.accepted
        assert not websocket.closed
        assert server.websockets["session-1"] is websocket
        assert server.sessions["session-1"] is created_sessions[0]
        assert pipeline.vad.get_session_data("session-1", "context_id") == "context-1"
        assert pipeline.vad.get_session_data("session-1", "user_id") == "user-1"
        pipeline.finalize.assert_not_awaited()
        pipeline.vad.process_samples.assert_has_awaits([
            call(b"\x01\x00", "session-1"),
            call(b"\x02\x00", "session-1"),
        ])
        assert pipeline.vad.process_samples.await_count == 2
        assert pipeline.vad.get_session("session-1").vad_buffer == b"\x02\x00"
        pipeline.vad.reset_session_audio_state.assert_not_called()
        pipeline.vad.reset_session.assert_not_called()
        pipeline.vad.delete_session.assert_not_called()
        pipeline.vad.audio_filters[0].reset_session.assert_not_called()

    websocket = FakeWebSocket(
        {
            "type": "start",
            "session_id": "session-1",
            "user_id": "user-1",
            "context_id": "context-1",
        },
        data_request(b"\x01\x00"),
        data_request(b"\x02\x00"),
        before_disconnect=check_before_disconnect,
    )

    await endpoint(websocket)

    assert [message["type"] for message in websocket.sent_messages] == ["connected"]
    pipeline.finalize.assert_awaited_once_with("session-1")
    assert "session-1" not in server.sessions
    assert "session-1" not in server.websockets


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [BufferError("unrelated buffer error"), RuntimeError("inference failed")])
async def test_other_audio_errors_still_propagate(monkeypatch, error):
    server, pipeline, endpoint = make_server(monkeypatch, error)
    websocket = FakeWebSocket(
        {"type": "start", "session_id": "session-1"},
        data_request(b"\x01\x00"),
        data_request(b"\x02\x00"),
    )

    with pytest.raises(type(error)) as raised:
        await endpoint(websocket)

    assert raised.value is error
    pipeline.vad.process_samples.assert_awaited_once_with(b"\x01\x00", "session-1")
    pipeline.finalize.assert_awaited_once_with("session-1")
    assert "session-1" not in server.sessions
    assert "session-1" not in server.websockets
