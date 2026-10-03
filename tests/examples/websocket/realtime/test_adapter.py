"""Local protocol checks; no provider clients, audio devices, or databases."""

import asyncio
import base64
import io
import json
import wave

import pytest
from fastapi import WebSocketDisconnect

from aiavatar.adapter.websocket.server import WebSocketSessionData
from aiavatar.sts.models import STSResponse
from examples.websocket.realtime.adapter import RealtimeWebSocketServer


class FakeDetector:
    def __init__(self):
        self.sessions = {}
        self.audio = []

    def on_voiced(self, callback):
        return callback

    def get_session_data(self, session_id, key):
        return self.sessions.get(session_id, {}).get(key)

    def set_session_data(self, session_id, key, value, create_session=False):
        self.sessions.setdefault(session_id, {})[key] = value

    async def process_samples(self, audio, session_id):
        self.audio.append((session_id, audio))


class FakePipeline:
    input_sample_rate = 16000

    def __init__(self):
        self.vad = FakeDetector()
        self.prepared = []
        self.activated = []
        self.finalized = []
        self.handlers = []
        self.events = []
        self.prepare_release = None
        self.prepare_failure = False

    def add_response_handler(self, handler):
        self.handlers.append(handler)

    def on_accepted(self, **kwargs):
        return lambda callback: callback

    async def prepare_session(self, request):
        self.prepared.append(request)
        if self.prepare_release:
            await self.prepare_release.wait()
        if self.prepare_failure:
            raise RuntimeError("PRIVATE_PROVIDER_DETAILS")
        request.context_id = request.context_id or "generated-context"
        self.events.append("prepared")

    def activate_session(self, session_id):
        self.activated.append(session_id)
        self.events.append("activated")

    async def finalize(self, session_id):
        self.finalized.append(session_id)
        self.vad.sessions.pop(session_id, None)


class FakeSocket:
    def __init__(self, *requests, events=None):
        self.requests = list(requests)
        self.sent = []
        self.events = events if events is not None else []
        self.closed = False
        self.headers = {}
        self.sending = False
        self.fail_send = False

    async def receive_text(self):
        if not self.requests:
            raise WebSocketDisconnect(code=1000)
        return json.dumps(self.requests.pop(0))

    async def send_text(self, message):
        assert not self.sending, "Concurrent socket writes"
        self.sending = True
        try:
            await asyncio.sleep(0)
            if self.fail_send:
                raise RuntimeError("socket closed")
            response = json.loads(message)
            self.sent.append(response)
            self.events.append(response["type"])
        finally:
            self.sending = False

    async def accept(self, subprotocol=None):
        pass

    async def close(self, code=1000):
        self.closed = True


async def start(server, session_id="one", **fields):
    socket = FakeSocket({"type": "start", "session_id": session_id, **fields}, events=server.sts.events)
    session = WebSocketSessionData()
    await server.process_websocket(socket, session)
    return socket, session


def endpoint(server):
    return next(route.endpoint for route in server.get_websocket_router().routes if route.path == "/ws")


@pytest.mark.asyncio
async def test_prepare_finishes_before_connected_and_activation_preserves_context():
    pipeline = FakePipeline()
    pipeline.prepare_release = asyncio.Event()
    server = RealtimeWebSocketServer(sts=pipeline)
    task = asyncio.create_task(start(server, metadata={"input_sample_rate": 16000, "barge_in_enabled": False}))
    try:
        await asyncio.sleep(0)
        assert pipeline.prepared and not pipeline.activated and not server.sessions
        pipeline.prepare_release.set()
        socket, session = await task
        assert pipeline.events == ["prepared", "connected", "activated"]
        assert socket.sent[0]["context_id"] == "generated-context"
        assert socket.sent[0]["metadata"] == {
            "realtime": True, "input_sample_rate": 16000, "barge_in_enabled": True,
        }
        assert pipeline.vad.get_session_data("one", "context_id") == "generated-context"
        assert pipeline.vad.get_session_data("one", "barge_in_enabled") is True
        assert session.data["stream_activated"] is True
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_start_reports_sanitized_error_and_route_finalizes_partial_session():
    pipeline = FakePipeline()
    pipeline.prepare_failure = True
    server = RealtimeWebSocketServer(sts=pipeline)
    socket = FakeSocket({"type": "start", "session_id": "one"})
    with pytest.raises(WebSocketDisconnect):
        await endpoint(server)(socket)
    assert [message["type"] for message in socket.sent] == ["error"]
    assert "PRIVATE_PROVIDER" not in json.dumps(socket.sent)
    assert not pipeline.activated
    assert pipeline.finalized == ["one"]
    assert not server.sessions and not server.websockets and socket.closed


@pytest.mark.asyncio
async def test_disconnect_cleans_one_session_and_preserves_another():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline)
    other_socket, other = await start(server, "other")
    socket = FakeSocket({"type": "start", "session_id": "one"})
    with pytest.raises(WebSocketDisconnect):
        await endpoint(server)(socket)
    assert pipeline.finalized == ["one"]
    assert server.sessions == {"other": other}
    assert set(server.websockets) == {"other"}
    assert other_socket.sent[0]["type"] == "connected"


@pytest.mark.asyncio
async def test_rejected_duplicate_connection_cannot_finalize_original_session():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline)
    _, original = await start(server)
    impostor = FakeSocket({"type": "start", "session_id": "one"})
    with pytest.raises(WebSocketDisconnect):
        await endpoint(server)(impostor)
    assert server.sessions["one"] is original
    assert not pipeline.finalized
    assert len(pipeline.prepared) == 1
    assert impostor.sent[0]["type"] == "error"


@pytest.mark.asyncio
async def test_bad_requests_do_not_replace_ownership_or_block_following_audio():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline)
    socket, session = await start(server)
    other_socket, _ = await start(server, "other")
    requests = [
        {"type": "start", "session_id": "replacement"},
        {"type": "data", "session_id": "other", "audio_data": "AAA="},
        {"type": "stop", "session_id": "other"},
        {"type": "invoke", "session_id": "one", "text": "unsupported"},
        {"type": "data", "session_id": "one", "files": [{"url": "example"}], "audio_data": "AAA="},
        {"type": "data", "session_id": "one", "audio_data": "not base64!"},
        {"type": "config", "session_id": "one", "metadata": {"barge_in_enabled": False}},
        {"type": "data", "session_id": "one", "audio_data": "AAA="},
    ]
    for request in requests:
        socket.requests.append(request)
        await server.process_websocket(socket, session)
    assert sum(message["type"] == "error" for message in socket.sent) == 6
    assert session.id == "one" and len(pipeline.prepared) == 2
    assert not other_socket.closed and not socket.closed
    assert pipeline.vad.audio == [("one", b"\x00\x00")]
    assert pipeline.vad.get_session_data("one", "barge_in_enabled") is True


@pytest.mark.asyncio
async def test_raw_pcm_uses_one_descriptor_and_session_local_stream_ids_under_send_lock():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline, response_audio_chunk_size=4)
    first, _ = await start(server)
    second, _ = await start(server, "two")
    pcm_format = {"sample_rate": 16000, "channels": 1, "sample_width": 2}
    pcm = b"\x00\x80\xff\x7f" * 3
    observed = []

    @server.on_response
    async def observe(wire, original):
        observed.append(original.session_id)

    await asyncio.gather(*[
        server.handle_response(STSResponse(
            type="chunk", session_id=identifier, audio_data=pcm,
            metadata={"pcm_format": pcm_format},
        )) for identifier in ("one", "one", "two")
    ])
    stream_ids = []
    for socket, chunks in ((first, 2), (second, 1)):
        messages = socket.sent[1:]
        assert len(messages) == chunks + 1
        descriptor = messages[0]
        stream_ids.append(descriptor["metadata"]["audio_id"])
        assert descriptor["audio_data"] is None
        assert descriptor["metadata"]["continuous_audio"] is True
        assert all("audio_frame_count" not in message["metadata"] for message in messages)
        assert {message["metadata"]["audio_id"] for message in messages} == {stream_ids[-1]}
        assert all(base64.b64decode(message["audio_data"]) == pcm for message in messages[1:])
    assert len(set(stream_ids)) == 2
    assert len(observed) == 3


@pytest.mark.asyncio
async def test_format_header_and_stop_do_not_fabricate_turn_completion():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline)
    socket, _ = await start(server)
    pcm_format = {"sample_rate": 24000, "channels": 1, "sample_width": 2}
    for _ in range(2):
        await server.handle_response(STSResponse(type="chunk", session_id="one", metadata={"pcm_format": pcm_format}))
    assert len(socket.sent) == 2
    old_id = socket.sent[-1]["metadata"]["audio_id"]
    await server.stop_response("one", None)
    await server.handle_response(STSResponse(
        type="chunk", session_id="one", audio_data=b"\x01\x00", metadata={"pcm_format": pcm_format},
    ))
    assert socket.sent[-1]["metadata"]["audio_id"] != old_id
    assert all(message["type"] != "final" for message in socket.sent)


@pytest.mark.asyncio
async def test_stop_resets_pcm_descriptor_in_wire_order_when_audio_is_already_waiting():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline)
    socket, session = await start(server)
    pcm_format = {"sample_rate": 16000, "channels": 1, "sample_width": 2}

    async def send_audio():
        await server.handle_response(STSResponse(
            type="chunk", session_id="one", audio_data=b"\x01\x00",
            metadata={"pcm_format": pcm_format},
        ))

    await send_audio()
    tasks = []
    try:
        async with session.send_lock:
            tasks.append(asyncio.create_task(send_audio()))
            await asyncio.sleep(0)
            tasks.append(asyncio.create_task(server.stop_response("one", None)))
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)
        await send_audio()
        stop_index = next(index for index, message in enumerate(socket.sent) if message["type"] == "stop")
        after_stop = socket.sent[stop_index + 1:]
        assert len(after_stop) == 2
        assert after_stop[0]["audio_data"] is None
        assert after_stop[1]["audio_data"]
        assert after_stop[0]["metadata"]["audio_id"] == after_stop[1]["metadata"]["audio_id"]
        assert after_stop[0]["metadata"]["audio_id"] != socket.sent[stop_index - 1]["metadata"]["audio_id"]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_disconnection_during_response_hook_suppresses_stale_pcm():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline)
    socket, _ = await start(server)

    @server.on_response
    async def disconnect(wire, original):
        server.sessions.pop("one")
        server.websockets.pop("one")

    await server.handle_response(STSResponse(
        type="chunk", session_id="one", audio_data=b"\x01\x00",
        metadata={"pcm_format": {"sample_rate": 16000, "channels": 1, "sample_width": 2}},
    ))
    assert [message["type"] for message in socket.sent] == ["connected"]


@pytest.mark.asyncio
async def test_shared_viewer_playback_notifications_are_ignored_after_ownership_validation():
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline)
    socket, session = await start(server)
    for session_id in ("one", "other"):
        socket.requests.append({
            "type": "playback", "session_id": session_id,
            "metadata": {"playback_event": "started", "playback_id": "audio-one"},
        })
        await server.process_websocket(socket, session)
    assert [message["type"] for message in socket.sent] == ["connected", "error"]
    assert not pipeline.vad.audio


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [0, 4])
async def test_regular_tts_wav_keeps_existing_adapter_framing(chunk_size):
    pipeline = FakePipeline()
    server = RealtimeWebSocketServer(sts=pipeline, response_audio_chunk_size=chunk_size)
    socket, _ = await start(server)
    pcm = b"\x01\x00" * 8
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(pcm)
    wav = buffer.getvalue()
    await server.handle_response(STSResponse(type="chunk", session_id="one", audio_data=wav, metadata={}))
    messages = socket.sent[1:]
    assert all(not message["metadata"].get("continuous_audio") for message in messages)
    if chunk_size:
        assert messages[0]["metadata"]["audio_frame_count"] == 8
        assert b"".join(base64.b64decode(message["audio_data"]) for message in messages[1:]) == pcm
    else:
        assert len(messages) == 1 and base64.b64decode(messages[0]["audio_data"]) == wav
