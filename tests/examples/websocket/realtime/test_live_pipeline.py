"""Selected migrated GPT-Live regressions; never contact a provider or read credentials."""

import asyncio
import base64
import json

import pytest

from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.pipeline import ResponseHandler
from examples.websocket.realtime.pipeline.live import OpenAILivePipeline


class FakeWebSocket:
    def __init__(self, *, acknowledge_start=True, acknowledge_close=True):
        self.acknowledge_start = acknowledge_start
        self.acknowledge_close = acknowledge_close
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.sending = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)

    def push(self, event):
        self.incoming.put_nowait(event)

    async def send(self, payload):
        assert not self.sending, "Concurrent sends must be serialized"
        self.sending = True
        try:
            await asyncio.sleep(0)
            event = json.loads(payload)
            self.sent.append(event)
            if event["type"] == "session.start" and self.acknowledge_start:
                self.push({"type": "session.started", "session": {"id": "remote", **event["session"]}})
            elif event["type"] == "session.close" and self.acknowledge_close:
                self.push({"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 3}})
        finally:
            self.sending = False

    async def close(self):
        if not self.closed:
            self.closed = True
            self.push(None)


async def until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


def transcript(text, *, role="assistant", start=0, end=20):
    direction = "input" if role == "user" else "output"
    return {"type": f"session.{direction}_transcript.delta", "delta": text, "start_ms": start, "end_ms": end}


def create_pipeline(monkeypatch, *, sockets=None, **kwargs):
    sockets = sockets or [FakeWebSocket()]
    connections = []

    def fake_connect(url, **options):
        connections.append((url, options))
        return sockets[len(connections) - 1]

    monkeypatch.setattr("examples.websocket.realtime.pipeline.live.connect", fake_connect)
    pipeline = OpenAILivePipeline(openai_api_key="test-key", close_timeout=0.02, **kwargs)
    events = []

    async def handle(response):
        events.append(response)

    async def stop(session_id, context_id):
        events.append(STSResponse(type="stop", session_id=session_id, context_id=context_id))

    pipeline.add_response_handler(ResponseHandler(lambda session_id: True, handle, stop))
    return pipeline, sockets, events, connections


@pytest.mark.asyncio
async def test_handshake_precedes_audio_and_captions_and_uses_raw_protocol(monkeypatch):
    socket = FakeWebSocket(acknowledge_start=False)
    pipeline, _, events, connections = create_pipeline(monkeypatch, sockets=[socket], instructions="Be brief")
    task = asyncio.create_task(pipeline.start_session(STSRequest(
        session_id="s", context_id="c", user_id="u", channel="web", transaction_id="unused",
    )))
    try:
        await until(lambda: socket.sent)
        assert events == []
        assert not task.done()
        with pytest.raises(RuntimeError, match="not ready"):
            await pipeline.process_audio_samples(b"\0\0", "s")
        config = socket.sent[0]["session"]
        assert config == {
            "model": "gpt-live-1", "instructions": "Be brief",
            "audio": {"format": {"type": "audio/pcm", "rate": 16000}, "output": {"voice": "marin"}},
            "delegation": {"type": "responses", "responses": {"model": "gpt-5.6-luna"}},
        }
        socket.push({"type": "session.started", "session": {"id": "remote", **config}})
        socket.push(transcript("こんにちは。"))
        socket.push({"type": "session.output_audio.delta", "delta": base64.b64encode(b"\0\1\2\3").decode()})
        await task
        await until(lambda: len(events) == 4)
        assert [e.type for e in events] == ["connected", "info", "chunk", "chunk"]
        assert all((e.session_id, e.context_id, e.user_id, e.transaction_id) == ("s", "c", "u", None) for e in events)
        assert all(e.metadata["channel"] == "web" for e in events)
        assert events[0].metadata == {"channel": "web", "input_sample_rate": 16000}
        assert events[1].text is None
        assert events[1].metadata == {
            "channel": "web", "role": "assistant", "start_ms": 0, "end_ms": 20,
            "partial_response_text": "こんにちは。",
        }
        assert events[2].audio_data is None
        assert events[3].audio_data == b"\0\1\2\3"
        assert events[2].metadata == events[3].metadata == {
            "channel": "web", "pcm_format": {"sample_rate": 16000, "channels": 1, "sample_width": 2},
        }
        assert connections[0][0] == "wss://api.openai.com/v1/live/sessions"
        assert connections[0][1]["additional_headers"] == {"Authorization": "Bearer test-key"}
    finally:
        await pipeline.shutdown()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_audio_sends_keep_partial_sample_and_preserve_session_isolation(monkeypatch):
    pipeline, sockets, events, _ = create_pipeline(monkeypatch, sockets=[FakeWebSocket(), FakeWebSocket()])
    try:
        await asyncio.gather(*(pipeline.start_session(STSRequest(session_id=s)) for s in ("a", "b")))
        await pipeline.process_audio_samples(b"\1", "a")
        await pipeline.process_audio_samples(b"\7", "b")
        await asyncio.gather(
            pipeline.process_audio_samples(b"\2\3\4", "a"),
            pipeline.process_audio_samples(b"\5\6", "a"),
            pipeline.process_audio_samples(b"\10", "b"),
        )
        assert [base64.b64decode(e["audio"]) for e in sockets[0].sent[1:]] == [b"\1\2\3\4", b"\5\6"]
        assert [base64.b64decode(e["audio"]) for e in sockets[1].sent[1:]] == [b"\7\10"]
        sockets[1].push(transcript("はい", role="user", start=100, end=250))
        await until(lambda: any(e.type == "info" for e in events))
        assert events[-1].session_id == "b"
        assert events[-1].metadata == {
            "role": "user", "start_ms": 100, "end_ms": 250, "partial_request_text": "はい",
        }
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [16000, 24000])
async def test_output_audio_fragments_relay_native_pcm_after_one_format_header(monkeypatch, rate):
    pipeline, sockets, events, _ = create_pipeline(monkeypatch, input_sample_rate=rate)
    assert pipeline.vad.sample_rate == rate
    fragments = [bytes(range(256)) * 2, b"\xff\x7f\x00\x80"]
    try:
        await pipeline.start_session(STSRequest(session_id="s"))
        sockets[0].push({"type": "session.output_audio.delta", "delta": ""})
        for audio in fragments:
            sockets[0].push({"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()})
        await until(lambda: len(events) == 4)
        assert [event.type for event in events] == ["connected", "chunk", "chunk", "chunk"]
        assert events[1].audio_data is None
        assert all(event.metadata == {"pcm_format": {"sample_rate": rate, "channels": 1, "sample_width": 2}}
                   for event in events[1:])
        assert all(event.text is None for event in events[1:])
        assert [event.audio_data for event in events[2:]] == fragments
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
async def test_graceful_close_preserves_usage_clears_playback_and_is_idempotent(monkeypatch):
    pipeline, sockets, events, _ = create_pipeline(monkeypatch)
    await pipeline.start_session(STSRequest(session_id="s", context_id="c"))
    await asyncio.gather(pipeline.finalize("s"), pipeline.finalize("s"))
    await pipeline.finalize("s")
    await pipeline.shutdown()
    assert [e["type"] for e in sockets[0].sent] == ["session.start", "session.close"]
    assert [e.type for e in events] == ["connected", "stop", "session_closed"]
    assert events[-1].metadata == {"reason": "close_requested", "finalized": True, "usage": {"seconds": 3}}
    assert sockets[0].closed and not pipeline._sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "cancel", "error", "wrong_format"])
async def test_failed_or_cancelled_startup_releases_connection(monkeypatch, failure):
    socket = FakeWebSocket(acknowledge_start=False)
    pipeline, _, events, _ = create_pipeline(monkeypatch, sockets=[socket], startup_timeout=0.03)
    task = asyncio.create_task(pipeline.start_session(STSRequest(session_id="s")))
    await until(lambda: socket.sent)
    if failure == "cancel":
        task.cancel()
        error_type = asyncio.CancelledError
    elif failure == "error":
        socket.push({"type": "error", "error": {"code": "invalid_request", "message": "private config"}})
        error_type = RuntimeError
    elif failure == "wrong_format":
        socket.push({"type": "session.started", "session": {"audio": {"format": {"type": "audio/pcm", "rate": 24000}}}})
        error_type = RuntimeError
    else:
        error_type = TimeoutError
    with pytest.raises(error_type):
        await task
    assert not pipeline._sessions and socket.closed
    assert "private config" not in repr(events)
    assert not any(e.type in ("connected", "final") for e in events)


@pytest.mark.asyncio
async def test_prepared_session_waits_for_activation_and_releases_metadata(monkeypatch):
    pipeline, sockets, events, _ = create_pipeline(monkeypatch)
    try:
        await pipeline.prepare_session(STSRequest(session_id="s", user_id="u", context_id="c"))
        session = pipeline._sessions["s"]
        assert session.started and session.ready.done()
        assert pipeline.vad.get_session_data("s", "user_id") == "u"
        sockets[0].push({"type": "session.output_audio.delta", "delta": "AQI="})
        sockets[0].push(transcript("接続直後の発話です。"))
        await asyncio.sleep(0)
        assert events == []
        pipeline.activate_session("s")
        await until(lambda: any(e.type == "info" for e in events))
        assert [e.type for e in events] == ["chunk", "chunk", "info"]
        assert events[0].audio_data is None and events[1].audio_data == b"\1\2"
    finally:
        await pipeline.shutdown()
    assert not pipeline._prepared_sessions
    assert pipeline.vad.get_session_data("s", "user_id") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [True, False])
async def test_remote_or_unacknowledged_close_cleans_up(monkeypatch, remote):
    socket = FakeWebSocket(acknowledge_close=False)
    pipeline, _, events, _ = create_pipeline(monkeypatch, sockets=[socket])
    try:
        await pipeline.start_session(STSRequest(session_id="s"))
        if remote:
            socket.push({"type": "session.closed", "reason": "expired", "usage": {"seconds": 10}})
            await until(lambda: events[-1].type == "session_closed")
        else:
            await pipeline.finalize("s")
        assert [e.type for e in events][-2:] == ["stop", "session_closed"]
        assert events[-1].metadata["finalized"] is remote
        assert socket.closed and not pipeline._sessions
    finally:
        await pipeline.shutdown()
