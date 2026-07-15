import asyncio
import json

import pytest

from aiavatar.sts.vad.parapper_stream import ParapperStreamSpeechDetector


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.closed = False

    async def send(self, value):
        self.sent.append(value)

    async def close(self):
        self.closed = True
        await self.incoming.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.incoming.get()
        if value is None:
            raise StopAsyncIteration
        return value

    async def server_message(self, message):
        await self.incoming.put(json.dumps(message))


async def wait_for_client_message(socket):
    for _ in range(20):
        if socket.sent:
            return
        await asyncio.sleep(0)
    raise AssertionError("client did not send session.start")


@pytest.fixture
def fake_connection(monkeypatch):
    socket = FakeWebSocket()

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setattr("aiavatar.sts.vad.parapper_stream.websockets.connect", connect)
    return socket


@pytest.mark.asyncio
async def test_sends_protocol_v1_start_and_splits_audio_at_server_frame_limit(fake_connection):
    detector = ParapperStreamSpeechDetector(api_key="secret")
    task = asyncio.create_task(detector.process_samples(b"\x00\x00" * 2000, "client-session"))
    await wait_for_client_message(fake_connection)
    start = json.loads(fake_connection.sent[0])
    await fake_connection.server_message({
        "version": 1, "type": "session.ready", "session_id": start["session_id"],
        "capabilities": {"partial": True, "speech_started": True, "cancel": True},
    })
    await task

    assert start == {
        "version": 1,
        "type": "session.start",
        "session_id": start["session_id"],
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
    }
    assert [len(frame) for frame in fake_connection.sent[1:]] == [3200, 800]

    await fake_connection.server_message({
        "version": 1, "type": "session.done", "session_id": start["session_id"]
    })
    await detector.finalize_session("client-session")
    assert json.loads(fake_connection.sent[-1]) == {
        "version": 1, "type": "session.stop", "session_id": start["session_id"]
    }


@pytest.mark.asyncio
async def test_only_server_turn_final_ends_turn_without_client_silence_timer(fake_connection):
    detector = ParapperStreamSpeechDetector()
    partials = []
    finals = []

    @detector.on_speech_detecting
    async def on_partial(text, session):
        partials.append(text)

    @detector.on_speech_detected
    async def on_final(data, text, metadata, duration, session_id):
        finals.append((text, duration, session_id, metadata["turn_id"]))

    task = asyncio.create_task(detector.process_samples(b"\x01\x00" * 512, "local-id"))
    await wait_for_client_message(fake_connection)
    protocol_id = json.loads(fake_connection.sent[0])["session_id"]
    await fake_connection.server_message({
        "version": 1, "type": "session.ready", "session_id": protocol_id, "capabilities": {}
    })
    await task
    await fake_connection.server_message({
        "version": 1, "type": "speech.started", "session_id": protocol_id
    })
    await fake_connection.server_message({
        "version": 1, "type": "turn.partial", "session_id": protocol_id,
        "turn_session_id": 1, "turn_id": 2, "revision": 1, "segment_id": 3,
        "previous_segment_id": None, "text": "こんに", "source_asr_model": "test",
        "source_language": "ja", "detected_language": None, "elapsed_ms": 10,
    })
    await asyncio.sleep(0.01)
    assert partials == ["こんに"]
    assert finals == []

    # Waiting locally must never promote a partial result to a final turn.
    await asyncio.sleep(0.02)
    assert finals == []

    await fake_connection.server_message({
        "version": 1, "type": "turn.final", "session_id": protocol_id,
        "turn_session_id": 1, "turn_id": 2, "revision": 2, "segment_id": 3,
        "previous_segment_id": None, "text": "こんにちは。", "source_asr_model": "test",
        "source_language": "ja", "detected_language": None,
        "audio_duration_ms": 1280, "elapsed_ms": 20,
    })
    await asyncio.sleep(0.01)
    assert finals == [("こんにちは。", 1.28, "local-id", 2)]

    await fake_connection.server_message({
        "version": 1, "type": "session.done", "session_id": protocol_id
    })
    await detector.finalize_session("local-id")


@pytest.mark.parametrize("sample_rate,channels", [(8000, 1), (16000, 2)])
def test_rejects_audio_formats_not_supported_by_protocol_v1(sample_rate, channels):
    with pytest.raises(ValueError):
        ParapperStreamSpeechDetector(sample_rate=sample_rate, channels=channels)
