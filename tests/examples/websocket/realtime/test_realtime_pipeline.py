"""Selected migrated pipeline regressions using only fake provider sockets and TTS."""

import asyncio
import base64
import json
import logging
import struct

import pytest

from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.pipeline import ResponseHandler
from examples.websocket.realtime.pipeline.realtime import OpenAIRealtimePipeline


class FakeWebSocket:
    def __init__(self, *, acknowledge_start=True):
        self.acknowledge_start = acknowledge_start
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
        assert not self.sending, "Upstream writes must be serialized"
        self.sending = True
        try:
            await asyncio.sleep(0)
            event = json.loads(payload)
            self.sent.append(event)
            if event["type"] == "session.update" and self.acknowledge_start:
                self.push({"type": "session.updated", "session": event["session"]})
        finally:
            self.sending = False

    async def close(self):
        if not self.closed:
            self.closed = True
            self.push(None)


class FakeTTS:
    def __init__(self, *, blocked=False, fail=False):
        self.calls = []
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.cancelled = False
        self.closed = False
        self.fail = fail

    @property
    def texts(self):
        return [call[0] for call in self.calls]

    async def synthesize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.fail:
            raise RuntimeError("private synthesis diagnostics")
        return b"RIFF" + text.encode()

    async def close(self):
        self.closed = True


class FakeContextManager:
    async def get_histories(self, context_id, limit=100):
        return []

    async def add_histories(self, context_id, data_list, context_schema=None):
        pass


async def until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


def create_pipeline(monkeypatch, *, sockets=None, tts=None, **kwargs):
    sockets = sockets or [FakeWebSocket()]
    connections = []

    def fake_connect(url, **options):
        connections.append((url, options))
        return sockets[len(connections) - 1]

    monkeypatch.setattr("examples.websocket.realtime.pipeline.realtime.connection.connect", fake_connect)
    tts = tts or FakeTTS()
    kwargs.setdefault("context_manager", FakeContextManager())
    pipeline = OpenAIRealtimePipeline(
        tts=tts, openai_api_key="test-key", close_timeout=0.02, **kwargs,
    )
    events = []

    async def handle(response):
        events.append(response)

    async def stop(session_id, context_id):
        events.append(STSResponse(type="stop", session_id=session_id, context_id=context_id))

    pipeline.add_response_handler(ResponseHandler(lambda session_id: True, handle, stop))
    return pipeline, sockets, events, connections


def sent(socket, event_type):
    return [event for event in socket.sent if event["type"] == event_type]


async def commit(socket, *, item_id="item-1"):
    count = len(sent(socket, "response.create"))
    socket.push({"type": "input_audio_buffer.committed", "item_id": item_id})
    await until(lambda: len(sent(socket, "response.create")) > count)
    return sent(socket, "response.create")[-1]["response"]["metadata"]["transaction_id"]


async def begin_turn(socket, events, *, response_id="response-1", item_id="item-1"):
    transaction_id = await commit(socket, item_id=item_id)
    socket.push({"type": "response.created", "response": {
        "id": response_id, "metadata": {"transaction_id": transaction_id},
    }})
    await until(lambda: any(event.type == "start" and event.transaction_id == transaction_id for event in events))
    return transaction_id


def text_delta(socket, text, response_id="response-1"):
    socket.push({"type": "response.output_text.delta", "response_id": response_id, "delta": text})


def response_done(socket, response_id="response-1", status="completed"):
    socket.push({"type": "response.done", "response": {"id": response_id, "status": status}})


async def collect_invoke(pipeline, request, responses):
    async for response in pipeline.invoke(request):
        responses.append(response)


def invoke_created(socket, transaction_id, response_id="invoke-response"):
    socket.push({"type": "response.created", "response": {
        "id": response_id, "metadata": {"transaction_id": transaction_id},
    }})


@pytest.mark.asyncio
async def test_ga_text_only_handshake_precedes_audio_and_preserves_identity(monkeypatch):
    socket = FakeWebSocket(acknowledge_start=False)
    pipeline, _, events, connections = create_pipeline(monkeypatch, sockets=[socket], instructions="Be brief")
    task = asyncio.create_task(pipeline.start_session(STSRequest(
        session_id="s", context_id="c", user_id="u", channel="web", transaction_id="unused",
    )))
    try:
        await until(lambda: socket.sent)
        assert events == [] and not task.done()
        with pytest.raises(RuntimeError, match="not ready"):
            await pipeline.process_audio_samples(b"\0\0", "s")
        event = socket.sent[0]
        assert event["type"] == "session.update"
        config = event["session"]
        assert config["type"] == "realtime"
        assert config["output_modalities"] == ["text"]
        assert config["instructions"] == "Be brief"
        assert config["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        vad = config["audio"]["input"]["turn_detection"]
        assert vad["type"] == "server_vad"
        assert vad["create_response"] is False and vad["interrupt_response"] is False
        assert "modalities" not in config and "input_audio_format" not in config
        socket.push({"type": "session.updated", "session": config})
        await task
        connected, = events
        assert connected.type == "connected"
        assert (connected.session_id, connected.context_id, connected.user_id, connected.transaction_id) == ("s", "c", "u", None)
        assert connected.metadata["channel"] == "web"
        assert "streaming" not in connected.metadata
        assert "client_barge_in_control" not in connected.metadata
        assert connected.metadata["input_sample_rate"] == 16000
        assert "pcm_format" not in connected.metadata
        assert connections[0][0].startswith("wss://api.openai.com/v1/realtime?")
        assert "model=gpt-realtime-2.1" in connections[0][0]
        assert connections[0][1]["additional_headers"] == {"Authorization": "Bearer test-key"}
    finally:
        await pipeline.shutdown()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_response_metadata_correlates_transaction_through_tts_and_final(monkeypatch):
    tts = FakeTTS(blocked=True)
    pipeline, sockets, events, _ = create_pipeline(monkeypatch, tts=tts)
    try:
        await pipeline.start_session(STSRequest(session_id="s", context_id="c", user_id="u", channel="web"))
        transaction_id = await begin_turn(sockets[0], events)
        text_delta(sockets[0], "こんにちは。")
        response_done(sockets[0])
        await until(lambda: tts.texts)
        assert not any(event.type == "final" for event in events)
        tts.release.set()
        await until(lambda: any(event.type == "final" for event in events))
        turn_events = [event for event in events if event.type in ("accepted", "start", "chunk", "final")]
        assert [event.type for event in turn_events] == ["accepted", "start", "chunk", "final"]
        assert all(event.transaction_id == transaction_id for event in turn_events)
        assert all((event.session_id, event.context_id, event.user_id) == ("s", "c", "u") for event in turn_events)
        assert all(event.metadata["channel"] == "web" for event in turn_events)
        assert "request_text" not in turn_events[1].metadata
        assert "recognized_text" not in turn_events[1].metadata
        assert turn_events[2].audio_data == b"RIFF" + "こんにちは。".encode()
    finally:
        await pipeline.shutdown()
    assert not tts.closed


@pytest.mark.asyncio
async def test_barge_in_cancels_tts_queue_provider_and_ignores_late_old_events(monkeypatch):
    tts = FakeTTS(blocked=True)
    pipeline, sockets, events, _ = create_pipeline(monkeypatch, tts=tts)
    try:
        await pipeline.start_session(STSRequest(session_id="s"))
        old_transaction = await begin_turn(sockets[0], events)
        text_delta(sockets[0], "古い発話。キュー中。末尾")
        await until(lambda: tts.texts)
        sockets[0].push({"type": "input_audio_buffer.speech_started", "item_id": "item-2"})
        await until(lambda: tts.cancelled and sent(sockets[0], "response.cancel"))
        assert sent(sockets[0], "response.cancel")[-1]["response_id"] == "response-1"
        assert any(event.type == "stop" for event in events)
        text_delta(sockets[0], "遅れて到着。")
        response_done(sockets[0])
        new_transaction = await begin_turn(sockets[0], events, response_id="response-2", item_id="item-2")
        assert new_transaction != old_transaction
        tts.release.set()
        text_delta(sockets[0], "新しい発話。", "response-2")
        response_done(sockets[0], "response-2")
        await until(lambda: any(event.type == "final" and event.transaction_id == new_transaction for event in events))
        assert tts.texts == ["古い発話。", "新しい発話。"]
        outputs = [event for event in events if event.type in ("chunk", "final")]
        assert all(event.transaction_id == new_transaction for event in outputs)
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", [8000, 16000, 24000, 44100, 48000, 192000])
async def test_input_resampling_preserves_chunk_boundaries_and_isolates_sessions(monkeypatch, sample_rate):
    pipeline, sockets, events, _ = create_pipeline(
        monkeypatch, sockets=[FakeWebSocket(), FakeWebSocket(), FakeWebSocket()], input_sample_rate=sample_rate,
    )
    assert pipeline.vad.sample_rate == sample_rate
    pcm = struct.pack("<12h", *range(-6000, 6000, 1000))
    other = struct.pack("<12h", *([7000] * 12))
    try:
        await asyncio.gather(*(pipeline.start_session(STSRequest(session_id=s)) for s in ("whole", "split", "other")))
        await pipeline.process_audio_samples(pcm, "whole")
        await pipeline.process_audio_samples(pcm[:1], "split")
        await pipeline.process_audio_samples(other[:5], "other")
        await asyncio.gather(
            pipeline.process_audio_samples(pcm[1:9], "split"),
            pipeline.process_audio_samples(other[5:], "other"),
        )
        await pipeline.process_audio_samples(pcm[9:], "split")
        audio = [b"".join(base64.b64decode(event["audio"]) for event in sent(socket, "input_audio_buffer.append")) for socket in sockets]
        assert audio[0] == audio[1]
        assert len(audio[0]) % 2 == 0 and audio[0] != audio[2]
        if sample_rate == 24000:
            assert audio[0] == pcm and audio[2] == other
        elif sample_rate < 24000:
            assert len(audio[0]) > len(pcm)
        else:
            assert len(audio[0]) < len(pcm)
        assert all(event["session"]["audio"]["input"]["format"]["rate"] == 24000 for socket in sockets for event in sent(socket, "session.update"))
        connected = [event for event in events if event.type == "connected"]
        assert len(connected) == 3
        assert all(event.metadata["input_sample_rate"] == sample_rate for event in connected)
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "cancel", "error", "transport", "reader"])
async def test_startup_failure_closes_transport_and_redacts_provider_details(monkeypatch, caplog, failure):
    caplog.set_level(logging.DEBUG, logger="examples.websocket.realtime.pipeline.realtime")

    class FailingReaderWebSocket(FakeWebSocket):
        def __aiter__(self):
            raise RuntimeError("private reader details test-key")

    socket_type = FailingReaderWebSocket if failure == "reader" else FakeWebSocket
    socket = socket_type(acknowledge_start=False)
    pipeline, _, events, _ = create_pipeline(monkeypatch, sockets=[socket], startup_timeout=0.03, instructions="private instructions")
    task = asyncio.create_task(pipeline.start_session(STSRequest(session_id="s")))
    try:
        await until(lambda: socket.sent)
        if failure == "cancel":
            task.cancel()
            error_type = asyncio.CancelledError
        elif failure == "error":
            socket.push({"type": "error", "error": {"code": "invalid_request", "message": "private provider details test-key"}})
            error_type = RuntimeError
        elif failure == "transport":
            socket.push(ConnectionError("private transport details test-key"))
            error_type = RuntimeError
        elif failure == "reader":
            error_type = RuntimeError
        else:
            error_type = TimeoutError
        with pytest.raises(error_type) as error:
            await task
        assert "private" not in str(error.value) and "test-key" not in str(error.value)
        assert not any(event.type in ("connected", "final") for event in events)
        assert not pipeline._sessions and socket.closed
        assert "private" not in repr(events) and "test-key" not in repr(events)
        assert "private" not in caplog.text and "test-key" not in caplog.text
    finally:
        await pipeline.shutdown()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_cancels_synthesis_and_is_idempotent_without_closing_injected_tts(monkeypatch):
    tts = FakeTTS(blocked=True)
    pipeline, sockets, events, _ = create_pipeline(monkeypatch, tts=tts)
    await pipeline.start_session(STSRequest(session_id="s"))
    await begin_turn(sockets[0], events)
    text_delta(sockets[0], "処理中。キュー中。末尾")
    await until(lambda: tts.texts)
    await asyncio.gather(pipeline.finalize("s"), pipeline.finalize("s"))
    await pipeline.finalize("s")
    await pipeline.shutdown()
    tts.release.set()
    assert tts.cancelled and not tts.closed
    assert sockets[0].closed and not pipeline._sessions
    assert not any(event.type in ("chunk", "final") for event in events)
    assert len([event for event in events if event.type == "session_closed"]) == 1


@pytest.mark.asyncio
async def test_text_invoke_streams_sentences_and_final_once_with_request_identifiers(monkeypatch):
    pipeline, sockets, events, _ = create_pipeline(monkeypatch)
    responses = []
    request = STSRequest(session_id="s", user_id="u", context_id="c", channel="web", text="挨拶してください。", transaction_id="text-turn")
    task = None
    try:
        await pipeline.start_session(STSRequest(session_id="s", user_id="u", context_id="c", channel="web"))
        task = asyncio.create_task(collect_invoke(pipeline, request, responses))
        await until(lambda: sent(sockets[0], "response.create"))
        item = sent(sockets[0], "conversation.item.create")[0]["item"]
        assert item["type"] == "message" and item["role"] == "user"
        assert item["content"] == [{"type": "input_text", "text": request.text}]
        assert sent(sockets[0], "response.create")[0]["response"] == {
            "output_modalities": ["text"], "metadata": {"transaction_id": request.transaction_id},
        }
        assert [event.type for event in events if event.transaction_id == request.transaction_id] == ["accepted"]
        invoke_created(sockets[0], request.transaction_id)
        text_delta(sockets[0], '<face name="joy" />こんにちは。次の文', "invoke-response")
        await until(lambda: any(response.type == "chunk" for response in responses))
        assert not task.done()
        assert pipeline.tts.texts == ["こんにちは。"]
        text_delta(sockets[0], "です。", "invoke-response")
        response_done(sockets[0], "invoke-response")
        await asyncio.wait_for(task, 1)
        assert [response.type for response in responses] == ["start", "chunk", "chunk", "final"]
        assert [response.voice_text for response in responses if response.type == "chunk"] == ["こんにちは。", "次の文です。"]
        assert responses[-1].text == '<face name="joy" />こんにちは。次の文です。'
        assert all((r.session_id, r.user_id, r.context_id, r.transaction_id) == ("s", "u", "c", "text-turn") for r in responses)
        assert all(r.metadata["channel"] == "web" for r in responses)
        assert [event.type for event in events if event.transaction_id == request.transaction_id] == ["accepted"]
        assert "s" in pipeline._sessions and not sockets[0].closed
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await pipeline.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript_status", ["completed", "failed"])
async def test_disabled_local_history_keeps_audio_turns_working_without_database(monkeypatch, tmp_path, transcript_status):
    def unexpected_database(*args, **kwargs):
        raise AssertionError("History-disabled sessions must not open a database")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "examples.websocket.realtime.pipeline.realtime.history.SQLiteContextManager",
        unexpected_database,
    )
    pipeline, sockets, events, _ = create_pipeline(
        monkeypatch, context_manager=None, context_db_path=None,
        transcription_model="gpt-4o-mini-transcribe",
    )
    try:
        await pipeline.start_session(STSRequest(session_id="s", context_id="existing-context"))
        assert pipeline.context_manager is None
        assert pipeline._sessions["s"].history is None
        await begin_turn(sockets[0], events)
        sockets[0].push({
            "type": f"conversation.item.input_audio_transcription.{transcript_status}",
            "item_id": "item-1", "transcript": "こんにちは。",
        })
        text_delta(sockets[0], "おはようございます。")
        response_done(sockets[0])
        await until(lambda: any(event.type == "final" for event in events))
        assert pipeline.tts.texts == ["おはようございます。"]
        assert not any(event.type == "error" for event in events)
    finally:
        await pipeline.shutdown()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_tts_failure_reports_sanitized_error_and_cleans_session(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="examples.websocket.realtime.pipeline.realtime")
    pipeline, sockets, events, _ = create_pipeline(monkeypatch, tts=FakeTTS(fail=True))
    try:
        await pipeline.start_session(STSRequest(session_id="s"))
        await begin_turn(sockets[0], events)
        text_delta(sockets[0], "発話。")
        await until(lambda: not pipeline._sessions)
        assert any(event.type == "error" for event in events)
        assert all(event.voice_text == event.metadata["error"] for event in events if event.type == "error")
        assert not any(event.type in ("chunk", "final") for event in events)
        assert sockets[0].closed
        assert "private synthesis" not in repr(events) and "private synthesis" not in caplog.text
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "transport", "queue"])
async def test_running_session_failures_cancel_synthesis_and_redact_errors(monkeypatch, caplog, failure):
    caplog.set_level(logging.DEBUG, logger="examples.websocket.realtime.pipeline.realtime")
    tts = FakeTTS(blocked=True)
    pipeline, sockets, events, _ = create_pipeline(monkeypatch, tts=tts, tts_queue_size=1)
    try:
        await pipeline.start_session(STSRequest(session_id="s"))
        await begin_turn(sockets[0], events)
        text_delta(sockets[0], "処理中。")
        await until(lambda: tts.texts)
        if failure == "provider":
            sockets[0].push({"type": "error", "error": {"code": "invalid_request", "message": "private provider details test-key"}})
        elif failure == "transport":
            sockets[0].push(ConnectionError("private transport details test-key"))
        else:
            text_delta(sockets[0], "キュー中。超過。")
        await until(lambda: not pipeline._sessions)
        assert tts.cancelled and sockets[0].closed
        assert any(event.type == "error" for event in events)
        assert any(event.type == "session_closed" for event in events)
        assert not any(event.type in ("chunk", "final") for event in events)
        assert all(value not in repr(events) + caplog.text for value in ("private ", "test-key"))
    finally:
        await pipeline.shutdown()
