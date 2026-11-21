"""Mute on barge-in in the WebSocket adapter.

No providers, audio models, network, or files outside tmp_path. The end-to-end test
uses STSPipeline with dummy components and SQLite databases in tmp_path.
"""

import asyncio
import base64
import io
import json
import wave

import pytest

from aiavatar.adapter.models import AIAvatarResponse
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData
from aiavatar.sts.llm import LLMResponse, LLMServiceDummy
from aiavatar.sts.models import STSResponse
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy


SESSION_ID = "session-1"


class FakeVAD:
    def __init__(self):
        self.recording = False
        self.recording_started_handlers = []
        self.data = {}

    def on_recording_started(self, func):
        self.recording_started_handlers.append(func)
        return func

    def on_voiced(self, func):
        return func

    async def process_samples(self, samples, session_id):
        return self.recording

    def set_session_data(self, session_id, key, value, create_session=False):
        self.data.setdefault(session_id, {})[key] = value

    def get_session_data(self, session_id, key):
        return self.data.get(session_id, {}).get(key)

    async def barge_in(self, session_id=SESSION_ID):
        for handler in self.recording_started_handlers:
            await handler(session_id)


class FakePipeline:
    def __init__(self, vad):
        self.vad = vad

    def add_response_handler(self, handler):
        pass

    def on_accepted(self, func=None, *, channels=None):
        if func is not None:
            return func
        return lambda callback: callback


class FakeWebSocket:
    def __init__(self, *requests):
        self.requests = [json.dumps(request) for request in requests]
        self.sent_messages = []
        self.on_send = None

    async def receive_text(self):
        return self.requests.pop(0)

    async def send_text(self, message):
        self.sent_messages.append(message)
        if self.on_send is not None:
            await self.on_send(AIAvatarResponse.model_validate_json(message))


def connect(server):
    session_data = WebSocketSessionData()
    session_data.id = SESSION_ID
    websocket = FakeWebSocket()
    server.sessions[SESSION_ID] = session_data
    server.websockets[SESSION_ID] = websocket
    return session_data, websocket


@pytest.fixture
def make_server():
    def create(mute_on_barge_in=True, response_audio_chunk_size=0):
        vad = FakeVAD()
        server = AIAvatarWebSocketServer(
            sts=FakePipeline(vad),
            mute_on_barge_in=mute_on_barge_in,
            response_audio_chunk_size=response_audio_chunk_size,
        )
        session_data, websocket = connect(server)
        return server, vad, session_data, websocket

    return create


def chunk(transaction_id, text, audio_data=None):
    # Complete-audio mode sends the bytes as they are; they need not be a WAV.
    return STSResponse(type="chunk", session_id=SESSION_ID, transaction_id=transaction_id,
                       text=text, voice_text=text, audio_data=audio_data or b"audio:" + text.encode(),
                       metadata={"is_first_chunk": False, "is_guardrail_triggered": False})


def final(transaction_id, text):
    return STSResponse(type="final", session_id=SESSION_ID, transaction_id=transaction_id,
                       text=text, voice_text=text)


def sent(websocket):
    messages = [AIAvatarResponse.model_validate_json(m) for m in websocket.sent_messages]
    return [
        (m.type, m.transaction_id, m.text, bool((m.metadata or {}).get("interrupted")))
        for m in messages
    ]


async def begin(server, transaction_id):
    await server.handle_response(STSResponse(
        type="accepted", session_id=SESSION_ID, transaction_id=transaction_id,
        metadata={"block_barge_in": False}))
    await server.handle_response(STSResponse(
        type="start", session_id=SESSION_ID, transaction_id=transaction_id,
        metadata={"request_text": "question"}))


async def feed_audio(server, session_data):
    socket = FakeWebSocket({
        "type": "data", "session_id": SESSION_ID,
        "audio_data": base64.b64encode(b"\x00\x00").decode(),
    })
    await server.process_websocket(socket, session_data)


@pytest.mark.asyncio
async def test_response_arriving_during_barge_in_is_muted_and_closed(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")
    vad.recording = True
    await vad.barge_in()

    await server.handle_response(chunk("A", "unheard"))
    await server.handle_response(final("A", "unheard [operation:hangup]"))

    assert sent(websocket) == [
        ("accepted", "A", None, False),
        ("start", "A", None, False),
        ("stop", None, None, False),
        ("final", "A", "", True),
    ]


@pytest.mark.asyncio
async def test_speaking_response_does_not_resume_after_barge_in(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")
    await server.handle_response(chunk("A", "heard"))
    vad.recording = True
    await vad.barge_in()

    await server.handle_response(chunk("A", "unheard"))
    await server.handle_response(final("A", "heardunheard"))

    assert sent(websocket) == [
        ("accepted", "A", None, False),
        ("start", "A", None, False),
        ("chunk", "A", "heard", False),
        ("stop", None, None, False),
        ("final", "A", "", True),
    ]


@pytest.mark.asyncio
async def test_only_responses_accepted_before_the_user_stops_speaking_are_muted(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")
    vad.recording = True
    await vad.barge_in()

    await feed_audio(server, session_data)  # The user is still speaking.
    await begin(server, "N")
    await server.handle_response(chunk("N", "old"))

    vad.recording = False
    await feed_audio(server, session_data)  # The user's speech has ended.
    await begin(server, "B")
    await server.handle_response(chunk("B", "answer"))
    await server.handle_response(final("B", "answer"))

    assert [message[2] for message in sent(websocket) if message[0] == "chunk"] == ["answer"]
    assert ("start", "N", None, False) in sent(websocket)
    assert sent(websocket)[-1] == ("final", "B", "answer", False)


@pytest.mark.asyncio
async def test_new_turn_response_is_muted_by_the_next_barge_in(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")
    vad.recording = True
    await vad.barge_in()
    vad.recording = False
    await feed_audio(server, session_data)

    await begin(server, "B")
    await server.handle_response(chunk("B", "first"))
    vad.recording = True
    await vad.barge_in()
    await server.handle_response(chunk("B", "second"))

    assert [message[2] for message in sent(websocket) if message[0] == "chunk"] == ["first"]
    assert session_data.muted_transaction_id == "B"


@pytest.mark.asyncio
async def test_muted_response_is_closed_once(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")
    vad.recording = True
    await vad.barge_in()
    await server.handle_response(final("A", "unheard"))
    vad.recording = False
    await feed_audio(server, session_data)

    # The next turn's accepted does not close the muted response again.
    await begin(server, "B")
    await server.handle_response(chunk("A", "late"))
    await server.handle_response(final("B", "answer"))

    finals = [message for message in sent(websocket) if message[0] == "final"]
    assert finals == [("final", "A", "", True), ("final", "B", "answer", False)]


@pytest.mark.asyncio
async def test_muted_response_closed_by_a_newer_accepted_gets_one_final(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")
    vad.recording = True
    await vad.barge_in()
    vad.recording = False
    await feed_audio(server, session_data)  # The user's speech has ended.
    release = asyncio.Event()

    async def hold_interrupted_final(message):
        if message.type == "final" and message.transaction_id == "A":
            await release.wait()

    websocket.on_send = hold_interrupted_final
    # The next request is accepted while the muted response is ending.
    switching = asyncio.create_task(server.handle_response(STSResponse(
        type="accepted", session_id=SESSION_ID, transaction_id="B",
        metadata={"block_barge_in": False})))
    await asyncio.sleep(0)
    finishing = asyncio.create_task(server.handle_response(final("A", "unheard")))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(switching, finishing)

    assert [message for message in sent(websocket) if message[0] == "final"] == [("final", "A", "", True)]


@pytest.mark.asyncio
async def test_response_muted_while_handlers_run_is_not_sent(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")

    @server.on_response
    async def barge_in_during_final(aiavatar_response, response):
        if response.type == "final":
            vad.recording = True
            await vad.barge_in()

    await server.handle_response(final("A", "unheard [vision:camera]"))

    assert sent(websocket) == [
        ("accepted", "A", None, False),
        ("start", "A", None, False),
        ("stop", None, None, False),
        ("final", "A", "", True),
    ]


@pytest.mark.asyncio
async def test_streamed_audio_stops_when_barge_in_mutes_it_mid_chunk(make_server):
    server, vad, session_data, websocket = make_server(response_audio_chunk_size=512)
    await begin(server, "A")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 2048)  # Eight 512-byte PCM chunks
    tasks = []

    async def barge_in_after_descriptor(message):
        if message.type == "chunk" and (message.metadata or {}).get("pcm_format") and not message.audio_data:
            vad.recording = True
            tasks.append(asyncio.create_task(vad.barge_in()))
            await asyncio.sleep(0)

    websocket.on_send = barge_in_after_descriptor
    await server.handle_response(chunk("A", "long", audio_data=buffer.getvalue()))
    await asyncio.gather(*tasks)

    messages = [AIAvatarResponse.model_validate_json(m) for m in websocket.sent_messages]
    assert [m.type for m in messages] == ["accepted", "start", "chunk", "stop"]
    assert messages[2].audio_data is None  # Only the PCM descriptor was sent.


@pytest.mark.asyncio
async def test_barge_in_on_another_session_is_ignored(make_server):
    server, vad, session_data, websocket = make_server()
    await begin(server, "A")

    await vad.barge_in("another-session")
    await server.handle_response(chunk("A", "one"))

    assert session_data.muted_transaction_id is None
    assert sent(websocket)[-1] == ("chunk", "A", "one", False)


@pytest.mark.asyncio
async def test_without_mute_barge_in_changes_nothing(make_server):
    server, vad, session_data, websocket = make_server(mute_on_barge_in=False)
    await begin(server, "A")
    vad.recording = True
    await vad.barge_in()
    await feed_audio(server, session_data)

    await server.handle_response(chunk("A", "one"))
    await server.handle_response(final("A", "one"))

    assert vad.recording_started_handlers == []
    assert sent(websocket)[2:] == [("chunk", "A", "one", False), ("final", "A", "one", False)]


class GatedLLM(LLMServiceDummy):
    """Streams one sentence, then waits for `release` before streaming another."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.release = asyncio.Event()
        self.received_messages = []

    async def get_llm_stream_response(self, context_id, user_id, messages, system_prompt_params=None,
                                      tools=None, inline_llm_params=None, session_id=None, channel=None):
        self.received_messages.append([(m["role"], m["content"]) for m in messages])
        yield LLMResponse(context_id=context_id, text="First. ", voice_text="First. ")
        await self.release.wait()
        yield LLMResponse(context_id=context_id, text="Second.", voice_text="Second.")


@pytest.mark.asyncio
async def test_muted_response_is_generated_and_kept_in_the_context(tmp_path):
    llm = GatedLLM(db_connection_str=str(tmp_path / "context.db"))
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=llm,
        tts=SpeechSynthesizerDummy(synthesized_bytes=b"voice"),
        db_connection_str=str(tmp_path / "pipeline.db"),
        voice_recorder_dir=str(tmp_path / "voices"),
        voice_recorder_enabled=False,
        use_invoke_queue=True,
    )
    server = AIAvatarWebSocketServer(sts=pipeline, mute_on_barge_in=True)
    session_data, websocket = connect(server)
    pipeline.vad.set_session_data(SESSION_ID, "channel", "websocket", True)
    first_chunk = asyncio.Event()

    async def notify_first_chunk(message):
        if message.type == "chunk":
            first_chunk.set()

    websocket.on_send = notify_first_chunk
    try:
        turn = asyncio.create_task(
            pipeline.on_speech_detected(b"", "question", {}, 2.0, SESSION_ID))
        await asyncio.wait_for(first_chunk.wait(), 2)

        # The user keeps speaking while the first answer is generated.
        for handler in pipeline.vad._on_recording_started:
            await handler(SESSION_ID)
        llm.release.set()
        await asyncio.wait_for(turn, 2)
        await feed_audio(server, session_data)  # The user's speech has ended.
        await asyncio.wait_for(
            pipeline.on_speech_detected(b"", "next question", {}, 2.0, SESSION_ID), 2)
        await asyncio.sleep(0)
    finally:
        workers = list(pipeline._invoke_workers.values())
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await pipeline.shutdown()
        await pipeline.stt.close()
        await pipeline.tts.close()

    def summary(message):
        return (message["type"], message["transaction_id"], message["text"],
                (message.get("metadata") or {}).get("interrupted"))

    messages = [json.loads(message) for message in websocket.sent_messages]
    first_chunk_index = next(index for index, message in enumerate(messages) if message["type"] == "chunk")
    first_turn = messages[first_chunk_index]["transaction_id"]

    # Only the first sentence of the first answer is delivered: the barge-in stop
    # follows it, and the answer ends with an empty final.
    assert summary(messages[first_chunk_index]) == ("chunk", first_turn, "First. ", None)
    assert [summary(message) for message in messages[first_chunk_index + 1:first_chunk_index + 3]] == [
        ("stop", None, None, None),
        ("final", first_turn, "", True),
    ]
    later = messages[first_chunk_index + 3:]
    assert all(message["transaction_id"] != first_turn for message in later)

    # The first answer is still generated in full, so the next request sees it in the context.
    assert llm.received_messages[-1] == [
        ("user", "question"),
        ("assistant", "First. Second."),
        ("user", "next question"),
    ]
    next_turn = next(message["transaction_id"] for message in later if message["type"] == "accepted")
    assert [message["text"] for message in later
            if message["type"] == "chunk" and message["transaction_id"] == next_turn] == ["First. ", "Second."]
    assert [summary(message) for message in later if message["type"] == "final"] == [
        ("final", next_turn, "First. Second.", None),
    ]
