"""Bridge hooks, in-memory fakes only: no providers, models or databases."""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from aiavatar.adapter.models import AIAvatarResponse
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData
from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from examples.nod import NodDecision
from examples.nod.integrations.aiavatar import NodPipelineBridge


class Engine:
    candidates = [{"id": "neutral", "phrase": "うん"}]

    def __init__(self):
        self.inputs = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def decide(self, text):
        self.inputs.append(text)
        self.started.set()
        await self.release.wait()
        return NodDecision("neutral", "うん", "stop")


class Socket:
    def __init__(self):
        self.messages = []
        self.sent = asyncio.Event()

    async def send_text(self, data):
        self.messages.append(json.loads(data))
        self.sent.set()


@pytest_asyncio.fixture
async def rig():
    connection = WebSocketSessionData()
    connection.id = "s"
    socket, engine = Socket(), Engine()
    async def synthesize(text):
        return b"fake-wav"
    app = SimpleNamespace(sessions={"s": connection}, websockets={"s": socket},
                          sts=SimpleNamespace(tts=SimpleNamespace(synthesize=synthesize)))
    bridge = NodPipelineBridge(app, engine, min_interval=0, auto_register_hooks=False)
    await bridge.prepare_audio()
    await bridge.open_session(connection)
    try:
        yield bridge, app, connection, socket, engine
    finally:
        await bridge.close()


def partial(bridge, identifier="a", text="週末に子供と出かけて、"):
    recording = SimpleNamespace(session_id="s", recording_id=identifier)
    bridge.on_partial(text, recording)
    return recording


def accepted(bridge, identifier="a", transaction="t1", text="週末に子供と出かけた。"):
    request = STSRequest(session_id="s", transaction_id=transaction, text=text,
                         metadata={"recording_id": identifier})
    bridge.on_accepted(request)
    return request


async def settled(bridge):
    tasks = [task for state in bridge._sessions.values() for task in state.tasks]
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)


@pytest.fixture
def hook_app():
    # Use real registration methods without constructing providers or loading VAD models.
    app = AIAvatarWebSocketServer.__new__(AIAvatarWebSocketServer)
    app.sessions, app.websockets = {}, {}
    app._on_session_start_handlers = []
    app._on_response_handlers = []
    app._on_disconnect = None
    app.sts = STSPipeline.__new__(STSPipeline)
    app.sts._on_accepted_handlers = []
    app.sts.vad = SileroStreamSpeechDetector.__new__(SileroStreamSpeechDetector)
    app.sts.vad._on_speech_detecting = []
    app.sts.tts = SimpleNamespace(synthesize=AsyncMock(return_value=b"fake-wav"))
    return app


@pytest.mark.asyncio
async def test_auto_registered_hooks_drive_history_cancellation_and_disconnect(hook_app):
    app, engine, socket = hook_app, Engine(), Socket()
    connection = WebSocketSessionData()
    connection.id = "s"
    app.sessions["s"], app.websockets["s"] = connection, socket
    bridge = NodPipelineBridge(app, engine, min_interval=0)
    try:
        app.sts.tts.synthesize.assert_not_awaited()
        assert not bridge._sessions and not engine.inputs
        await bridge.prepare_audio()
        for handler in app._on_session_start_handlers:
            await handler(SimpleNamespace(session_id="s"), connection)

        engine.release.clear()
        recording = SimpleNamespace(session_id="s", recording_id="a")
        await asyncio.wait_for(app.sts.vad._execute_on_speech_detecting(
            "週末に子供と出かけて、", recording), 1)
        await asyncio.wait_for(engine.started.wait(), 1)
        request = STSRequest(session_id="s", transaction_id="t1",
                             text="週末に子供と出かけた。", metadata={"recording_id": "a"})
        await app.sts._execute_hooks(app.sts._on_accepted_handlers, None, request)
        await settled(bridge)
        assert not socket.messages

        response = AIAvatarResponse(type="final", session_id="s", voice_text="楽しそう。")
        sts_response = STSResponse(type="final", session_id="s", transaction_id="t1")
        for handler in app._on_response_handlers:
            await handler(response, sts_response)
        engine.release.set()
        recording.recording_id = "b"
        await app.sts.vad._execute_on_speech_detecting("それからね、", recording)
        await settled(bridge)
        assert len(socket.messages) == 1
        assert socket.messages[0]["metadata"]["recording_id"] == "b"
        assert "週末に子供と出かけた。" in engine.inputs[-1]
        assert "AI：楽しそう。" in engine.inputs[-1]

        engine.started.clear()
        engine.release.clear()
        recording.recording_id = "c"
        await app.sts.vad._execute_on_speech_detecting("あとね、", recording)
        await asyncio.wait_for(engine.started.wait(), 1)
        pending = bridge._sessions["s"].pending
        await app._on_disconnect(connection)
        assert pending.done() and not bridge._sessions
        assert len(socket.messages) == 1
    finally:
        await bridge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_register_hooks", [True, False])
async def test_registration_preserves_additive_hooks_and_disconnect_contract(
        hook_app, auto_register_hooks):
    app = hook_app
    existing = AsyncMock()
    app.on_session_start(existing)
    app.sts.vad.on_speech_detecting(existing)
    app.sts.on_accepted(existing)
    app.on_response(existing)
    app.on_disconnect(existing)
    bridge = NodPipelineBridge(app, Engine(), auto_register_hooks=auto_register_hooks)
    try:
        for handlers in (app._on_session_start_handlers, app.sts.vad._on_speech_detecting,
                         app._on_response_handlers):
            assert handlers[0] is existing
            assert len(handlers) == 1 + auto_register_hooks
        assert app.sts._on_accepted_handlers[0] == (None, existing)
        assert len(app.sts._on_accepted_handlers) == 1 + auto_register_hooks
        assert (app._on_disconnect is existing) is (not auto_register_hooks)
        # Later registration still replaces the single callback, as documented.
        later = AsyncMock()
        app.on_disconnect(later)
        connection = WebSocketSessionData()
        await app._on_disconnect(connection)
        later.assert_awaited_once_with(connection)
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_partial_schedules_and_emits_nod_metadata_with_cached_audio(rig):
    bridge, _, _, socket, engine = rig
    engine.release.clear()
    partial(bridge)
    await asyncio.wait_for(engine.started.wait(), 1)
    assert not socket.messages
    partial(bridge, text="週末に子供と出かけて、楽しかった")
    assert len(engine.inputs) == 1
    engine.release.set()
    await settled(bridge)
    message, = socket.messages
    assert message["metadata"]["nod"] is True
    assert message["metadata"]["recording_id"] == "a"
    assert message["metadata"]["nod_id"]
    assert base64.b64decode(message["audio_data"]) == b"fake-wav"
    assert message["text"] == message["voice_text"] == ""
    assert "<nod_assistant>うん</nod_assistant>" in bridge._sessions["s"].nod.build_input("a")


@pytest.mark.asyncio
async def test_accepted_cancels_pending_and_does_not_reopen_late_partial(rig):
    bridge, _, _, socket, engine = rig
    engine.release.clear()
    partial(bridge)
    await asyncio.wait_for(engine.started.wait(), 1)
    accepted(bridge)
    engine.release.set()
    await settled(bridge)
    partial(bridge, text="遅れてきた最終認識")
    await settled(bridge)
    assert not socket.messages
    assert len(engine.inputs) == 1


@pytest.mark.asyncio
async def test_accepted_before_task_starts_also_cancels_it(rig):
    bridge, _, _, socket, engine = rig
    partial(bridge)
    accepted(bridge)
    await settled(bridge)
    assert not engine.inputs and not socket.messages


@pytest.mark.asyncio
async def test_accepted_old_recording_preserves_new_recording(rig):
    bridge, _, _, socket, engine = rig
    engine.release.clear()
    partial(bridge, "a")
    await asyncio.wait_for(engine.started.wait(), 1)
    partial(bridge, "b", "それから遊園地にも行って、")
    accepted(bridge, "a")
    engine.release.set()
    await settled(bridge)
    assert [m["metadata"]["recording_id"] for m in socket.messages] == ["b"]


@pytest.mark.asyncio
async def test_vad_reset_after_partial_does_not_cancel_nod(rig):
    bridge, _, _, socket, engine = rig
    engine.release.clear()
    recording = partial(bridge)
    await asyncio.wait_for(engine.started.wait(), 1)
    recording.recording_id = None
    engine.release.set()
    await settled(bridge)
    assert socket.messages[0]["metadata"]["recording_id"] == "a"


@pytest.mark.asyncio
async def test_acceptance_while_waiting_for_send_lock_prevents_delivery(rig):
    bridge, _, connection, socket, engine = rig
    await connection.send_lock.acquire()
    partial(bridge)
    await asyncio.wait_for(engine.started.wait(), 1)
    accepted(bridge)
    connection.send_lock.release()
    await settled(bridge)
    assert not socket.messages


@pytest.mark.asyncio
async def test_close_cancels_and_does_not_recreate_session(rig):
    bridge, _, connection, socket, engine = rig
    engine.release.clear()
    partial(bridge)
    await asyncio.wait_for(engine.started.wait(), 1)
    await bridge.close_session(connection)
    partial(bridge, "b")
    accepted(bridge)
    engine.release.set()
    assert not socket.messages and not bridge._sessions


@pytest.mark.asyncio
async def test_replaced_socket_does_not_receive_old_nod(rig):
    bridge, app, _, socket, engine = rig
    engine.release.clear()
    partial(bridge)
    await asyncio.wait_for(engine.started.wait(), 1)
    replacement = WebSocketSessionData()
    replacement.id = "s"
    app.sessions["s"] = replacement
    engine.release.set()
    await settled(bridge)
    assert not socket.messages


@pytest.mark.asyncio
async def test_chunks_final_and_sent_nods_form_history_without_duplicates(rig):
    bridge, _, _, _, engine = rig
    partial(bridge)
    await settled(bridge)
    accepted(bridge, text="週末に子供と出かけて、楽しかった。")
    def response(kind, text, transaction="t1", **metadata):
        r = STSResponse(type=kind, session_id="s", transaction_id=transaction,
                        voice_text=text, metadata=metadata)
        bridge.on_response(AIAvatarResponse(type=kind, session_id="s", voice_text=text,
                                            metadata=metadata), r)
    response("chunk", "それは")
    response("chunk", "楽しそう。")
    response("chunk", "うん", nod=True)
    response("final", "それは楽しそう。")
    response("chunk", "古い応答", transaction="old")
    partial(bridge, "b", "まあね")
    await settled(bridge)
    text = engine.inputs[-1]
    assert text.count("AI：それは楽しそう。") == 1
    assert "古い応答" not in text
    assert "<nod_assistant>うん</nod_assistant>" in text


@pytest.mark.asyncio
async def test_accepted_without_partial_keeps_final_input(rig):
    bridge, _, _, _, engine = rig
    accepted(bridge)
    partial(bridge, "b", "それからね")
    await settled(bridge)
    assert "週末に子供と出かけた。" in engine.inputs[-1]


@pytest.mark.asyncio
async def test_response_stop_uses_original_message_without_nod_configuration():
    server = AIAvatarWebSocketServer.__new__(AIAvatarWebSocketServer)
    connection = WebSocketSessionData()
    connection.id = "s"
    socket = Socket()
    server.sessions, server.websockets = {"s": connection}, {"s": socket}
    await server.stop_response("s", "ctx")
    message, = socket.messages
    assert message["type"] == "stop"
    assert message["session_id"] == "s" and message["context_id"] == "ctx"
    assert message["metadata"] is None


@pytest.mark.asyncio
async def test_final_only_old_input_is_remembered_without_ending_new_input(rig):
    bridge, _, _, socket, engine = rig
    engine.release.clear()
    partial(bridge, "b", "次の話なんだけど、")
    accepted(bridge, "a", text="前の発言")
    engine.release.set()
    await settled(bridge)
    assert socket.messages[0]["metadata"]["recording_id"] == "b"
    assert "ユーザー：前の発言" in engine.inputs[0]
    assert "【今回のユーザー発言】\nユーザー：次の話なんだけど、" in engine.inputs[0]
