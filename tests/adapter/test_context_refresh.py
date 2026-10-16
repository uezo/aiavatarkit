import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiavatar.adapter.channel_context_bridge.base import (
    ChannelUser,
    SQLiteChannelContextBridge,
    UserContext,
)
from aiavatar.adapter.websocket import server as websocket_module
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy


class FakeWebSocket:
    def __init__(self):
        self.requests = []
        self.responses = []

    async def receive_text(self):
        return json.dumps(self.requests.pop(0))

    async def send_text(self, message):
        await asyncio.sleep(0)
        self.responses.append(json.loads(message))


class Harness:
    def __init__(self, pipeline, server, bridge, clock):
        self.pipeline, self.server, self.bridge, self.clock = pipeline, server, bridge, clock
        self.socket, self.session = FakeWebSocket(), WebSocketSessionData()
        self.starts = []
        self.read_context = bridge.get_context
        bridge.get_context = AsyncMock(wraps=bridge.get_context)

        @pipeline.on_accepted
        async def wait_for_accepted_delivery(request):
            # Keep scheduling of the out-of-band event independent of refresh.
            if not server.can_handle(request.session_id):
                return

            async def accepted():
                while self.session.active_transaction_id != request.transaction_id:
                    await asyncio.sleep(0)

            await asyncio.wait_for(accepted(), timeout=2)

        @server.on_response
        async def record_start(response, _):
            if response.type == "start":
                self.starts.append(response)

    async def set_shared(self, context_id, expired=False):
        if context_id is None:
            await self.bridge.delete_context("app-user")
            return
        await self.pipeline.llm.context_manager.add_histories(
            context_id, [{"role": "user", "content": f"History for {context_id}"}], "dummy",
        )
        await self.bridge.upsert_context(UserContext(
            user_id="app-user", context_id=context_id,
            updated_at=datetime.now(timezone.utc) - timedelta(seconds=7200 if expired else 0),
        ))

    async def connect(self):
        await self.bridge.upsert_channel_user(ChannelUser(
            channel_id="device", channel_user_id="device-user", user_id="app-user",
        ))
        await self.set_shared("A")
        await self.invoke(type="start")

    async def invoke(self, **payload):
        self.socket.requests.append({
            "type": "invoke", "session_id": "session", "user_id": "device-user", **payload,
        })
        await self.server.process_websocket(self.socket, self.session)
        await asyncio.sleep(0)

    async def voice(self, text="hello"):
        await self.pipeline.on_speech_detected(b"\0\0", text, None, 0.1, "session")
        await asyncio.sleep(0)

    def context(self):
        return self.pipeline.vad.get_session_data("session", "context_id")


@asynccontextmanager
async def running_server(tmp_path, monkeypatch, *, refresh=300):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(websocket_module, "monotonic", lambda: clock.now)
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(), stt=SpeechRecognizerDummy(),
        llm=LLMServiceDummy(
            response_text="response", wait_sec=0,
            db_connection_str=str(tmp_path / "history.db"),
        ),
        tts=SpeechSynthesizerDummy(synthesized_bytes=b""),
        db_connection_str=str(tmp_path / "pipeline.db"),
        voice_recorder_dir=str(tmp_path / "voices"), voice_recorder_enabled=False,
    )
    server = AIAvatarWebSocketServer(sts=pipeline, channel="device")
    bridge = SQLiteChannelContextBridge(str(tmp_path / "bridge.db"))
    bridge.bind(server, "device", refresh_context_after_idle_seconds=refresh)
    try:
        yield Harness(pipeline, server, bridge, clock)
    finally:
        await pipeline.finalize("session")
        await pipeline.shutdown()
        await pipeline.stt.close()
        await pipeline.tts.close()
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["voice", "invoke", "explicit_invoke"])
async def test_idle_refresh_uses_shared_context_and_mapped_user(tmp_path, monkeypatch, route):
    async with running_server(tmp_path, monkeypatch) as h:
        await h.connect()
        await h.voice("first turn")
        await h.set_shared("B")
        h.clock.now += 300
        h.bridge.get_context.reset_mock()

        if route == "voice":
            await h.voice("resumed turn")
        else:
            payload = {"context_id": "A"} if route == "explicit_invoke" else {}
            await h.invoke(text="resumed turn", **payload)

        assert h.starts[-1].context_id == "B"
        assert h.starts[-1].user_id == "app-user"
        assert h.context() == "B"
        assert (await h.read_context("app-user")).context_id == "B"
        h.bridge.get_context.assert_awaited_once_with("app-user")


@pytest.mark.asyncio
@pytest.mark.parametrize(("delay", "expected"), [(0, "A"), (299, "A"), (7200, "B")])
async def test_first_invocation_refreshes_only_after_connection_idle(tmp_path, monkeypatch, delay, expected):
    async with running_server(tmp_path, monkeypatch) as h:
        await h.connect()
        await h.set_shared("B")
        h.clock.now += delay
        h.bridge.get_context.reset_mock()

        await h.voice("first actual conversation")

        assert h.context() == expected
        if expected == "B":
            h.bridge.get_context.assert_awaited_once_with("app-user")
        else:
            h.bridge.get_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["absent", "expired"])
async def test_no_valid_shared_mapping_issues_new_context(tmp_path, monkeypatch, missing):
    async with running_server(tmp_path, monkeypatch) as h:
        await h.connect()
        await h.voice()
        await h.set_shared("A" if missing == "expired" else None, expired=True)
        h.clock.now += 300

        await h.invoke(text="new conversation", context_id="A")

        context = h.starts[-1].context_id
        assert context and context != "A"
        assert h.context() == context
        assert (await h.read_context("app-user")).context_id == context


@pytest.mark.asyncio
async def test_continuous_conversation_does_not_read_bridge(tmp_path, monkeypatch):
    async with running_server(tmp_path, monkeypatch) as h:
        await h.connect()
        await h.voice()
        await h.set_shared("B")
        h.bridge.get_context.reset_mock()
        h.clock.now += 299
        await h.voice("still the same conversation")
        assert h.context() == "A"
        h.bridge.get_context.assert_not_awaited()

        await h.set_shared("B")
        h.clock.now += 300
        await h.voice("after the threshold")
        assert h.context() == "B"
        h.bridge.get_context.assert_awaited_once_with("app-user")


@pytest.mark.asyncio
async def test_disabled_refresh_preserves_existing_behavior(tmp_path, monkeypatch):
    async with running_server(tmp_path, monkeypatch, refresh=None) as h:
        await h.connect()
        await h.voice()
        await h.set_shared("B")
        h.clock.now += 7200
        h.bridge.get_context.reset_mock()
        await h.voice("original behavior")
        assert h.context() == "A"
        assert (await h.read_context("app-user")).context_id == "A"
        h.bridge.get_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_updates_response_time_but_audio_frames_do_not(tmp_path, monkeypatch):
    async with running_server(tmp_path, monkeypatch) as h:
        await h.connect()
        assert h.session.last_response_at == h.clock.now
        h.clock.now += 100
        await h.voice()
        response_at = h.clock.now
        assert h.session.last_response_at == response_at

        h.clock.now += 299
        await h.invoke(type="data", audio_data="AAA=")
        assert h.session.last_response_at == response_at
        await h.set_shared("B")
        h.bridge.get_context.reset_mock()
        h.clock.now += 1
        await h.voice()
        assert h.context() == "B"
        assert h.session.last_response_at == h.clock.now
        h.bridge.get_context.assert_awaited_once_with("app-user")


@pytest.mark.asyncio
async def test_lookup_cannot_recreate_disconnected_session(tmp_path, monkeypatch):
    async with running_server(tmp_path, monkeypatch) as h:
        await h.connect()
        await h.voice()
        await h.set_shared("B")
        h.clock.now += 300
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_lookup(user_id):
            started.set()
            await release.wait()
            return await h.read_context(user_id)

        h.bridge.get_context = AsyncMock(side_effect=blocked_lookup)
        task = asyncio.create_task(h.voice())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            await h.pipeline.finalize("session")
            h.server.sessions.pop("session")
            h.server.websockets.pop("session")
            release.set()
            await asyncio.wait_for(task, timeout=2)
            assert "session" not in h.pipeline.vad._session_data
            assert (await h.read_context("app-user")).context_id == "B"
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("-inf"), float("nan")])
async def test_refresh_threshold_must_be_positive_and_finite(tmp_path, monkeypatch, seconds):
    async with running_server(tmp_path, monkeypatch, refresh=None) as h:
        with pytest.raises(ValueError):
            h.bridge.bind(h.server, "device", refresh_context_after_idle_seconds=seconds)
