import asyncio
import base64
import json

import pytest

from aiavatar.adapter.asterisk.protocol import AsteriskProtocolError
from aiavatar.adapter.asterisk.server import AIAvatarAsteriskServer
from aiavatar.sts.models import STSResponse

from .conftest import DummySTS, FakeMediaWebSocket


def _media_start(
    session_id="call-1",
    *,
    channel_id="media-1",
    connection_id="connection-1",
):
    return json.dumps({
        "event": "MEDIA_START",
        "connection_id": connection_id,
        "channel": "WebSocket/media",
        "channel_id": channel_id,
        "format": "slin16",
        "optimal_frame_size": 640,
        "ptime": 20,
        "channel_variables": {
            "AIAVATAR_SESSION_ID": session_id,
            "AIAVATAR_CALLER_NUMBER": "0312345678",
        },
    })


def _response(**values):
    if values.get("session_id") == "call-1":
        values["session_id"] = "media-1"
    return STSResponse(**values)


class ManagerStub:
    def __init__(self):
        self.media_connections = []
        self.hangups = []

    async def media_connected(self, session_id, channel_id):
        self.media_connections.append((session_id, channel_id))

    async def hangup(self, session_id):
        self.hangups.append(session_id)


def _register_media_session(server, session_id="call-1", channel_id="media-1"):
    return server.register_session(
        session_id,
        media_channel_id=channel_id,
    )


@pytest.mark.asyncio
async def test_media_start_and_binary_audio_reach_vad():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    _register_media_session(server)
    websocket = FakeMediaWebSocket()

    session = await server.process_control_frame(websocket, _media_start())
    await server.process_binary_frame(websocket, session, b"\x01\x00" * 320)

    assert server.can_handle("media-1")
    assert sts.vad.samples == [("media-1", b"\x01\x00" * 320)]
    assert sts.vad.session_data["media-1"]["user_id"] == "0312345678"


@pytest.mark.asyncio
async def test_media_start_notifies_bound_call_manager_when_ready():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    manager = ManagerStub()
    server.bind_call_manager(manager)
    server.register_session(
        "call-1",
        media_channel_id="media-1",
        user_id="user-1",
        context_id="context-1",
    )

    await server.process_control_frame(FakeMediaWebSocket(), _media_start())

    assert manager.media_connections == [("call-1", "media-1")]
    assert sts.vad.session_data["media-1"]["user_id"] == "user-1"
    assert sts.vad.session_data["media-1"]["context_id"] == "context-1"


@pytest.mark.asyncio
async def test_xoff_blocks_binary_output_until_xon():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(websocket, _media_start())
    await server.process_control_frame(
        websocket,
        json.dumps({"event": "MEDIA_XOFF", "channel_id": "media-1"}),
        session,
    )

    send_task = asyncio.create_task(
        server.send_voice("call-1", audio_data=b"\x01\x00" * 320)
    )
    await asyncio.sleep(0)
    assert not websocket.binary_messages

    await server.process_control_frame(
        websocket,
        json.dumps({"event": "MEDIA_XON", "channel_id": "media-1"}),
        session,
    )
    await send_task
    assert websocket.binary_messages
    assert all(len(chunk) <= 65_500 for chunk in websocket.binary_messages)
    assert session.buffering is True


@pytest.mark.asyncio
async def test_slow_tts_does_not_send_to_replacement_websocket():
    synthesis_started = asyncio.Event()
    release_synthesis = asyncio.Event()

    class SlowTTS:
        async def synthesize(self, text):
            synthesis_started.set()
            await release_synthesis.wait()
            return b"\x01\x00" * 320

    sts = DummySTS()
    sts.tts = SlowTTS()
    server = AIAvatarAsteriskServer(sts=sts)
    server.bind_call_manager(ManagerStub())
    _register_media_session(server)
    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())

    send = asyncio.create_task(server.send_voice("call-1", text="hello"))
    await synthesis_started.wait()
    await server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    )
    server.register_session("call-1", media_channel_id="media-2")
    new_websocket = FakeMediaWebSocket()
    await server.process_control_frame(
        new_websocket,
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )

    release_synthesis.set()
    await send

    assert new_websocket.binary_messages == []


@pytest.mark.asyncio
async def test_transfer_waits_for_last_media_mark():
    class FakeManager:
        def __init__(self):
            self.transfers = []
            self.media_connections = []

        async def media_connected(self, session_id, channel_id):
            self.media_connections.append((session_id, channel_id))

        async def transfer(self, session_id, destination):
            self.transfers.append((session_id, destination))

    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = FakeManager()
    server.bind_call_manager(manager)
    server.register_session("call-1", media_channel_id="media-1")
    websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(websocket, _media_start())

    await server.handle_response(_response(
        type="chunk",
        session_id="call-1",
        audio_data=b"\x00\x00" * 320,
    ))
    await server.handle_response(_response(
        type="final",
        session_id="call-1",
        text='<operation name="transfer" destination="operator" />',
    ))
    assert manager.transfers == []
    mark = session.pending_operation_mark
    assert mark

    await server.process_control_frame(
        websocket,
        json.dumps({
            "event": "MEDIA_MARK_PROCESSED",
            "channel_id": "media-1",
            "correlation_id": mark,
        }),
        session,
    )
    assert manager.transfers == [("call-1", "operator")]


@pytest.mark.asyncio
async def test_transfer_prepare_hook_receives_trusted_conversation_keys():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    _register_media_session(server)
    session = await server.process_control_frame(
        FakeMediaWebSocket(),
        _media_start(),
    )
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        user_id="user-1",
        context_id="context-1",
    ))

    @server.on_transfer_prepare
    async def prepare_transfer(request, active_session):
        assert active_session is session
        request.variables["AIAVATAR_CONTEXT_ID"] = request.context_id
        request.variables["AIAVATAR_HANDOFF_ID"] = "handoff-1"

    request = await server.prepare_transfer(
        session,
        destination_alias="operator",
        destination="1234",
        transfer_strategy="refer",
    )

    assert request.session_id == "call-1"
    assert request.user_id == "user-1"
    assert request.context_id == "context-1"
    assert request.destination_alias == "operator"
    assert request.destination == "1234"
    assert request.transfer_strategy == "refer"
    assert request.variables == {
        "AIAVATAR_CONTEXT_ID": "context-1",
        "AIAVATAR_HANDOFF_ID": "handoff-1",
    }


@pytest.mark.asyncio
async def test_transfer_prepare_hook_cannot_override_control_variables():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    session = server.register_session("call-1")

    @server.on_transfer_prepare
    async def prepare_transfer(request, active_session):
        request.variables["AIAVATAR_SESSION_ID"] = "different-call"

    with pytest.raises(ValueError, match="reserved"):
        await server.prepare_transfer(
            session,
            destination_alias="operator",
            destination="1234",
            transfer_strategy="bridge",
        )


@pytest.mark.asyncio
async def test_transfer_notification_callback_failures_are_isolated(caplog):
    server = AIAvatarAsteriskServer(sts=DummySTS())
    invoked = []

    @server.on_transfer_started
    async def transfer_started(session_id, destination):
        invoked.append(("started", session_id, destination))
        raise RuntimeError("started callback failed")

    @server.on_transfer_completed
    async def transfer_completed(session_id, destination, method):
        invoked.append(("completed", session_id, destination, method))
        raise RuntimeError("completed callback failed")

    @server.on_transfer_failed
    async def transfer_failed(session_id, destination, reason):
        invoked.append(("failed", session_id, destination, reason))
        raise RuntimeError("failed callback failed")

    @server.on_transfer_unknown
    async def transfer_unknown(session_id, destination, reason):
        invoked.append(("unknown", session_id, destination, reason))
        raise RuntimeError("unknown callback failed")

    await server.notify_transfer_started("call-1", "operator")
    await server.notify_transfer_completed("call-1", "operator", "refer")
    await server.notify_transfer_failed("call-1", "operator", "busy")
    await server.notify_transfer_unknown("call-1", "operator", "caller_hangup")

    assert invoked == [
        ("started", "call-1", "operator"),
        ("completed", "call-1", "operator", "refer"),
        ("failed", "call-1", "operator", "busy"),
        ("unknown", "call-1", "operator", "caller_hangup"),
    ]
    assert "Asterisk transfer started callback failed" in caplog.text
    assert "Asterisk transfer completed callback failed" in caplog.text
    assert "Asterisk transfer failed callback failed" in caplog.text
    assert "Asterisk transfer outcome callback failed" in caplog.text


@pytest.mark.asyncio
async def test_stale_transaction_is_rejected_before_callbacks_and_operations():
    class FakeManager:
        def __init__(self):
            self.hangups = []

        async def media_connected(self, session_id, channel_id):
            pass

        async def hangup(self, session_id):
            self.hangups.append(session_id)

    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = FakeManager()
    server.bind_call_manager(manager)
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    await server.process_control_frame(websocket, _media_start())
    seen = []

    @server.on_response
    async def response_seen(aiavatar_response, response):
        seen.append((response.type, response.transaction_id))

    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="old",
    ))
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="current",
    ))
    await server.handle_response(_response(
        type="final",
        session_id="call-1",
        transaction_id="old",
        text='<operation name="hangup" />',
    ))

    assert seen == [("accepted", "old"), ("accepted", "current")]
    assert manager.hangups == []


@pytest.mark.asyncio
async def test_transactionless_response_expires_when_transaction_starts():
    response_started = asyncio.Event()
    release_response = asyncio.Event()
    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = ManagerStub()
    server.bind_call_manager(manager)
    _register_media_session(server)
    await server.process_control_frame(FakeMediaWebSocket(), _media_start())

    @server.on_response
    async def wait_during_old_response(aiavatar_response, response):
        if response.type == "final":
            response_started.set()
            await release_response.wait()

    old_response = asyncio.create_task(server.handle_response(_response(
        type="final",
        session_id="call-1",
        text='<operation name="hangup" />',
    )))
    await response_started.wait()
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="new",
    ))
    release_response.set()
    await old_response

    assert manager.hangups == []


@pytest.mark.asyncio
async def test_stale_response_stops_before_next_response_handler():
    first_handler_started = asyncio.Event()
    release_first_handler = asyncio.Event()
    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = ManagerStub()
    server.bind_call_manager(manager)
    _register_media_session(server)
    await server.process_control_frame(FakeMediaWebSocket(), _media_start())
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="old",
    ))

    @server.on_response
    async def first_handler(aiavatar_response, response):
        if response.type == "final":
            first_handler_started.set()
            await release_first_handler.wait()

    @server.on_response
    async def second_handler(aiavatar_response, response):
        if response.type == "final":
            await manager.hangup(response.session_id)

    old_response = asyncio.create_task(server.handle_response(_response(
        type="final",
        session_id="call-1",
        transaction_id="old",
        text='<operation name="hangup" />',
    )))
    await first_handler_started.wait()
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="new",
    ))
    release_first_handler.set()
    await old_response

    assert manager.hangups == []


@pytest.mark.asyncio
async def test_first_transaction_invalidates_transactionless_playback():
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class BlockingStopWebSocket(FakeMediaWebSocket):
        async def send_text(self, source):
            if json.loads(source)["command"] == "STOP_MEDIA_BUFFERING":
                stop_started.set()
                await release_stop.wait()
            await super().send_text(source)

    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = ManagerStub()
    server.bind_call_manager(manager)
    _register_media_session(server)
    websocket = BlockingStopWebSocket()
    session = await server.process_control_frame(websocket, _media_start())
    session.buffering = True
    session.audio_sent = True
    playback_generation = session.playback_generation

    old_response = asyncio.create_task(server.handle_response(_response(
        type="final",
        session_id="call-1",
        text='<operation name="hangup" />',
    )))
    await stop_started.wait()
    new_transaction = asyncio.create_task(server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="new",
    )))
    await asyncio.sleep(0)

    assert session.active_transaction_id == "new"
    assert session.playback_generation == playback_generation + 1

    release_stop.set()
    await asyncio.gather(old_response, new_transaction)

    commands = [json.loads(source)["command"] for source in websocket.text_messages]
    assert "STOP_MEDIA_BUFFERING" in commands
    assert "FLUSH_MEDIA" in commands
    assert "MARK_MEDIA" not in commands
    assert session.buffering is False
    assert manager.hangups == []


@pytest.mark.asyncio
async def test_response_expires_when_media_connection_is_replaced():
    response_started = asyncio.Event()
    release_response = asyncio.Event()
    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = ManagerStub()
    server.bind_call_manager(manager)
    _register_media_session(server)
    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())

    @server.on_response
    async def wait_during_old_response(aiavatar_response, response):
        if response.type == "final":
            response_started.set()
            await release_response.wait()

    old_response = asyncio.create_task(server.handle_response(_response(
        type="final",
        session_id="call-1",
        text='<operation name="hangup" />',
    )))
    await response_started.wait()
    await server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    )
    server.register_session("call-1", media_channel_id="media-2")
    await server.process_control_frame(
        FakeMediaWebSocket(),
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )
    release_response.set()
    await old_response

    assert manager.hangups == []


@pytest.mark.asyncio
async def test_old_media_response_cannot_control_replacement_connection():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    manager = ManagerStub()
    server.bind_call_manager(manager)
    _register_media_session(server)
    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())
    await server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    )

    server.register_session("call-1", media_channel_id="media-2")
    new_websocket = FakeMediaWebSocket()
    await server.process_control_frame(
        new_websocket,
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )
    seen = []

    @server.on_response
    async def response_seen(aiavatar_response, response):
        seen.append((aiavatar_response.session_id, response.session_id, response.type))

    await server.handle_response(STSResponse(
        type="accepted",
        session_id="media-1",
        transaction_id="old",
    ))
    await server.handle_response(STSResponse(
        type="final",
        session_id="media-1",
        transaction_id="old",
        text='<operation name="hangup" />',
    ))
    await server.handle_response(STSResponse(
        type="accepted",
        session_id="media-2",
        transaction_id="current",
    ))

    assert manager.hangups == []
    assert seen == [("call-1", "call-1", "accepted")]
    assert session.websocket is new_websocket
    assert session.pipeline_session_id == "media-2"


@pytest.mark.asyncio
async def test_transaction_switch_claims_new_id_before_flush_completes():
    flush_started = asyncio.Event()
    release_flush = asyncio.Event()

    class BlockingFlushWebSocket(FakeMediaWebSocket):
        block_flush = False

        async def send_text(self, source):
            if (
                self.block_flush
                and json.loads(source)["command"] == "FLUSH_MEDIA"
            ):
                flush_started.set()
                await release_flush.wait()
            await super().send_text(source)

    class FakeManager:
        def __init__(self):
            self.hangups = []

        async def media_connected(self, session_id, channel_id):
            pass

        async def hangup(self, session_id):
            self.hangups.append(session_id)

    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = FakeManager()
    server.bind_call_manager(manager)
    _register_media_session(server)
    websocket = BlockingFlushWebSocket()
    await server.process_control_frame(websocket, _media_start())
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="old",
    ))
    websocket.block_flush = True

    transaction_switch = asyncio.create_task(server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="current",
    )))
    await flush_started.wait()
    await server.handle_response(_response(
        type="final",
        session_id="call-1",
        transaction_id="old",
        text='<operation name="hangup" />',
    ))
    release_flush.set()
    await transaction_switch

    assert manager.hangups == []


@pytest.mark.asyncio
async def test_transaction_switch_revokes_old_mark_before_flush_completes():
    flush_started = asyncio.Event()
    release_flush = asyncio.Event()

    class BlockingFlushWebSocket(FakeMediaWebSocket):
        block_flush = False

        async def send_text(self, source):
            if (
                self.block_flush
                and json.loads(source)["command"] == "FLUSH_MEDIA"
            ):
                flush_started.set()
                await release_flush.wait()
            await super().send_text(source)

    class FakeManager:
        def __init__(self):
            self.hangups = []

        async def media_connected(self, session_id, channel_id):
            pass

        async def hangup(self, session_id):
            self.hangups.append(session_id)

    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = FakeManager()
    server.bind_call_manager(manager)
    _register_media_session(server)
    websocket = BlockingFlushWebSocket()
    session = await server.process_control_frame(websocket, _media_start())
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="old",
    ))
    websocket.block_flush = True
    await server.handle_response(_response(
        type="chunk",
        session_id="call-1",
        transaction_id="old",
        audio_data=b"\x00\x00" * 320,
    ))
    await server.handle_response(_response(
        type="final",
        session_id="call-1",
        transaction_id="old",
        text='<operation name="hangup" />',
    ))
    old_mark = session.pending_operation_mark
    assert old_mark

    transaction_switch = asyncio.create_task(server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="current",
    )))
    await flush_started.wait()
    await server.process_control_frame(
        websocket,
        json.dumps({
            "event": "MEDIA_MARK_PROCESSED",
            "channel_id": "media-1",
            "correlation_id": old_mark,
        }),
        session,
    )
    release_flush.set()
    await transaction_switch

    assert manager.hangups == []


@pytest.mark.asyncio
async def test_transaction_is_rechecked_after_final_media_mark_await():
    mark_started = asyncio.Event()
    release_mark = asyncio.Event()

    class BlockingMarkWebSocket(FakeMediaWebSocket):
        async def send_text(self, source):
            if json.loads(source)["command"] == "MARK_MEDIA":
                mark_started.set()
                await release_mark.wait()
            await super().send_text(source)

    class FakeManager:
        def __init__(self):
            self.hangups = []

        async def media_connected(self, session_id, channel_id):
            pass

        async def hangup(self, session_id):
            self.hangups.append(session_id)

    server = AIAvatarAsteriskServer(sts=DummySTS())
    manager = FakeManager()
    server.bind_call_manager(manager)
    _register_media_session(server)
    websocket = BlockingMarkWebSocket()
    await server.process_control_frame(websocket, _media_start())
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="old",
    ))
    await server.handle_response(_response(
        type="chunk",
        session_id="call-1",
        transaction_id="old",
        audio_data=b"\x00\x00" * 320,
    ))

    old_final = asyncio.create_task(server.handle_response(_response(
        type="final",
        session_id="call-1",
        transaction_id="old",
        text='<operation name="hangup" />',
    )))
    await mark_started.wait()
    new_transaction = asyncio.create_task(server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="current",
    )))
    await asyncio.sleep(0)
    release_mark.set()
    await asyncio.gather(old_final, new_transaction)

    assert manager.hangups == []


@pytest.mark.asyncio
@pytest.mark.parametrize("response_type", ["stop", "error", "canceled", "cancelled"])
async def test_pipeline_termination_unmutes_blocked_input(response_type):
    server = AIAvatarAsteriskServer(sts=DummySTS())
    _register_media_session(server)
    session = await server.process_control_frame(
        FakeMediaWebSocket(),
        _media_start(),
    )
    await server.handle_response(_response(
        type="accepted",
        session_id="call-1",
        transaction_id="transaction-1",
        metadata={"block_barge_in": True},
    ))
    assert session.muted is True

    await server.handle_response(_response(
        type=response_type,
        session_id="call-1",
        transaction_id="transaction-1",
    ))
    assert session.muted is False


def test_structured_operation_requires_object_shape():
    server = AIAvatarAsteriskServer(sts=DummySTS())

    assert server._extract_operation(None, {"operation": "hangup"}) is None
    operation = server._extract_operation(
        None,
        {"operation": {"name": "hangup"}},
    )
    assert operation is not None
    assert operation.name == "hangup"


@pytest.mark.asyncio
async def test_barge_in_flushes_queued_media_and_cancels_operation():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(websocket, _media_start())
    session.pending_operation_mark = "old-mark"
    await server.stop_response("call-1", "")

    commands = [json.loads(source)["command"] for source in websocket.text_messages]
    assert commands[-1] == "FLUSH_MEDIA"
    assert session.pending_operation_mark == ""


@pytest.mark.asyncio
async def test_disconnect_finalizes_pipeline_and_removes_media_session():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(websocket, _media_start())

    await server._cleanup_media_session(session)

    assert sts.finalized == ["media-1"]
    assert "call-1" not in server.sessions


def test_basic_auth_and_media_subprotocol_are_required():
    server = AIAvatarAsteriskServer(
        sts=DummySTS(),
        api_username="media-user",
        api_password="secret",
    )
    credentials = base64.b64encode(b"media-user:secret").decode("ascii")
    server._authenticate_websocket(FakeMediaWebSocket(headers={
        "sec-websocket-protocol": "media",
        "authorization": f"Basic {credentials}",
    }))

    with pytest.raises(AsteriskProtocolError):
        server._authenticate_websocket(FakeMediaWebSocket(headers={
            "sec-websocket-protocol": "media",
            "authorization": "Basic invalid",
        }))

    with pytest.raises(AsteriskProtocolError, match="subprotocol"):
        server._authenticate_websocket(FakeMediaWebSocket(headers={
            "authorization": f"Basic {credentials}",
        }))


def test_non_ascii_basic_auth_is_compared_without_type_error():
    server = AIAvatarAsteriskServer(
        sts=DummySTS(),
        api_username="média-user",
        api_password="秘密",
    )
    credentials = base64.b64encode("média-user:wrong".encode()).decode("ascii")

    with pytest.raises(AsteriskProtocolError, match="Invalid Basic"):
        server._authenticate_websocket(FakeMediaWebSocket(headers={
            "sec-websocket-protocol": "media",
            "authorization": f"Basic {credentials}",
        }))


def test_runtime_media_config_is_validated_before_mutation():
    server = AIAvatarAsteriskServer(sts=DummySTS())

    with pytest.raises(ValueError, match="media_chunk_duration_ms"):
        server.set_config({"media_chunk_duration_ms": 0})
    with pytest.raises(ValueError, match="tts_sample_rate"):
        server.set_config({"tts_sample_rate": -1})
    with pytest.raises(ValueError, match="media_chunk_duration_ms"):
        server.set_config({"media_chunk_duration_ms": 100.5})
    for invalid_timeout in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="positive finite number"):
            server.set_config({"media_flow_timeout": invalid_timeout})

    assert server.media_chunk_duration_ms == 100
    assert server.tts_sample_rate == 24_000
    assert server.media_flow_timeout == 10.0


@pytest.mark.asyncio
async def test_hangup_stops_accepting_binary_audio():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(websocket, _media_start())

    await server.process_control_frame(
        websocket,
        json.dumps({"event": "HANGUP", "channel_id": "media-1"}),
        session,
    )
    await server.process_binary_frame(websocket, session, b"\x01\x00" * 320)

    assert session.media_connected is False
    assert session.websocket is None
    assert sts.vad.samples == []


@pytest.mark.asyncio
async def test_same_media_channel_cannot_reconnect_after_disconnect():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    server.bind_call_manager(ManagerStub())
    server.register_session("call-1", media_channel_id="media-1")
    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())
    await server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    )

    with pytest.raises(AsteriskProtocolError, match="reconnection is not supported"):
        await server.process_control_frame(FakeMediaWebSocket(), _media_start())

    assert session.media_connected is False
    assert session.websocket is None
    assert sts.finalized == ["media-1"]


@pytest.mark.asyncio
async def test_manager_registered_replacement_during_old_cleanup_is_preserved():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    server.bind_call_manager(ManagerStub())
    server.register_session("call-1", media_channel_id="media-1")
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    @server.on_disconnect
    async def on_disconnect(session):
        cleanup_started.set()
        await allow_cleanup.wait()

    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())
    cleanup_task = asyncio.create_task(server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    ))
    await cleanup_started.wait()

    server.register_session("call-1", media_channel_id="media-2")
    new_websocket = FakeMediaWebSocket()
    replacement = await server.process_control_frame(
        new_websocket,
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )
    allow_cleanup.set()
    await cleanup_task

    assert replacement is session
    assert session.media_connected is True
    assert session.websocket is new_websocket
    assert sts.finalized == ["media-1"]
    assert "media-1" not in sts.vad.session_data
    assert sts.vad.session_data["media-2"]["channel"] == server.channel


@pytest.mark.asyncio
async def test_new_channel_disconnect_waits_for_old_cleanup_then_cleans_itself():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    server.bind_call_manager(ManagerStub())
    server.register_session("call-1", media_channel_id="media-1")
    disconnect_started = asyncio.Event()
    release_disconnect = asyncio.Event()

    @server.on_disconnect
    async def on_disconnect(session):
        disconnect_started.set()
        await release_disconnect.wait()

    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())
    old_cleanup = asyncio.create_task(server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    ))
    await disconnect_started.wait()

    server.register_session("call-1", media_channel_id="media-2")
    new_websocket = FakeMediaWebSocket()
    await server.process_control_frame(
        new_websocket,
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )
    new_cleanup = asyncio.create_task(server._cleanup_media_session(
        session,
        expected_websocket=new_websocket,
    ))
    release_disconnect.set()
    await asyncio.gather(old_cleanup, new_cleanup)

    assert session.websocket is None
    assert session.media_connected is False
    assert sts.finalized == ["media-1", "media-2"]


@pytest.mark.asyncio
async def test_old_cleanup_does_not_cancel_new_channel_connect_callback():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    server.bind_call_manager(ManagerStub())
    server.register_session("call-1", media_channel_id="media-1")
    old_callback_started = asyncio.Event()
    new_callback_started = asyncio.Event()
    callback_blocker = asyncio.Event()
    callback_canceled = []
    callback_count = 0

    @server.on_connect
    async def on_connect(request, session):
        nonlocal callback_count
        callback_count += 1
        callback_number = callback_count
        if callback_number == 1:
            old_callback_started.set()
        else:
            new_callback_started.set()
        try:
            await callback_blocker.wait()
        finally:
            callback_canceled.append(callback_number)

    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())
    await old_callback_started.wait()

    cancel_started = asyncio.Event()
    allow_cancel = asyncio.Event()
    cancel_generation = server._cancel_session_tasks

    async def pause_generation_cancel(session_id, connection_generation):
        cancel_started.set()
        await allow_cancel.wait()
        await cancel_generation(session_id, connection_generation)

    server._cancel_session_tasks = pause_generation_cancel
    cleanup = asyncio.create_task(server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    ))
    await cancel_started.wait()

    server.register_session("call-1", media_channel_id="media-2")
    new_websocket = FakeMediaWebSocket()
    await server.process_control_frame(
        new_websocket,
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )
    await new_callback_started.wait()
    allow_cancel.set()
    await cleanup

    assert callback_canceled == [1]
    assert session.websocket is new_websocket
    assert session.media_connected is True

    callback_blocker.set()
    await asyncio.sleep(0)
    await server._cleanup_media_session(
        session,
        expected_websocket=new_websocket,
    )


@pytest.mark.asyncio
async def test_manager_registered_replacement_resets_response_state():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    server.bind_call_manager(ManagerStub())
    server.register_session("call-1", media_channel_id="media-1")
    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())
    session.buffering = True
    session.audio_sent = True
    session.muted = True
    session.last_mark = "old-mark"
    session.unmute_mark = "old-unmute"
    session.pending_operation_mark = "old-operation-mark"
    session.pending_operation = server._extract_operation(
        '<operation name="hangup" />',
        None,
    )
    session.active_transaction_id = "old-transaction"
    await server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    )

    server.register_session("call-1", media_channel_id="media-2")
    new_websocket = FakeMediaWebSocket()
    replacement = await server.process_control_frame(
        new_websocket,
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )

    assert replacement is session
    assert session.buffering is False
    assert session.audio_sent is False
    assert session.muted is False
    assert session.last_mark == ""
    assert session.unmute_mark == ""
    assert session.pending_operation_mark == ""
    assert session.pending_operation is None
    assert session.active_transaction_id == ""


@pytest.mark.asyncio
async def test_manager_media_ready_failure_rolls_back_websocket():
    class FailingManager:
        async def media_connected(self, session_id, channel_id):
            raise RuntimeError("manager notification failed")

    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    server.bind_call_manager(FailingManager())
    session = server.register_session("call-1", media_channel_id="media-1")
    websocket = FakeMediaWebSocket()

    with pytest.raises(RuntimeError, match="manager notification failed"):
        await server.process_control_frame(websocket, _media_start())

    assert session.websocket is None
    assert session.media_connected is False
    assert sts.finalized == ["media-1"]


@pytest.mark.asyncio
async def test_session_start_unregister_does_not_revive_vad_or_callbacks():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    manager = ManagerStub()
    server.bind_call_manager(manager)
    server.register_session("call-1", media_channel_id="media-1")
    connected = []

    @server.on_session_start
    async def unregister_during_start(request, session):
        await server.unregister_session(session.session_id)

    @server.on_connect
    async def on_connect(request, session):
        connected.append(session.session_id)

    with pytest.raises(AsteriskProtocolError, match="startup was in progress"):
        await server.process_control_frame(FakeMediaWebSocket(), _media_start())

    await asyncio.sleep(0)
    assert "call-1" not in server.sessions
    assert "media-1" not in sts.vad.session_data
    assert connected == []
    assert manager.media_connections == []
    assert sts.finalized == ["media-1"]


@pytest.mark.asyncio
async def test_old_websocket_cannot_send_to_manager_registered_replacement():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    server.bind_call_manager(ManagerStub())
    server.register_session("call-1", media_channel_id="media-1")
    old_websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(old_websocket, _media_start())
    await server._cleanup_media_session(
        session,
        expected_websocket=old_websocket,
    )
    server.register_session("call-1", media_channel_id="media-2")
    current_websocket = FakeMediaWebSocket()
    await server.process_control_frame(
        current_websocket,
        _media_start(channel_id="media-2", connection_id="connection-2"),
    )

    await server.process_binary_frame(
        old_websocket,
        session,
        b"\x01\x00" * 320,
    )
    with pytest.raises(AsteriskProtocolError, match="superseded"):
        await server.process_control_frame(
            old_websocket,
            json.dumps({"event": "MEDIA_XOFF", "channel_id": "media-1"}),
            session,
        )

    assert sts.vad.samples == []
    assert session.websocket is current_websocket
    assert session.flow_blocked is False


@pytest.mark.asyncio
async def test_full_call_cleanup_rejects_media_reconnect():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    server.register_session("call-1", media_channel_id="media-1")
    old_websocket = FakeMediaWebSocket()
    await server.process_control_frame(old_websocket, _media_start())
    disconnect_started = asyncio.Event()
    release_disconnect = asyncio.Event()

    @server.on_disconnect
    async def on_disconnect(session):
        disconnect_started.set()
        await release_disconnect.wait()

    cleanup = asyncio.create_task(server.unregister_session("call-1"))
    await disconnect_started.wait()
    with pytest.raises(AsteriskProtocolError, match="already closing"):
        await server.process_control_frame(FakeMediaWebSocket(), _media_start())

    release_disconnect.set()
    await cleanup
    assert "call-1" not in server.sessions


@pytest.mark.asyncio
async def test_session_start_failure_rolls_back_media_ownership():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    _register_media_session(server)

    @server.on_session_start
    async def fail_start(request, session):
        raise RuntimeError("application setup failed")

    with pytest.raises(RuntimeError, match="application setup failed"):
        await server.process_control_frame(FakeMediaWebSocket(), _media_start())

    assert "call-1" not in server.sessions
    assert sts.finalized == ["media-1"]


@pytest.mark.asyncio
async def test_cleanup_releases_session_when_pipeline_finalize_fails():
    class FailingFinalizeSTS(DummySTS):
        async def finalize(self, session_id):
            self.finalized.append(session_id)
            raise RuntimeError("finalize failed")

    sts = FailingFinalizeSTS()
    server = AIAvatarAsteriskServer(sts=sts)
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(websocket, _media_start())

    await server._cleanup_media_session(session)

    assert session.websocket is None
    assert "call-1" not in server.sessions
    assert sts.finalized == ["media-1"]


@pytest.mark.asyncio
async def test_canceling_unregister_waiter_does_not_interrupt_cleanup():
    sts = DummySTS()
    server = AIAvatarAsteriskServer(sts=sts)
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    await server.process_control_frame(websocket, _media_start())
    disconnect_started = asyncio.Event()
    release_disconnect = asyncio.Event()

    @server.on_disconnect
    async def slow_disconnect(session):
        disconnect_started.set()
        await release_disconnect.wait()

    unregister = asyncio.create_task(server.unregister_session("call-1"))
    await disconnect_started.wait()
    unregister.cancel()
    release_disconnect.set()

    with pytest.raises(asyncio.CancelledError):
        await unregister
    assert "call-1" not in server.sessions
    assert sts.finalized == ["media-1"]


@pytest.mark.asyncio
async def test_xoff_timeout_drops_output_until_xon():
    server = AIAvatarAsteriskServer(
        sts=DummySTS(),
        media_flow_timeout=0.01,
    )
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    session = await server.process_control_frame(websocket, _media_start())
    await server.process_control_frame(
        websocket,
        json.dumps({"event": "MEDIA_XOFF", "channel_id": "media-1"}),
        session,
    )

    await asyncio.wait_for(
        server.send_voice("call-1", audio_data=b"\x01\x00" * 320),
        timeout=0.5,
    )
    assert websocket.binary_messages == []
    assert session.flow_timeout_expired is True

    await server.process_control_frame(
        websocket,
        json.dumps({"event": "MEDIA_XON", "channel_id": "media-1"}),
        session,
    )
    await server.send_voice("call-1", audio_data=b"\x01\x00" * 320)
    assert websocket.binary_messages


@pytest.mark.asyncio
async def test_partial_audio_before_xoff_still_uses_mark_before_transfer():
    class XoffAfterFirstChunk(FakeMediaWebSocket):
        session = None

        async def send_bytes(self, source):
            await super().send_bytes(source)
            if len(self.binary_messages) == 1:
                self.session.flow_blocked = True
                self.session.media_writable.clear()

    class FakeManager:
        def __init__(self):
            self.transfers = []

        async def media_connected(self, session_id, channel_id):
            pass

        async def transfer(self, session_id, destination):
            self.transfers.append((session_id, destination))

    server = AIAvatarAsteriskServer(
        sts=DummySTS(),
        media_flow_timeout=0.01,
    )
    manager = FakeManager()
    server.bind_call_manager(manager)
    _register_media_session(server)
    websocket = XoffAfterFirstChunk()
    session = await server.process_control_frame(websocket, _media_start())
    websocket.session = session

    await server.send_voice("call-1", audio_data=b"\x01\x00" * 10_000)
    await server.handle_response(_response(
        type="final",
        session_id="call-1",
        text='<operation name="transfer" destination="operator" />',
    ))

    commands = [json.loads(source)["command"] for source in websocket.text_messages]
    assert "MARK_MEDIA" in commands
    assert session.pending_operation_mark
    assert manager.transfers == []


@pytest.mark.asyncio
async def test_media_requires_registered_session():
    server = AIAvatarAsteriskServer(sts=DummySTS())

    with pytest.raises(AsteriskProtocolError):
        await server.process_control_frame(FakeMediaWebSocket(), _media_start())


@pytest.mark.asyncio
async def test_media_session_id_sources_must_match():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    _register_media_session(server)
    websocket = FakeMediaWebSocket(query_params={"session_id": "other-call"})

    with pytest.raises(AsteriskProtocolError):
        await server.process_control_frame(websocket, _media_start("call-1"))


@pytest.mark.asyncio
async def test_media_session_id_does_not_fall_back_to_channel_id():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    _register_media_session(server, session_id="media-1")
    payload = json.loads(_media_start())
    payload["channel_variables"] = {"session_id": "media-1"}

    with pytest.raises(AsteriskProtocolError, match="requires.*session_id"):
        await server.process_control_frame(
            FakeMediaWebSocket(),
            json.dumps(payload),
        )


@pytest.mark.asyncio
async def test_cleanup_cancels_session_callback_tasks():
    server = AIAvatarAsteriskServer(sts=DummySTS())
    _register_media_session(server)
    websocket = FakeMediaWebSocket()
    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    @server.on_connect
    async def on_connect(request, session):
        callback_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            callback_cancelled.set()

    session = await server.process_control_frame(websocket, _media_start())
    await callback_started.wait()
    connection_generation = session.connection_generation
    playback_generation = session.playback_generation
    await server.stop_response("call-1", "")
    assert session.connection_generation == connection_generation
    assert session.playback_generation == playback_generation + 1
    await server._cleanup_media_session(session)

    assert callback_cancelled.is_set()
    assert "call-1" not in server._session_tasks
