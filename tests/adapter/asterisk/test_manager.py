import asyncio

import httpx
import pytest

from aiavatar.adapter.asterisk.ari_client import AsteriskARIClient
from aiavatar.adapter.asterisk.manager import AsteriskCallManager
from aiavatar.adapter.asterisk.server import AIAvatarAsteriskServer

from .conftest import DummySTS, FakeAdapter, FakeARIClient, FakeResponse


def _ari_client(transport, **values):
    return AsteriskARIClient(
        base_url="https://asterisk.invalid:8089/ari",
        username="ari-user",
        password="ari-secret",
        http_client=transport,
        **values,
    )


def _manager(*, strategy="refer_then_bridge"):
    adapter = FakeAdapter()
    ari = FakeARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234", "sales": "2345"},
        transfer_strategy=strategy,
    )
    return manager, adapter, ari


def _active_call(manager, adapter):
    session = adapter.register_session(
        "call-1",
        ari_caller_channel_id="caller-1",
        media_channel_id="media-1",
        bridge_id="bridge-1",
        caller_number="0312345678",
        caller_name="Example Caller",
        caller_presentation="allowed",
    )
    manager.registry.register(session)
    return session


async def _wait_for(predicate, *, timeout=0.3):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.005)


async def _connect_restored_media(manager, session):
    await _wait_for(lambda: session.transfer_state == "restoring_ai")
    await manager.media_connected(
        session.session_id,
        session.media_channel_id,
    )
    await _wait_for(lambda: session.transfer_state == "active")


def test_arbitrary_destination_is_rejected():
    manager, _, _ = _manager()

    with pytest.raises(ValueError):
        manager.resolve_destination("sip:attacker@example.invalid")
    with pytest.raises(ValueError):
        manager.resolve_destination("09012345678")


def test_refer_strategy_does_not_require_bridge_endpoint():
    manager = AsteriskCallManager(
        adapter=FakeAdapter(),
        ari_client=_ari_client(FakeARIClient()),
        transfer_destinations={},
        transfer_strategy="refer",
    )

    assert manager.call_service.bridge_endpoint == ""


@pytest.mark.parametrize("strategy", ["bridge", "refer_then_bridge"])
def test_bridge_strategy_requires_bridge_endpoint(strategy):
    with pytest.raises(
        ValueError,
        match="bridge_endpoint is required when transfer_strategy uses bridge",
    ):
        AsteriskCallManager(
            adapter=FakeAdapter(),
            ari_client=_ari_client(FakeARIClient()),
            transfer_destinations={},
            transfer_strategy=strategy,
        )


def test_refer_timeout_must_be_positive():
    with pytest.raises(ValueError, match="refer_timeout must be positive"):
        AsteriskCallManager(
            adapter=FakeAdapter(),
            ari_client=_ari_client(FakeARIClient()),
            bridge_endpoint="operator-trunk",
            transfer_destinations={},
            refer_timeout=0,
        )


def test_media_start_timeout_must_be_positive():
    with pytest.raises(ValueError, match="media_start_timeout must be positive"):
        AsteriskCallManager(
            adapter=FakeAdapter(),
            ari_client=_ari_client(FakeARIClient()),
            bridge_endpoint="operator-trunk",
            transfer_destinations={},
            media_start_timeout=0,
        )


@pytest.mark.asyncio
async def test_inbound_stasis_creates_slin16_websocket_media_bridge():
    manager, adapter, ari = _manager()

    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["inbound"],
        "channel": {
            "id": "caller-1",
            "caller": {"number": "0312345678", "name": "Example Caller"},
            "dialplan": {"exten": "5000"},
        },
    })

    session = manager.sessions["caller-1"]
    assert session.called_number == "5000"
    external_media = next(
        call for call in ari.calls
        if call[0] == "POST" and call[1] == "channels/externalMedia"
    )
    assert external_media[2]["transport"] == "websocket"
    assert external_media[2]["encapsulation"] == "none"
    assert external_media[2]["format"] == "slin16"
    assert external_media[2]["transport_data"].startswith("f(json)")
    assert adapter.sessions["caller-1"] is session


@pytest.mark.asyncio
async def test_uncertain_external_media_create_retains_ids_for_cleanup():
    class TimeoutExternalMediaARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels/externalMedia":
                self.calls.append((method, path, dict(params or {}), json))
                raise asyncio.TimeoutError("ARI response was lost")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = TimeoutExternalMediaARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={},
    )

    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["inbound"],
        "channel": {
            "id": "caller-timeout",
            "dialplan": {"exten": "5000"},
        },
    })

    assert not manager.sessions
    assert any(
        call[0] == "DELETE" and call[1].startswith("channels/aiavatar-media-")
        for call in ari.calls
    )
    assert any(
        call[0] == "DELETE" and call[1].startswith("bridges/aiavatar-")
        for call in ari.calls
    )


@pytest.mark.asyncio
async def test_inbound_duplicate_session_id_is_rejected_without_rebinding_call():
    manager, adapter, ari = _manager()
    existing = _active_call(manager, adapter)
    ari.variables[(
        "channels/caller-2/variable",
        "AIAVATAR_SESSION_ID",
    )] = "call-1"

    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["inbound"],
        "channel": {"id": "caller-2"},
    })

    assert manager.sessions["call-1"] is existing
    assert existing.ari_caller_channel_id == "caller-1"
    assert manager.registry.by_caller("caller-2") is None
    assert any(
        call[:2] == ("DELETE", "channels/caller-2")
        for call in ari.calls
    )


@pytest.mark.asyncio
async def test_nonblocking_inbound_setup_does_not_block_event_processing():
    manager, _, _ = _manager()
    setup_started = asyncio.Event()
    setup_cancelled = asyncio.Event()

    async def blocking_setup(channel):
        setup_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            setup_cancelled.set()

    manager.event_handler._handle_inbound_start = blocking_setup

    await asyncio.wait_for(
        manager.handle_ari_event({
            "type": "StasisStart",
            "args": ["inbound"],
            "channel": {"id": "caller-1"},
        }, wait=False),
        timeout=0.05,
    )
    await setup_started.wait()
    assert "caller-1" in manager.event_handler._inbound_setup_tasks

    await manager.handle_ari_event({
        "type": "StasisEnd",
        "channel": {"id": "caller-1"},
    }, wait=False)

    assert setup_cancelled.is_set()
    assert manager.event_handler._inbound_setup_tasks == {}


@pytest.mark.asyncio
async def test_close_hangs_up_caller_during_inbound_setup():
    manager, adapter, ari = _manager()
    setup_started = asyncio.Event()

    async def blocking_media_setup(session, *, answer_caller):
        setup_started.set()
        await asyncio.Event().wait()

    manager.call_service.create_conversation_media = blocking_media_setup
    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["inbound"],
        "channel": {"id": "caller-1", "dialplan": {"exten": "5000"}},
    }, wait=False)
    await setup_started.wait()

    await manager.close()

    assert not manager.sessions
    assert adapter.unregistered == ["caller-1"]
    assert any(
        call[:2] == ("DELETE", "channels/caller-1")
        for call in ari.calls
    )


@pytest.mark.asyncio
async def test_close_hangs_up_caller_before_inbound_registration():
    variable_read_started = asyncio.Event()

    class BlockingVariableARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if (
                method == "GET"
                and path == "channels/caller-1/variable"
                and (params or {}).get("variable") == "AIAVATAR_SESSION_ID"
            ):
                self.calls.append((method, path, dict(params or {}), json))
                variable_read_started.set()
                await asyncio.Event().wait()
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = BlockingVariableARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={},
    )
    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["inbound"],
        "channel": {"id": "caller-1", "dialplan": {"exten": "5000"}},
    }, wait=False)
    await variable_read_started.wait()

    await manager.close()

    assert not manager.sessions
    assert adapter.sessions == {}
    assert any(
        call[:2] == ("DELETE", "channels/caller-1")
        for call in ari.calls
    )


@pytest.mark.asyncio
async def test_external_media_waits_until_channel_enters_stasis():
    class DelayedStasisARIClient(FakeARIClient):
        def __init__(self):
            super().__init__()
            self.media_add_attempts = 0

        async def request(self, method, path, params=None, json=None):
            if (
                method == "POST"
                and path.endswith("/addChannel")
                and str((params or {}).get("channel", "")).startswith("aiavatar-media-")
            ):
                self.media_add_attempts += 1
                if self.media_add_attempts == 1:
                    self.calls.append((method, path, dict(params or {}), json))
                    return FakeResponse(
                        422,
                        text='{"message":"Channel not in Stasis application"}',
                    )
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = DelayedStasisARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={},
    )

    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["inbound"],
        "channel": {"id": "caller-1", "dialplan": {"exten": "5000"}},
    })

    assert ari.media_add_attempts == 2
    assert "caller-1" in manager.sessions


@pytest.mark.asyncio
async def test_refer_continues_to_allowlisted_dialplan_extension():
    manager, adapter, ari = _manager(strategy="refer")
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "refer_pending"
    continues = [
        call for call in ari.calls
        if call[0] == "POST" and call[1] == "channels/caller-1/continue"
    ]
    assert continues[0][2] == {
        "context": "aiavatar-transfer",
        "extension": "1234",
        "priority": 1,
    }
    assert adapter.started == [("call-1", "operator")]
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_lost_refer_continue_response_reconciles_before_fallback():
    class LostContinueResponseARIClient(FakeARIClient):
        def __init__(self):
            super().__init__()
            self.channel_checks = 0

        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels/caller-1/continue":
                self.calls.append((method, path, dict(params or {}), json))
                raise asyncio.TimeoutError("continue response was lost")
            if method == "GET" and path == "channels/caller-1":
                self.channel_checks += 1
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(200, {
                    "id": "caller-1",
                    "dialplan": {
                        "app_name": "Stasis",
                        "app_data": "aiavatar,transfer-failed",
                    },
                })
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = LostContinueResponseARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer_then_bridge",
        refer_timeout=0.01,
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "refer_pending"
    assert ari.channel_checks == 0
    assert not any(call[:2] == ("POST", "channels") for call in ari.calls)

    await _wait_for(lambda: session.transfer_state == "bridge_dialing")
    assert ari.channel_checks >= 1
    assert len([
        call for call in ari.calls
        if call[:2] == ("POST", "channels")
    ]) == 1
    await manager.hangup("call-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 500, 504])
async def test_indeterminate_refer_http_status_reconciles_before_fallback(
    status_code,
):
    class IndeterminateContinueARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels/caller-1/continue":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(status_code, text="continue outcome unknown")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = IndeterminateContinueARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer_then_bridge",
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "refer_pending"
    assert not any(call[:2] == ("POST", "channels") for call in ari.calls)
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_definite_refer_http_failure_starts_bridge_fallback():
    class RejectedContinueARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels/caller-1/continue":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(400, text="invalid transfer extension")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = RejectedContinueARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer_then_bridge",
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "bridge_dialing"
    assert len([
        call for call in ari.calls
        if call[:2] == ("POST", "channels")
    ]) == 1
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_refer_client_bug_is_not_treated_as_unknown():
    class BuggyContinueARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels/caller-1/continue":
                raise ValueError("client implementation bug")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = BuggyContinueARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer_then_bridge",
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "bridge_dialing"
    assert len([
        call for call in ari.calls
        if call[:2] == ("POST", "channels")
    ]) == 1
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_refer_response_decoding_failure_waits_for_reconciliation():
    class DecodingFailureARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels/caller-1/continue":
                self.calls.append((method, path, dict(params or {}), json))
                raise httpx.DecodingError(
                    "response body could not be decoded",
                    request=httpx.Request(
                        method,
                        f"https://asterisk.invalid/{path}",
                    ),
                )
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = DecodingFailureARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer_then_bridge",
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "refer_pending"
    assert not any(call[:2] == ("POST", "channels") for call in ari.calls)
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_transfer_started_notification_failure_does_not_stop_refer():
    adapter = AIAvatarAsteriskServer(sts=DummySTS())
    ari = FakeARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer",
    )
    session = adapter.register_session(
        "call-1",
        ari_caller_channel_id="caller-1",
        media_channel_id="media-1",
        bridge_id="bridge-1",
    )
    manager.registry.register(session)

    @adapter.on_transfer_started
    async def transfer_started(session_id, destination):
        raise RuntimeError("audit service unavailable")

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "refer_pending"
    assert any(
        call[0] == "POST" and call[1] == "channels/caller-1/continue"
        for call in ari.calls
    )
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_transfer_hook_variables_reach_refer_channel():
    manager, adapter, ari = _manager(strategy="refer")
    session = _active_call(manager, adapter)
    session.user_id = "+81312345678"
    session.context_id = "context-1"
    adapter.transfer_variables = {
        "AIAVATAR_CONTEXT_ID": "context-1",
        "AIAVATAR_HANDOFF_ID": "handoff-1",
    }

    await manager.transfer("call-1", "operator")

    assigned = {
        call[2]["variable"]: call[2]["value"]
        for call in ari.calls
        if call[:2] == ("POST", "channels/caller-1/variable")
    }
    assert assigned["AIAVATAR_SESSION_ID"] == "call-1"
    assert assigned["AIAVATAR_CONTEXT_ID"] == "context-1"
    assert assigned["AIAVATAR_HANDOFF_ID"] == "handoff-1"
    assert adapter.prepared[0].user_id == "+81312345678"
    assert adapter.prepared[0].context_id == "context-1"

    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_transfer_hook_variables_reach_bridge_outbound_channel():
    manager, adapter, ari = _manager(strategy="bridge")
    _active_call(manager, adapter)
    adapter.transfer_variables = {"AIAVATAR_HANDOFF_ID": "handoff-1"}

    await manager.transfer("call-1", "operator")

    originate = next(
        call for call in ari.calls
        if call[:2] == ("POST", "channels")
    )
    assert originate[3]["variables"]["AIAVATAR_HANDOFF_ID"] == "handoff-1"
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_transfer_prepare_failure_aborts_before_ari():
    manager, adapter, ari = _manager(strategy="refer")
    _active_call(manager, adapter)

    async def fail_prepare(*args, **kwargs):
        raise RuntimeError("handoff store unavailable")

    adapter.prepare_transfer = fail_prepare
    await manager.transfer("call-1", "operator")

    assert ari.calls == []
    assert adapter.started == []
    assert adapter.failed == [
        ("call-1", "operator", "transfer_prepare_failed")
    ]
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_refer_timeout_closes_call_when_original_channel_is_gone():
    adapter = FakeAdapter()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(FakeARIClient()),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer",
        refer_timeout=0.01,
    )
    _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")
    await _wait_for(
        lambda: (
            "call-1" not in manager.sessions
            and "call-1" not in manager._actors
        )
    )

    assert adapter.completed == []
    assert adapter.unknown == [
        ("call-1", "operator", "caller_channel_missing"),
    ]
    assert adapter.unregistered == ["call-1"]
    assert manager._actors == {}


@pytest.mark.asyncio
async def test_refer_timeout_restores_ai_when_failed_stasis_event_was_missed():
    class FailedReferInStasisARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "GET" and path == "channels/caller-1":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(200, {
                    "id": "caller-1",
                    "dialplan": {
                        "app_name": "Stasis",
                        "app_data": "aiavatar,transfer-failed",
                    },
                })
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(FailedReferInStasisARIClient()),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer",
        refer_timeout=0.01,
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")
    await _connect_restored_media(manager, session)

    assert adapter.failed == [("call-1", "operator", "refer_timeout")]
    assert session.media_channel_id.startswith("aiavatar-media-")
    assert manager._actors["call-1"].state.value == "active"

    await manager.hangup("call-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_status", ["", "UNKNOWN"])
async def test_unknown_refer_status_fails_closed_without_bridge_fallback(
    transfer_status,
):
    class UnknownStatusARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "GET" and path == "channels/caller-1":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(200, {
                    "id": "caller-1",
                    "dialplan": {
                        "app_name": "Stasis",
                        "app_data": "aiavatar",
                    },
                })
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = UnknownStatusARIClient()
    ari.variables[(
        "channels/caller-1/variable",
        "TRANSFERSTATUS",
    )] = transfer_status
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer_then_bridge",
        refer_timeout=0.01,
    )
    _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")
    await _wait_for(lambda: "call-1" not in manager.sessions)

    assert adapter.failed == []
    assert adapter.unknown == [
        ("call-1", "operator", "unknown_transfer_status"),
    ]
    assert not any(
        method == "POST" and path == "channels"
        for method, path, _, _ in ari.calls
    )
    assert any(
        method == "DELETE" and path == "channels/caller-1"
        for method, path, _, _ in ari.calls
    )


@pytest.mark.asyncio
async def test_refer_timeout_rechecks_while_transfer_is_outside_stasis():
    class DelayedReferExitARIClient(FakeARIClient):
        def __init__(self):
            super().__init__()
            self.channel_checks = 0

        async def request(self, method, path, params=None, json=None):
            if method == "GET" and path == "channels/caller-1":
                self.channel_checks += 1
                self.calls.append((method, path, dict(params or {}), json))
                if self.channel_checks == 1:
                    return FakeResponse(200, {
                        "id": "caller-1",
                        "dialplan": {
                            "app_name": "Transfer",
                            "app_data": "PJSIP/sip:1234@example.invalid",
                        },
                    })
                return FakeResponse(404)
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = DelayedReferExitARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer",
        refer_timeout=0.01,
    )
    _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")
    await _wait_for(lambda: "call-1" not in manager.sessions)

    assert ari.channel_checks >= 2
    assert adapter.completed == []
    assert adapter.unknown == [
        ("call-1", "operator", "caller_channel_missing"),
    ]


@pytest.mark.asyncio
async def test_refer_completion_cancels_timeout():
    adapter = FakeAdapter()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(FakeARIClient()),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer",
        refer_timeout=0.01,
    )
    _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")
    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["transfer-completed"],
        "channel": {"id": "caller-1"},
    })
    await asyncio.sleep(0.02)

    assert adapter.completed == [("call-1", "operator", "refer")]
    assert adapter.unregistered == ["call-1"]


@pytest.mark.asyncio
async def test_caller_hangup_during_refer_is_reported_as_unknown():
    manager, adapter, _ = _manager(strategy="refer")
    _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")
    await manager.handle_ari_event({
        "type": "ChannelDestroyed",
        "channel": {"id": "caller-1"},
        "cause_txt": "caller_hung_up",
    })

    assert adapter.completed == []
    assert adapter.failed == []
    assert adapter.unknown == [
        ("call-1", "operator", "caller_hung_up"),
    ]
    assert "call-1" not in manager.sessions


@pytest.mark.asyncio
async def test_refer_failure_originates_bridge_fallback_with_original_identity():
    manager, adapter, ari = _manager(strategy="refer_then_bridge")
    session = _active_call(manager, adapter)
    await manager.transfer("call-1", "operator")

    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["transfer-failed"],
        "channel": {"id": "caller-1"},
    })

    assert session.transfer_state == "bridge_dialing"
    originates = [call for call in ari.calls if call[0] == "POST" and call[1] == "channels"]
    assert len(originates) == 1
    assert originates[0][2]["endpoint"] == "PJSIP/1234@operator-trunk"
    assert originates[0][3]["variables"]["CALLERID(num)"] == "0312345678"


@pytest.mark.asyncio
async def test_failed_bridge_setup_restores_ai_and_actor_state():
    class FailingOriginateARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(500, text="originate failed")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = FailingOriginateARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="bridge",
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert manager._actors["call-1"].state.value == "restoring_ai"
    assert session.transfer_state == "restoring_ai"
    assert adapter.failed == []

    await _connect_restored_media(manager, session)

    assert manager._actors["call-1"].state.value == "active"
    assert session.transfer_state == "active"
    assert session.destination_channel_id == ""
    assert session.holding_bridge_id == ""
    assert session.media_channel_id.startswith("aiavatar-media-")
    assert len(adapter.failed) == 1
    assert adapter.failed[0][:2] == ("call-1", "operator")
    assert adapter.failed[0][2].startswith("bridge_setup_failed:")
    assert any(
        call[0] == "DELETE" and call[1].startswith("channels/aiavatar-transfer-")
        for call in ari.calls
    )

    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_partial_refer_setup_removes_old_topology_before_restore():
    class FailingVariableARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels/caller-1/variable":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(500, text="variable service failed")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = FailingVariableARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        transfer_destinations={"operator": "1234"},
        transfer_strategy="refer",
    )
    session = _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert session.transfer_state == "restoring_ai"
    assert any(call[:2] == ("DELETE", "channels/media-1") for call in ari.calls)
    assert any(call[:2] == ("DELETE", "bridges/bridge-1") for call in ari.calls)
    assert session.media_channel_id.startswith("aiavatar-media-")
    assert session.bridge_id.startswith("aiavatar-")

    await _connect_restored_media(manager, session)
    await manager.hangup("call-1")


@pytest.mark.asyncio
async def test_failed_destination_delete_is_retried_during_cleanup():
    class FailFirstDestinationDeleteARIClient(FakeARIClient):
        def __init__(self):
            super().__init__()
            self.destination_delete_attempts = 0

        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(500, text="originate outcome unknown")
            if method == "DELETE" and path.startswith("channels/aiavatar-transfer-"):
                self.destination_delete_attempts += 1
                if self.destination_delete_attempts == 1:
                    self.calls.append((method, path, dict(params or {}), json))
                    return FakeResponse(500, text="delete temporarily failed")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = FailFirstDestinationDeleteARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="bridge",
    )
    _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")

    assert ari.destination_delete_attempts == 2
    assert "call-1" not in manager.sessions


@pytest.mark.asyncio
async def test_media_restore_timeout_cleans_up_call():
    class FailingOriginateARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "POST" and path == "channels":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(500, text="originate failed")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = FailingOriginateARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={"operator": "1234"},
        transfer_strategy="bridge",
        media_start_timeout=0.01,
    )
    _active_call(manager, adapter)

    await manager.transfer("call-1", "operator")
    await _wait_for(lambda: "call-1" not in manager.sessions)

    assert adapter.failed == []
    assert adapter.unregistered == ["call-1"]
    assert any(call[:2] == ("DELETE", "channels/caller-1") for call in ari.calls)


@pytest.mark.asyncio
async def test_caller_hangup_is_processed_while_media_restore_is_pending():
    manager, adapter, _ = _manager(strategy="bridge")
    _active_call(manager, adapter)

    async def fail_bridge_setup(session):
        raise RuntimeError("originate failed")

    manager.call_service.begin_bridge_transfer = fail_bridge_setup
    await manager.transfer("call-1", "operator")
    assert manager._actors["call-1"].state.value == "restoring_ai"

    await manager.handle_ari_event({
        "type": "ChannelDestroyed",
        "channel": {"id": "caller-1"},
        "cause_txt": "caller_hung_up",
    })

    assert not manager.sessions
    assert manager._actors == {}
    assert adapter.unregistered == ["call-1"]


@pytest.mark.asyncio
async def test_answered_fallback_bridges_caller_and_destination():
    manager, adapter, ari = _manager(strategy="bridge")
    session = _active_call(manager, adapter)
    await manager.transfer("call-1", "sales")

    await manager.handle_ari_event({
        "type": "StasisStart",
        "args": ["transfer-destination", "call-1"],
        "channel": {"id": session.destination_channel_id},
    })

    assert session.transfer_state == "bridge_completed"
    assert adapter.completed == [("call-1", "sales", "bridge")]
    add_channel_calls = [call for call in ari.calls if call[1].endswith("/addChannel")]
    assert any(call[2]["channel"] == "caller-1" for call in add_channel_calls)
    assert any(
        call[2]["channel"] == session.destination_channel_id
        for call in add_channel_calls
    )


@pytest.mark.asyncio
async def test_disallowed_transfer_never_calls_ari():
    manager, adapter, ari = _manager()
    _active_call(manager, adapter)

    await manager.transfer("call-1", "9999")

    assert ari.calls == []
    assert adapter.failed == [("call-1", "9999", "destination_not_allowed")]


@pytest.mark.asyncio
async def test_hangup_waits_for_in_progress_transfer_operation():
    manager, adapter, _ = _manager(strategy="bridge")
    session = _active_call(manager, adapter)
    transfer_started = asyncio.Event()
    release_transfer = asyncio.Event()

    async def begin_bridge_transfer(active_session):
        assert active_session is session
        transfer_started.set()
        await release_transfer.wait()

    manager.call_service.begin_bridge_transfer = begin_bridge_transfer
    transfer = asyncio.create_task(manager.transfer("call-1", "operator"))
    await transfer_started.wait()
    hangup = asyncio.create_task(manager.hangup("call-1"))
    await asyncio.sleep(0)

    assert "call-1" in manager.sessions
    assert hangup.done() is False

    release_transfer.set()
    await asyncio.gather(transfer, hangup)

    assert "call-1" not in manager.sessions
    assert "call-1" not in manager._actors
    assert adapter.unregistered == ["call-1"]


@pytest.mark.asyncio
async def test_close_cancels_in_progress_call_actor_before_cleanup():
    manager, adapter, _ = _manager(strategy="bridge")
    _active_call(manager, adapter)
    transfer_started = asyncio.Event()
    never_release = asyncio.Event()

    async def begin_bridge_transfer(active_session):
        transfer_started.set()
        await never_release.wait()

    manager.call_service.begin_bridge_transfer = begin_bridge_transfer
    transfer = asyncio.create_task(manager.transfer("call-1", "operator"))
    await transfer_started.wait()

    await asyncio.wait_for(manager.close(), timeout=0.2)

    assert transfer.cancelled()
    assert not manager.sessions
    assert manager._actors == {}
    assert adapter.unregistered == ["call-1"]


@pytest.mark.asyncio
async def test_close_waits_for_actor_cleanup_already_in_progress():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class BlockingDeleteARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "DELETE" and path == "channels/media-1":
                self.calls.append((method, path, dict(params or {}), json))
                cleanup_started.set()
                await release_cleanup.wait()
                return FakeResponse()
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(BlockingDeleteARIClient()),
        bridge_endpoint="operator-trunk",
        transfer_destinations={},
    )
    _active_call(manager, adapter)

    hangup = asyncio.create_task(manager.hangup("call-1"))
    await cleanup_started.wait()
    close = asyncio.create_task(manager.close())
    await asyncio.sleep(0)

    assert close.done() is False
    release_cleanup.set()
    await asyncio.gather(hangup, close)

    assert not manager.sessions
    assert manager._actors == {}
    assert adapter.unregistered == ["call-1"]


@pytest.mark.asyncio
async def test_bridge_fallback_preserves_prohibited_presentation():
    manager, adapter, ari = _manager(strategy="bridge")
    session = _active_call(manager, adapter)
    session.caller_presentation = "prohib_not_screened"

    await manager.transfer("call-1", "operator")

    originate = next(
        call for call in ari.calls
        if call[0] == "POST" and call[1] == "channels"
    )
    assert originate[2]["callerId"] == "Anonymous <anonymous>"
    assert (
        originate[3]["variables"]["CALLERID(num-pres)"]
        == "prohib_not_screened"
    )


@pytest.mark.asyncio
async def test_cleanup_releases_local_state_after_ari_delete_failure():
    class FailingDeleteARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "DELETE" and path == "channels/media-1":
                self.calls.append((method, path, dict(params or {}), json))
                return FakeResponse(500, text="temporary failure")
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = FailingDeleteARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={},
    )
    session = _active_call(manager, adapter)

    await manager.call_service.cleanup_call("call-1", hangup_caller=True)

    assert "call-1" not in manager.sessions
    assert manager.registry.by_caller("caller-1") is None
    assert manager.registry.by_media("media-1") is None
    assert adapter.unregistered == ["call-1"]
    assert any(call[:2] == ("DELETE", "bridges/bridge-1") for call in ari.calls)
    assert any(call[:2] == ("DELETE", "channels/caller-1") for call in ari.calls)
    assert session.cleanup_started is True


@pytest.mark.asyncio
async def test_canceling_cleanup_waiter_does_not_orphan_removed_registry_entry():
    delete_started = asyncio.Event()
    release_delete = asyncio.Event()

    class SlowDeleteARIClient(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            if method == "DELETE" and path == "channels/media-1":
                self.calls.append((method, path, dict(params or {}), json))
                delete_started.set()
                await release_delete.wait()
                return FakeResponse()
            return await super().request(method, path, params=params, json=json)

    adapter = FakeAdapter()
    ari = SlowDeleteARIClient()
    manager = AsteriskCallManager(
        adapter=adapter,
        ari_client=_ari_client(ari),
        bridge_endpoint="operator-trunk",
        transfer_destinations={},
    )
    _active_call(manager, adapter)

    first_waiter = asyncio.create_task(
        manager.call_service.cleanup_call("call-1", hangup_caller=True)
    )
    await delete_started.wait()
    assert "call-1" not in manager.sessions
    first_waiter.cancel()
    await asyncio.sleep(0)
    second_waiter = asyncio.create_task(
        manager.call_service.cleanup_call("call-1", hangup_caller=True)
    )
    release_delete.set()

    with pytest.raises(asyncio.CancelledError):
        await first_waiter
    await second_waiter
    assert adapter.unregistered == ["call-1"]
    assert any(call[:2] == ("DELETE", "bridges/bridge-1") for call in ari.calls)
    assert any(call[:2] == ("DELETE", "channels/caller-1") for call in ari.calls)
    assert manager.call_service._cleanup_tasks == {}
