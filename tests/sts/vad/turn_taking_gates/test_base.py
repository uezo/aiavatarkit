"""Provider-independent turn-taking contracts; no HTTP clients or API calls."""

import asyncio
from dataclasses import asdict, FrozenInstanceError, replace
import logging
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.turn_taking_gates import (
    PlaybackEstimate,
    TurnTakingDecision,
    TurnTakingGate,
    TurnTakingGateManager,
    TurnTakingSession,
)
from aiavatar.sts.vad.turn_taking_gates.bypass import TurnTakingBypassGate
import aiavatar.sts.vad.turn_taking_gates.session as session_module


class LocalTurnTakingGate(TurnTakingGate):
    """An application-owned policy with no provider client or credentials."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.full_texts = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
        self.calls.append((user_text, assistant_spoken_text, session_id))
        self.full_texts.append(assistant_full_text)
        return TurnTakingDecision(user_text != "うん", None, "local_policy")


@pytest.fixture
def playback_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    # Leave asyncio's monotonic clock intact so cancellation deadlines stay real.
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


def test_gate_requires_a_provider_implementation():
    class MissingJudge(TurnTakingGate):
        pass

    with pytest.raises(TypeError, match="abstract"):
        TurnTakingGate()
    with pytest.raises(TypeError, match="abstract"):
        MissingJudge()


@pytest.mark.parametrize("value", [True, False, None])
def test_decision_keeps_three_states_and_plain_frozen_dataclass_contract(value):
    positional = TurnTakingDecision(value, 0.4, "policy")
    keyword = TurnTakingDecision(should_take_turn=value, probability=0.4, reason="policy")
    assert positional == keyword
    assert keyword.should_take_turn is value
    assert asdict(keyword) == {"should_take_turn": value, "probability": 0.4, "reason": "policy"}
    assert replace(keyword, should_take_turn=False) == TurnTakingDecision(False, 0.4, "policy")
    with pytest.raises(FrozenInstanceError):
        keyword.should_take_turn = True


@pytest.mark.asyncio
async def test_unmanaged_session_and_evaluate_resolve_provider_continue(playback_clock):
    class ContinuingGate(TurnTakingGate):
        async def should_take_turn(self, *args, **kwargs):
            return TurnTakingDecision(None, None, "not_applicable")

    gate = ContinuingGate()
    unmanaged = gate.create_session("unmanaged")
    managed = gate.get_session("managed", create=True)
    for session in (unmanaged, managed):
        session.start_playback("chunk", "abcdefghij", 10)
    playback_clock.now += 5
    try:
        raw = await gate.should_take_turn("question", "prefix")
        assert raw.should_take_turn is None
        expected = TurnTakingDecision(True, None, "not_applicable")
        assert await unmanaged.should_take_turn("question") == expected
        assert await gate.evaluate("managed", "question") == expected
    finally:
        gate.close_session("managed")
        await asyncio.gather(unmanaged.aclose(), managed.aclose())


@pytest.mark.asyncio
async def test_local_provider_receives_session_and_estimated_spoken_prefix(playback_clock):
    judge = LocalTurnTakingGate()
    session = judge.create_session("local-session")
    assert isinstance(session, TurnTakingSession)
    assert judge.debug is False
    assert judge.bypass_enabled is True
    try:
        assert session.start_playback("chunk", "あいうえおかきくけこ", 10)
        playback_clock.now += 5
        estimate = session.estimate_playback()
        assert isinstance(estimate, PlaybackEstimate)
        assert estimate.assistant_spoken_text == "あいうえお"
        assert estimate.assistant_full_text == "あいうえおかきくけこ"
        assert await session.should_take_turn("うん") == TurnTakingDecision(False, None, "local_policy")
        assert await session.should_take_turn("質問です") == TurnTakingDecision(True, None, "local_policy")
        assert judge.calls == [
            ("うん", "あいうえお", "local-session"),
            ("質問です", "あいうえお", "local-session"),
        ]
        assert judge.full_texts == ["あいうえおかきくけこ"] * 2
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text", [None, "", "うん"])
async def test_disabling_bypass_calls_provider_even_when_idle_and_long(user_text):
    judge = LocalTurnTakingGate(bypass_enabled=False)

    def unused_default(text, duration):
        pytest.fail("Session must forward the default policy without applying it")

    session = judge.create_session("session", default_skip_condition=unused_default)
    try:
        decision = await session.should_take_turn(user_text, recorded_duration=10)
        assert decision.reason == "local_policy"
        assert judge.calls == [(user_text, "", "session")]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_standalone_gate_skips_long_input_and_can_override_session_default(playback_clock):
    judge = LocalTurnTakingGate()
    session = judge.create_session(
        "session", default_skip_condition=lambda text, duration: duration >= 1.5,
    )
    session.start_playback("chunk", "abcdefghij", 10)
    playback_clock.now += 5
    try:
        assert await session.should_take_turn("うん", recorded_duration=1.5) == TurnTakingDecision(
            True, None, "turn_take_skipped"
        )
        assert judge.calls == []
        assert await session.should_take_turn("うん", recorded_duration=0.3) == TurnTakingDecision(
            False, None, "local_policy"
        )
        judge.skip_condition = lambda text, duration: False
        assert await session.should_take_turn("うん", recorded_duration=1.5) == TurnTakingDecision(
            False, None, "local_policy"
        )
        assert judge.calls == [("うん", "abcde", "session")] * 2
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_cancelled_task_cannot_receive_closed_session_allowance():
    judge = LocalTurnTakingGate()
    session = judge.create_session("session")
    await session.aclose()
    returned_decisions = []

    async def cancelled_call():
        asyncio.current_task().cancel()
        returned_decisions.append(await session.should_take_turn("question"))

    task = asyncio.create_task(cancelled_call())
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert returned_decisions == []
        assert judge.calls == []
    finally:
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_cancels_pending_local_provider_without_provider_lifecycle(playback_clock):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class WaitingJudge(TurnTakingGate):
        async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    session = WaitingJudge().create_session("waiting-session")
    session.start_playback("chunk", "あいうえおかきくけこ", 10)
    playback_clock.now += 5
    pending = asyncio.create_task(session.should_take_turn("うん"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(session.aclose(), 1)
        assert cancelled.is_set()
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        await session.aclose()
        await asyncio.gather(pending, return_exceptions=True)


def test_generic_package_imports_without_http_or_jev():
    # Load only this package in a fresh stdlib-only process: aiavatar's existing
    # parent package imports unrelated providers, which are outside this contract.
    package_dir = Path(session_module.__file__).parent
    script = textwrap.dedent("""
        import importlib.util
        import pathlib
        import sys

        package_dir = pathlib.Path(sys.argv[1])
        name = "_standalone_turn_take"
        spec = importlib.util.spec_from_file_location(
            name, package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        for exported in (
            "TurnTakingGate", "TurnTakingGateManager", "TurnTakingDecision",
            "TurnTakingSession", "PlaybackEstimate",
        ):
            assert hasattr(module, exported), exported
        assert not hasattr(module, "JevTurnTakingGate")
        assert name + ".jev" not in sys.modules
        assert "httpx" not in sys.modules

        class LocalJudge(module.TurnTakingGate):
            async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
                return module.TurnTakingDecision(True, None, "local")

        assert isinstance(LocalJudge().create_session("local"), module.TurnTakingSession)
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(package_dir)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_synchronous_close_invalidates_now_and_aclose_joins_cleanup(playback_clock):
    entered, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowCloseGate(TurnTakingGate):
        async def should_take_turn(self, text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelling.set()
                await release.wait()
                raise

    session = SlowCloseGate().create_session("session")
    session.start_playback("chunk", "あいうえおかきくけこ", 10, transaction_id="response")
    playback_clock.now += 5
    pending = asyncio.create_task(session.should_take_turn("うん"))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        session.close()
        assert session.is_closed
        assert session.estimate_playback().assistant_spoken_text == ""
        assert pending.cancelling() == 1
        assert not session.start_playback("late", "遅い通知", 10)
        await asyncio.wait_for(cancelling.wait(), 1)
        session.close()
        closing = asyncio.create_task(session.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        assert pending.cancelling() == 1
        release.set()
        await asyncio.wait_for(closing, 1)
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not session._pending
    finally:
        release.set()
        await session.aclose()
        await asyncio.gather(*(task for task in (pending, closing) if task), return_exceptions=True)


def test_synchronous_close_does_not_require_an_event_loop():
    session = LocalTurnTakingGate().create_session("session")
    session.close()
    session.close()
    assert session.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("first_close", ["close", "aclose"])
async def test_helper_lifecycle_logs_once_only_when_debug_enabled(debug, first_close, caplog):
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    session = LocalTurnTakingGate(debug=debug).create_session("lifecycle-session")
    try:
        if first_close == "close":
            session.close()
        else:
            await session.aclose()
        session.close()
        await session.aclose()
    finally:
        await session.aclose()

    records = [record for record in caplog.records if record.name == session_module.__name__]
    expected = [
        "Turn Take Session: session=lifecycle-session, event=session_started, reason=initialized",
        "Turn Take Session: session=lifecycle-session, event=session_closed, reason=teardown",
    ] if debug else []
    assert [record.getMessage() for record in records] == expected
    assert all(record.levelno == logging.INFO for record in records)


@pytest.mark.asyncio
async def test_registry_creation_is_explicit_and_defaults_apply_only_at_creation(playback_clock):
    gate = TurnTakingGateManager([TurnTakingBypassGate(), LocalTurnTakingGate()])
    unmanaged = gate.create_session("session")
    first = replacement = None
    try:
        assert gate.get_session("session") is None
        assert not gate.is_current_session("session", unmanaged)
        assert not gate.is_current_session("session", None)
        first = gate.get_session("session", create=True, default_skip_condition=lambda text, duration: True)
        assert gate.get_session("session", create=True, default_skip_condition=lambda text, duration: False) is first
        assert gate.is_current_session("session", first)
        first.start_playback("first", "あいうえおかきくけこ", 10)
        playback_clock.now += 5
        assert (await gate.evaluate("session", "うん")).reason == "turn_take_skipped"
        first.close()
        assert gate.get_session("session") is first
        assert not gate.is_current_session("session", first)
        replacement = gate.get_session("session", create=True, default_skip_condition=lambda text, duration: False)
        assert replacement is not first
        replacement.start_playback("second", "あいうえおかきくけこ", 10)
        playback_clock.now += 5
        assert (await gate.evaluate("session", "うん")).reason == "local_policy"
        assert gate.close_session("session") is replacement
        assert replacement.is_closed
        assert gate.get_session("session") is None
        assert gate.close_session("session") is None
        assert not unmanaged.is_closed
    finally:
        gate.close_session("session")
        await asyncio.gather(*(session.aclose() for session in (unmanaged, first, replacement) if session))


@pytest.mark.asyncio
async def test_registry_isolates_connection_ids_and_default_conditions(playback_clock):
    bypass = TurnTakingBypassGate()
    gate = TurnTakingGateManager([bypass, LocalTurnTakingGate()])
    first = gate.get_session("first-connection", create=True,
                             default_skip_condition=lambda text, duration: duration >= 2)
    second = gate.get_session("second-connection", create=True,
                              default_skip_condition=lambda text, duration: duration >= 5)
    try:
        assert first is not second
        assert gate.get_session("unknown-connection") is None
        assert not gate.is_current_session("second-connection", first)
        for session in (first, second):
            session.start_playback("chunk", "あいうえおかきくけこ", 10)
        playback_clock.now += 5
        assert (await gate.evaluate("first-connection", "うん", recorded_duration=3)).reason == "turn_take_skipped"
        assert (await gate.evaluate("second-connection", "うん", recorded_duration=3)).reason == "local_policy"
        assert bypass.skip_condition is None
        assert gate.close_session("first-connection") is first
        assert gate.get_session("first-connection") is None
        assert gate.is_current_session("second-connection", second)
    finally:
        gate.close_session("first-connection")
        gate.close_session("second-connection")
        await asyncio.gather(first.aclose(), second.aclose())


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [False, True])
async def test_evaluate_rejects_missing_or_closed_state_without_creating_it(closed, caplog):
    gate = LocalTurnTakingGate(debug=True)
    session = gate.get_session("session", create=True) if closed else None
    if session:
        session.close()
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    try:
        assert await gate.evaluate("session", "質問です", recording_id="recording") == TurnTakingDecision(
            False, None, "session_missing_or_closed"
        )
        assert gate.get_session("session") is session
        assert gate.calls == []
        assert "recording_id=recording, action=discard, reason=session_missing_or_closed" in caplog.text
    finally:
        gate.close_session("session")
        if session:
            await session.aclose()


@pytest.mark.asyncio
async def test_evaluate_preserves_normal_decisions_and_logs_accept_or_reject(playback_clock, caplog):
    gate = LocalTurnTakingGate(debug=True)
    session = gate.get_session("session", create=True)
    session.start_playback("chunk", "あいうえおかきくけこ", 10)
    playback_clock.now += 5
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    try:
        assert await gate.evaluate("session", "うん", recording_id="backchannel") == TurnTakingDecision(
            False, None, "local_policy"
        )
        assert await gate.evaluate("session", "質問です", recording_id="question") == TurnTakingDecision(
            True, None, "local_policy"
        )
        assert gate.calls == [("うん", "あいうえお", "session"), ("質問です", "あいうえお", "session")]
        assert "recording_id=backchannel, action=discard, reason=local_policy" in caplog.text
        assert "recording_id=question, action=pass, reason=local_policy" in caplog.text
    finally:
        gate.close_session("session")
        await session.aclose()


@pytest.mark.asyncio
async def test_evaluate_fails_open_for_provider_error_without_logging_private_detail(playback_clock, caplog):
    class FailingGate(TurnTakingGate):
        async def should_take_turn(self, *args, **kwargs):
            raise RuntimeError("private-provider-detail")

    gate = FailingGate(debug=True)
    session = gate.get_session("session", create=True)
    session.start_playback("chunk", "あいうえおかきくけこ", 10)
    playback_clock.now += 5
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    try:
        assert await gate.evaluate("session", "質問です") == TurnTakingDecision(
            True, None, "turn_taking_error:RuntimeError"
        )
        assert "error_type=RuntimeError" in caplog.text
        assert "action=pass, reason=turn_taking_error:RuntimeError" in caplog.text
        assert "private-provider-detail" not in caplog.text
    finally:
        gate.close_session("session")
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["result", "error"])
async def test_evaluate_rejects_replaced_session_after_await(outcome, monkeypatch, caplog):
    entered, release = asyncio.Event(), asyncio.Event()
    gate = LocalTurnTakingGate(debug=True)
    session = gate.get_session("session", create=True)

    async def delayed_result(*args, **kwargs):
        entered.set()
        await release.wait()
        if outcome == "error":
            raise RuntimeError("obsolete-error")
        return TurnTakingDecision(True, 0.9, "obsolete-result")

    # Isolate evaluate's ownership check from the helper's own pending-task guard.
    monkeypatch.setattr(session, "should_take_turn", delayed_result)
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    pending = asyncio.create_task(gate.evaluate("session", "old-input"))
    replacement = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        gate.close_session("session")
        replacement = gate.get_session("session", create=True)
        release.set()
        assert await asyncio.wait_for(pending, 1) == TurnTakingDecision(
            False, None, "session_replaced_or_closed"
        )
        assert gate.is_current_session("session", replacement)
        assert "action=discard, reason=session_replaced_or_closed" in caplog.text
        assert "action=pass" not in caplog.text
    finally:
        release.set()
        gate.close_session("session")
        await asyncio.gather(pending, return_exceptions=True)
        await session.aclose()
        if replacement:
            await replacement.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_cancel", [False, True])
async def test_evaluate_propagates_cancellation_even_if_session_returns_a_result(
    suppress_cancel, monkeypatch, caplog
):
    entered = asyncio.Event()
    gate = LocalTurnTakingGate(debug=True)
    session = gate.get_session("session", create=True)

    async def cancelled_result(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not suppress_cancel:
                raise
            return TurnTakingDecision(True, 0.9, "cancelled-result")

    monkeypatch.setattr(session, "should_take_turn", cancelled_result)
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    pending = asyncio.create_task(gate.evaluate("session", "cancelled-input"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 1)
        assert "action=discard, reason=cancelled" in caplog.text
        assert "action=pass" not in caplog.text
        assert gate.is_current_session("session", session)
    finally:
        gate.close_session("session")
        await session.aclose()
        await asyncio.gather(pending, return_exceptions=True)
