"""Explicit one-use turn allowances without providers, models, or real clocks."""

import asyncio
from dataclasses import FrozenInstanceError
import logging
import threading
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.turn_taking_gates import TurnTakingDecision, TurnTakingGate, TurnTakingGateManager
from aiavatar.sts.vad.turn_taking_gates.bypass import TurnTakingBypassGate
from aiavatar.sts.vad.turn_taking_gates.session_allow import SessionAllowTurnTakingGate, SessionTurnAllowance
import aiavatar.sts.vad.turn_taking_gates.session as playback_module
import aiavatar.sts.vad.turn_taking_gates.session_allow as allowance_module


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    fake_time = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr(allowance_module, "time", fake_time)
    monkeypatch.setattr(playback_module, "time", fake_time)
    return clock


class DecliningGate(TurnTakingGate):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.full_texts = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
        self.calls.append((user_text, assistant_spoken_text, session_id))
        self.full_texts.append(assistant_full_text)
        return TurnTakingDecision(False, 0.1, "declined")


async def classify(gate, session_id):
    return await gate.should_take_turn("question", "spoken prefix", session_id=session_id)


@pytest.mark.asyncio
async def test_allowance_is_one_use_and_independent_for_each_session(clock):
    gate = SessionAllowTurnTakingGate()
    gate.allow("first")
    gate.allow("second", expires_in=10, reason="manual_interrupt")
    assert gate.get_allowance("first") == SessionTurnAllowance(400.0, "session_allow")
    assert gate.get_allowance("second") == SessionTurnAllowance(110.0, "manual_interrupt")

    assert await classify(gate, "first") == TurnTakingDecision(True, None, "session_allow")
    assert gate.get_allowance("first") is None
    assert await classify(gate, "first") == TurnTakingDecision(None, None, "session_allow_inactive")
    assert await classify(gate, "second") == TurnTakingDecision(True, None, "manual_interrupt")
    assert await classify(gate, "unknown") == TurnTakingDecision(None, None, "session_allow_inactive")


@pytest.mark.asyncio
async def test_allow_overwrites_pending_expiry_and_reason_and_read_is_immutable(clock):
    gate = SessionAllowTurnTakingGate(default_expires_in=5)
    gate.allow("session", reason="earlier")
    earlier = gate.get_allowance("session")
    assert earlier == SessionTurnAllowance(105.0, "earlier")
    with pytest.raises(FrozenInstanceError):
        earlier.reason = "mutated"

    clock.now += 2
    gate.allow("session", expires_in=20, reason="replacement")
    assert earlier == SessionTurnAllowance(105.0, "earlier")
    assert gate.get_allowance("session") == SessionTurnAllowance(122.0, "replacement")
    clock.now = 106.0
    assert await classify(gate, "session") == TurnTakingDecision(True, None, "replacement")


@pytest.mark.asyncio
async def test_expiry_boundary_is_exclusive_and_read_removes_expired_allowance(clock):
    gate = SessionAllowTurnTakingGate(default_expires_in=2)
    for session_id in ("before", "at_boundary", "read_at_boundary"):
        gate.allow(session_id)
    clock.now = 101.99
    assert (await classify(gate, "before")).should_take_turn
    clock.now = 102.0
    assert await classify(gate, "at_boundary") == TurnTakingDecision(None, None, "session_allow_expired")
    assert await classify(gate, "at_boundary") == TurnTakingDecision(None, None, "session_allow_inactive")
    assert gate.get_allowance("read_at_boundary") is None
    assert (await classify(gate, "read_at_boundary")).reason == "session_allow_inactive"


@pytest.mark.asyncio
async def test_adding_allowance_purges_other_expired_sessions(clock):
    gate = SessionAllowTurnTakingGate(default_expires_in=1)
    gate.allow("expired")
    gate.allow("still_valid", expires_in=10)
    clock.now += 1
    gate.allow("new")
    assert (await classify(gate, "expired")).reason == "session_allow_inactive"
    assert (await classify(gate, "still_valid")).should_take_turn
    assert (await classify(gate, "new")).should_take_turn


@pytest.mark.asyncio
async def test_release_and_reset_are_idempotent_and_session_scoped(clock):
    gate = SessionAllowTurnTakingGate()
    gate.allow("first")
    gate.allow("second")
    gate.release("first")
    gate.release("first")
    assert (await classify(gate, "first")).should_take_turn is None
    assert gate.get_allowance("second") is not None
    gate.reset_session("second")
    gate.reset_session("second")
    assert (await classify(gate, "second")).should_take_turn is None


def test_allowance_validates_expiry_and_session_id(clock):
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="default_expires_in"):
            SessionAllowTurnTakingGate(default_expires_in=invalid)
        gate = SessionAllowTurnTakingGate()
        with pytest.raises(ValueError, match="expires_in"):
            gate.allow("session", expires_in=invalid)
    with pytest.raises(ValueError, match="session_id"):
        SessionAllowTurnTakingGate().allow("")


@pytest.mark.asyncio
async def test_concurrent_thread_evaluations_consume_only_one_allowance(clock):
    gate = SessionAllowTurnTakingGate()
    gate.allow("session")
    start = threading.Barrier(2, timeout=5)

    def evaluate_in_thread():
        start.wait()
        return asyncio.run(classify(gate, "session"))

    results = await asyncio.gather(
        asyncio.to_thread(evaluate_in_thread), asyncio.to_thread(evaluate_in_thread),
    )
    assert sum(result.should_take_turn is True for result in results) == 1
    assert sum(result.should_take_turn is None for result in results) == 1
    assert {result.reason for result in results} == {"session_allow", "session_allow_inactive"}


@pytest.mark.asyncio
async def test_manager_stops_at_allowance_then_uses_later_gate_for_next_evaluation(clock):
    allowance = SessionAllowTurnTakingGate()
    fallback = DecliningGate()
    manager = TurnTakingGateManager([allowance, fallback])
    state = manager.get_session("session", create=True)
    state.start_playback("chunk", "abcdefghij", 10)
    clock.now += 5
    allowance.allow("session", reason="explicit_input")
    try:
        assert await manager.evaluate("session", "question") == TurnTakingDecision(True, None, "explicit_input")
        assert fallback.calls == []
        assert await manager.evaluate("session", "yes") == TurnTakingDecision(False, 0.1, "declined")
        assert fallback.calls == [("yes", "abcde", "session")]
        assert fallback.full_texts == ["abcdefghij"]
    finally:
        allowance.release("session")
        manager.close_session("session")
        await state.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bypass", ["skip", "grace", "idle"])
async def test_common_bypasses_leave_allowance_for_next_actual_gate_evaluation(clock, bypass):
    allowance = SessionAllowTurnTakingGate()
    bypass_gate = TurnTakingBypassGate(
        skip_condition=(lambda text, duration: True) if bypass == "skip" else None,
        response_end_grace_seconds=0.3 if bypass == "grace" else 0.0,
    )
    manager = TurnTakingGateManager([bypass_gate, allowance])
    state = manager.get_session("session", create=True)
    if bypass != "idle":
        state.start_playback("chunk", "abcdefghij", 10, transaction_id="response")
        state.mark_response_final("chunk", transaction_id="response")
        clock.now += 9.8 if bypass == "grace" else 5
    allowance.allow("session", reason="pending_input")
    try:
        result = await manager.evaluate("session", "question")
        assert result.should_take_turn
        assert result.reason == {
            "skip": "turn_take_skipped", "grace": "playback_response_end_grace", "idle": "playback_idle",
        }[bypass]
        assert allowance.get_allowance("session") is not None

        bypass_gate.skip_condition = lambda text, duration: False
        bypass_gate.response_end_grace_seconds = 0.0
        state.start_playback("next", "klmnopqrst", 10, transaction_id="next_response")
        clock.now += 5
        assert await manager.evaluate("session", "next question") == TurnTakingDecision(True, None, "pending_input")
        assert allowance.get_allowance("session") is None
    finally:
        allowance.release("session")
        manager.close_session("session")
        await state.aclose()


@pytest.mark.asyncio
async def test_debug_reports_consumed_inactive_and_expired_allowances(clock, caplog):
    gate = SessionAllowTurnTakingGate(default_expires_in=1, debug=True)
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates.session_allow")
    gate.allow("session", reason="explicit_input")
    await classify(gate, "session")
    await classify(gate, "session")
    gate.allow("session")
    clock.now += 1
    await classify(gate, "session")
    assert "session=session" in caplog.text
    assert "explicit_input" in caplog.text
    assert "inactive" in caplog.text
    assert "expired" in caplog.text
    caplog.clear()
    gate.debug = False
    await classify(gate, "session")
    assert caplog.records == []
