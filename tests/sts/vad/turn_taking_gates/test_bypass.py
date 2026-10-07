"""Explicit playback bypass policies with fake clocks and local gates."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.turn_taking_gates import TurnTakingDecision, TurnTakingGate, TurnTakingGateManager
from aiavatar.sts.vad.turn_taking_gates.bypass import TurnTakingBypassGate
import aiavatar.sts.vad.turn_taking_gates.session as session_module


class LocalTurnTakingGate(TurnTakingGate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, **kwargs):
        self.calls.append((user_text, assistant_spoken_text, session_id))
        return TurnTakingDecision(user_text != "うん", None, "local_policy")


@pytest.fixture
def playback_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


@pytest.mark.asyncio
async def test_explicit_bypass_preserves_skip_grace_and_closed_session_behavior(playback_clock):
    judge = LocalTurnTakingGate()
    bypass = TurnTakingBypassGate(
        response_end_grace_seconds=0.3, skip_condition=lambda text, duration: duration >= 5
    )
    session = TurnTakingGateManager([bypass, judge]).create_session("local-session")
    try:
        session.start_playback("last", "あいうえおかきくけこ", 10, transaction_id="response")
        assert session.mark_response_final("last", transaction_id="response")
        playback_clock.now += 5
        assert await session.should_take_turn("長い発話", recorded_duration=5.0) == TurnTakingDecision(
            True, None, "turn_take_skipped"
        )
        assert judge.calls == []
        assert (await session.should_take_turn("うん")).reason == "local_policy"
        playback_clock.now = 109.75
        assert await session.should_take_turn("うん") == TurnTakingDecision(
            True, None, "playback_response_end_grace"
        )
        await session.aclose()
        await session.aclose()
        assert session.is_closed
        assert not session.start_playback("late", "遅い通知", 3)
        assert await session.should_take_turn("次の発話") == TurnTakingDecision(
            True, None, "turn_take_session_closed"
        )
        assert judge.calls == [("うん", "あいうえお", "local-session")]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_shared_gate_keeps_different_session_defaults_and_input_arguments(playback_clock):
    bypass = TurnTakingBypassGate()
    gate = TurnTakingGateManager([bypass, LocalTurnTakingGate()])
    seen = []

    def first_default(text, duration):
        seen.append((text, duration))
        return duration >= 2

    first = gate.create_session("first", default_skip_condition=first_default)
    second = gate.create_session("second", default_skip_condition=lambda text, duration: duration >= 5)
    sessions = [first, second, gate.create_session("no-default")]
    try:
        for session in sessions:
            session.start_playback("chunk", "あいうえおかきくけこ", 10)
        playback_clock.now += 5
        assert (await first.should_take_turn("うん", recorded_duration=3)).reason == "turn_take_skipped"
        assert (await second.should_take_turn("うん", recorded_duration=3)).reason == "local_policy"
        assert (await sessions[2].should_take_turn("うん", recorded_duration=10)).reason == "local_policy"
        assert (await first.should_take_turn("質問です")).reason == "local_policy"
        assert seen == [("うん", 3), ("質問です", 0.0)]
        assert bypass.skip_condition is None
    finally:
        await asyncio.gather(*(session.aclose() for session in sessions))


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", [False, True])
async def test_explicit_gate_condition_overrides_session_default(skip, playback_clock):
    def unused_default(text, duration):
        pytest.fail("A configured gate condition must override the session default")

    gate = LocalTurnTakingGate()
    bypass = TurnTakingBypassGate(skip_condition=lambda text, duration: skip)
    session = TurnTakingGateManager([bypass, gate]).create_session("session", default_skip_condition=unused_default)
    session.start_playback("chunk", "あいうえおかきくけこ", 10)
    playback_clock.now += 5
    try:
        decision = await session.should_take_turn("うん", recorded_duration=20)
        assert decision.reason == ("turn_take_skipped" if skip else "local_policy")
        assert len(gate.calls) == (0 if skip else 1)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_skip_condition_error_allows_turn_and_cleans_up_for_next_input(playback_clock, caplog):
    def condition(text, duration):
        if text == "失敗":
            raise RuntimeError("private-exception-detail")
        return False

    gate = LocalTurnTakingGate()
    bypass = TurnTakingBypassGate(skip_condition=condition)
    session = TurnTakingGateManager([bypass, gate], debug=True).create_session("session")
    session.start_playback("chunk", "あいうえおかきくけこ", 10)
    playback_clock.now += 5
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    try:
        assert await session.should_take_turn("失敗") == TurnTakingDecision(
            True, None, "turn_taking_skip_condition_error"
        )
        assert gate.calls == []
        assert session._current_task is None
        assert not session._pending
        assert (await session.should_take_turn("うん")).reason == "local_policy"
    finally:
        await session.aclose()
    assert "RuntimeError" in caplog.text
    assert "private-exception-detail" not in caplog.text
    assert "reason=turn_taking_skip_condition_error" in caplog.text


@pytest.mark.asyncio
async def test_skip_condition_cancellation_propagates_and_releases_registration(playback_clock):
    def condition(text, duration):
        raise asyncio.CancelledError

    gate = LocalTurnTakingGate()
    session = TurnTakingGateManager([
        TurnTakingBypassGate(skip_condition=condition), gate,
    ]).create_session("session")
    session.start_playback("chunk", "あいうえおかきくけこ", 10)
    playback_clock.now += 5
    try:
        with pytest.raises(asyncio.CancelledError):
            await session.should_take_turn("うん")
        assert gate.calls == []
        assert session._current_task is None
        assert not session._pending
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text", [None, "", " \t\n"])
async def test_missing_text_bypasses_local_gate_and_cancels_previous_input(
    user_text, playback_clock
):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    conditions = []

    def condition(text, duration):
        conditions.append(text)
        return False

    class PendingGate(TurnTakingGate):
        async def should_take_turn(self, text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    session = TurnTakingGateManager([
        TurnTakingBypassGate(skip_condition=condition), PendingGate(),
    ]).create_session("session")
    session.start_playback("chunk", "あいうえおかきくけこ", 10)
    playback_clock.now += 5
    pending = asyncio.create_task(session.should_take_turn("old"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert await session.should_take_turn(user_text) == TurnTakingDecision(
            True, None, "turn_taking_no_text"
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 1)
        assert cancelled.is_set()
        assert conditions == ["old"]
    finally:
        await session.aclose()
        await asyncio.gather(pending, return_exceptions=True)


def test_invalid_conditions_are_rejected():
    with pytest.raises(ValueError, match="skip_condition"):
        TurnTakingBypassGate(skip_condition=False)
    with pytest.raises(ValueError, match="default_skip_condition"):
        LocalTurnTakingGate().create_session("session", default_skip_condition=False)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, None, "0.3"])
def test_invalid_grace_is_rejected(value):
    with pytest.raises(ValueError, match="response_end_grace_seconds"):
        TurnTakingBypassGate(response_end_grace_seconds=value)


@pytest.mark.asyncio
async def test_raw_bypass_without_playback_defers_unless_input_policy_matches():
    gate = TurnTakingBypassGate()
    assert await gate.should_take_turn("question", "") == TurnTakingDecision(
        None, None, "turn_taking_no_bypass"
    )
    assert await gate.should_take_turn(None, None) == TurnTakingDecision(
        True, None, "turn_taking_no_text"
    )
    assert await gate.should_take_turn(
        "question", "", recorded_duration=2,
        default_skip_condition=lambda text, duration: duration >= 1.5,
    ) == TurnTakingDecision(True, None, "turn_take_skipped")
