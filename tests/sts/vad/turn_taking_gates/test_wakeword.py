"""Local wakeword admission tests with caller-owned activity and fake clocks."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.turn_taking_gates import (
    TurnTakingDecision, TurnTakingGate, TurnTakingGateManager,
)
from aiavatar.sts.vad.turn_taking_gates.wakeword import WakewordGate
from aiavatar.sts.vad.turn_taking_gates.bypass import TurnTakingBypassGate
import aiavatar.sts.vad.turn_taking_gates.session as playback_module
import aiavatar.sts.vad.turn_taking_gates.wakeword as wakeword_module


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(monotonic=100.0)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now.astimezone(tz) if tz else clock.now.replace(tzinfo=None)

    clock.now = FixedDateTime(2026, 10, 6, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(wakeword_module, "datetime", FixedDateTime)
    monkeypatch.setattr(playback_module, "time", SimpleNamespace(monotonic=lambda: clock.monotonic))
    return clock


class RecordingGate(TurnTakingGate):
    def __init__(self, should_take_turn=False):
        super().__init__()
        self.decision = should_take_turn
        self.calls = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
        self.calls.append((user_text, session_id))
        return TurnTakingDecision(self.decision, None, "recording_gate")


@pytest.mark.asyncio
@pytest.mark.parametrize("wakewords", [None, []])
async def test_unconfigured_literal_gate_continues_without_activity_lookup(wakewords):
    async def unexpected_lookup(session_id):
        pytest.fail("A disabled literal gate must not read conversation state")

    gate = WakewordGate(wakewords, get_last_conversation_at=unexpected_lookup)
    decision = await gate.should_take_turn(None, None, session_id="session")
    assert decision.should_take_turn is None


@pytest.mark.asyncio
async def test_literal_match_preserves_whole_utterance_and_does_not_bypass_next_gate(clock):
    utterance = "  ねえアイちゃん、今日の天気を教えて。  "
    gate = WakewordGate(["アイちゃん"])
    next_gate = RecordingGate()
    manager = TurnTakingGateManager([gate, next_gate], bypass_enabled=False)
    state = manager.get_session("session", create=True)
    state.start_playback("chunk", "説明を続けます", 10)
    clock.monotonic += 5
    try:
        raw = await gate.should_take_turn(utterance, None)
        assert raw.should_take_turn is None
        result = await manager.evaluate("session", utterance)
        assert result.should_take_turn is False
        assert next_gate.calls == [(utterance, "session")]
        assert next_gate.calls[0][0] is utterance
    finally:
        manager.close_session("session")
        await state.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "", " \n\t", "HELLO", "独り言です"])
async def test_sleeping_gate_blocks_missing_or_nonmatching_text(text):
    decision = await WakewordGate(["hello"]).should_take_turn(text, None)
    assert decision.should_take_turn is False


@pytest.mark.asyncio
@pytest.mark.parametrize("playing", [False, True])
async def test_standalone_wakeword_is_not_bypassed_by_idle_or_long_input(clock, playing):
    gate = WakewordGate(["hello"])
    state = gate.get_session(
        "session", create=True, default_skip_condition=lambda text, duration: duration >= 3,
    )
    if playing:
        state.start_playback("chunk", "0123456789", 10)
        clock.monotonic += 5
    try:
        decision = await gate.evaluate(
            "session", "unaddressed speech", recorded_duration=5 if playing else 0.5,
        )
        assert decision.should_take_turn is False
        assert decision.reason == "wakeword_missing"
    finally:
        gate.close_session("session")
        await state.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("age,expected", [(None, False), (59.999, True), (60.0, False), (60.001, False)])
async def test_activity_timeout_is_strict_and_none_means_no_conversation(clock, age, expected):
    calls = []

    async def activity(session_id):
        calls.append(session_id)
        return None if age is None else clock.now - timedelta(seconds=age)

    gate = WakewordGate(["hello"], get_last_conversation_at=activity)
    decision = await gate.should_take_turn("continuation", None, session_id="session")
    assert (decision.should_take_turn is None) is expected
    assert calls == ["session"]


@pytest.mark.asyncio
async def test_activity_uses_session_id_timezone_and_external_updates_without_renewing(clock):
    recent = clock.now.astimezone(timezone(timedelta(hours=9))) - timedelta(seconds=30)
    activity_by_session = {"awake": recent, "asleep": None}
    calls = []

    async def activity(session_id):
        calls.append(session_id)
        return activity_by_session.get(session_id)

    gate = WakewordGate(["hello"], get_last_conversation_at=activity)
    assert (await gate.should_take_turn("question", None, session_id="awake")).should_take_turn is None
    assert (await gate.should_take_turn("question", None, session_id="asleep")).should_take_turn is False
    assert (await gate.should_take_turn("hello and question", None, session_id="asleep")).should_take_turn is None
    # A wakeword hit is not an activity write or an implicit per-gate wake latch.
    assert activity_by_session == {"awake": recent, "asleep": None}
    assert (await gate.should_take_turn("next question", None, session_id="asleep")).should_take_turn is False
    activity_by_session["asleep"] = clock.now
    assert (await gate.should_take_turn("next question", None, session_id="asleep")).should_take_turn is None
    assert (await gate.should_take_turn("question", None, session_id=None)).should_take_turn is False
    assert None not in calls


@pytest.mark.asyncio
async def test_zero_timeout_requires_detection_without_reading_activity():
    async def unexpected_lookup(session_id):
        pytest.fail("Zero timeout disables continuation lookup")

    gate = WakewordGate(["hello"], wakeword_timeout=0, get_last_conversation_at=unexpected_lookup)
    assert (await gate.should_take_turn("question", None, session_id="session")).should_take_turn is False
    assert (await gate.should_take_turn("hello question", None, session_id="session")).should_take_turn is None


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["timestamp", 123, datetime(2026, 10, 6)])
async def test_invalid_activity_results_block_even_a_literal_match(invalid):
    async def activity(session_id):
        return invalid

    decision = await WakewordGate(["hello"], get_last_conversation_at=activity).should_take_turn(
        "hello", None, session_id="session",
    )
    assert decision.should_take_turn is False
    assert decision.reason == "wakeword_error"


@pytest.mark.asyncio
async def test_callback_errors_fail_closed_without_exposing_content(caplog):
    async def activity(session_id):
        raise RuntimeError("private-transcript-and-secret")

    gate = WakewordGate(["hello"], get_last_conversation_at=activity)
    state = gate.get_session("session", create=True)
    try:
        decision = await gate.evaluate("session", "hello")
        assert decision.should_take_turn is False
        assert decision.reason == "wakeword_error"
        assert "RuntimeError" in caplog.text
        assert "private-transcript-and-secret" not in caplog.text
    finally:
        gate.close_session("session")
        await state.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bypass", ["idle", "duration", "grace", "no_text", "empty_prefix"])
async def test_wakeword_before_bypass_checks_idle_long_grace_and_empty_inputs(clock, bypass):
    next_gate = RecordingGate(True)
    manager = TurnTakingGateManager(
        [
            WakewordGate(["hello"]),
            TurnTakingBypassGate(
                response_end_grace_seconds=0.5,
                skip_condition=(lambda text, duration: duration >= 3) if bypass == "duration" else None,
            ),
            next_gate,
        ],
    )
    state = manager.get_session("session", create=True)
    if bypass != "idle":
        state.start_playback("chunk", "0123456789", 10, transaction_id="response")
        state.mark_response_final("chunk", transaction_id="response")
        clock.monotonic += {"grace": 9.8, "empty_prefix": 0.0}.get(bypass, 5.0)
    try:
        decision = await manager.evaluate(
            "session", None if bypass == "no_text" else "unaddressed speech", recorded_duration=5,
        )
        assert decision.should_take_turn is False
        assert next_gate.calls == []
        # Wakeword passes admission even while the ordinary classifier is bypassed.
        decision = await manager.evaluate("session", "hello, complete request", recorded_duration=5)
        assert decision.should_take_turn is True
    finally:
        manager.close_session("session")
        await state.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["cancel", "close", "supersede"])
@pytest.mark.parametrize("suppress_cancellation", [False, True])
async def test_pending_callback_cannot_emit_stale_admission(clock, cleanup, suppress_cancellation):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def activity(session_id):
        nonlocal calls
        calls += 1
        if calls > 1:
            return None
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            if suppress_cancellation:
                return clock.now
            raise

    gate = WakewordGate(["hello"], get_last_conversation_at=activity)
    manager = TurnTakingGateManager([gate], bypass_enabled=False)
    state = manager.get_session("session", create=True)
    task = asyncio.create_task(manager.evaluate("session", "old speech"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cleanup == "cancel":
            task.cancel()
        elif cleanup == "close":
            manager.close_session("session")
        else:
            decision = await manager.evaluate("session", "hello, new request")
            assert decision.should_take_turn is True
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert cancelled.is_set()
    finally:
        manager.close_session("session")
        await state.aclose()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("options", [
    {"wakewords": "hello"}, {"wakewords": [""]}, {"wakewords": [1]},
    {"get_last_conversation_at": 1},
    *[{"wakeword_timeout": value} for value in (-1, True, "60", float("nan"), float("inf"))],
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        WakewordGate(**options)
