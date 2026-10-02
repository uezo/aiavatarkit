"""Ordered gate composition using only local policies and a fake playback clock."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.vad.turn_taking_gates import (
    TurnTakingDecision,
    TurnTakingGate,
    TurnTakingGateManager,
)
import aiavatar.sts.vad.turn_taking_gates.session as session_module


class LocalGate(TurnTakingGate):
    def __init__(self, name, outcome, calls, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.outcome = outcome
        self.calls = calls
        self.full_texts = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None):
        self.calls.append((self.name, user_text, assistant_spoken_text, session_id))
        self.full_texts.append(assistant_full_text)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if callable(self.outcome):
            return await self.outcome(
                user_text, assistant_spoken_text, session_id=session_id,
                assistant_full_text=assistant_full_text,
            )
        return self.outcome

    async def evaluate(self, *args, **kwargs):
        pytest.fail("The manager must call the child's classifier, not its evaluate method")


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


def start_playback(manager, session_id="session", text="abcdefghij", final=False):
    session = manager.get_session(session_id, create=True)
    assert session.start_playback("chunk", text, 10, transaction_id="response")
    if final:
        assert session.mark_response_final("chunk", transaction_id="response")
    return session


async def close_sessions(manager, *session_ids):
    for session_id in session_ids:
        session = manager.close_session(session_id)
        if session is not None:
            await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_accept", [0, 1, 2])
async def test_children_run_in_order_until_first_acceptance(first_accept):
    calls = []
    decisions = [
        TurnTakingDecision(index >= first_accept, 0.8 if index >= first_accept else 0.1, f"gate_{index}")
        for index in range(3)
    ]
    manager = TurnTakingGateManager([
        LocalGate(str(index), decision, calls) for index, decision in enumerate(decisions)
    ])

    result = await manager.should_take_turn(
        "question", "spoken prefix", session_id="session",
        assistant_full_text="spoken prefix plus remainder",
    )

    assert result is decisions[first_accept]
    assert calls == [
        (str(index), "question", "spoken prefix", "session")
        for index in range(first_accept + 1)
    ]
    for index, child in enumerate(manager.gates):
        assert child.full_texts == (["spoken prefix plus remainder"] if index <= first_accept else [])


@pytest.mark.asyncio
async def test_all_children_declining_rejects_the_turn():
    calls = []
    manager = TurnTakingGateManager([
        LocalGate("first", TurnTakingDecision(False, 0.1, "first_declined"), calls),
        LocalGate("second", TurnTakingDecision(False, 0.2, "second_declined"), calls),
    ])

    assert await manager.should_take_turn("yes", "prefix", session_id="session") == TurnTakingDecision(
        False, None, "all_gates_declined"
    )
    assert [call[0] for call in calls] == ["first", "second"]


@pytest.mark.asyncio
async def test_empty_manager_allows_the_turn():
    for manager in (TurnTakingGateManager(), TurnTakingGateManager([])):
        assert await manager.should_take_turn("question", "prefix") == TurnTakingDecision(True, None, "no_gates")


@pytest.mark.asyncio
async def test_only_manager_skip_grace_and_playback_state_apply(clock):
    calls = []
    child = LocalGate(
        "child", TurnTakingDecision(False, 0.1, "declined"), calls,
        skip_condition=lambda text, duration: True,
        response_end_grace_seconds=10.0,
    )
    manager = TurnTakingGateManager([child], skip_condition=lambda text, duration: False)
    start_playback(manager, final=True)
    clock.now = 109.8
    try:
        # Neither the child's always-skip policy nor its large grace bypasses it.
        result = await manager.evaluate("session", "yes", recorded_duration=20)
        assert result == TurnTakingDecision(False, None, "all_gates_declined")
        assert calls == [("child", "yes", "abcdefghi", "session")]
        assert child.get_session("session") is None

        manager.response_end_grace_seconds = 0.3
        assert (await manager.evaluate("session", "yes")).reason == "playback_response_end_grace"
        manager.skip_condition = lambda text, duration: duration >= 2
        clock.now = 105.0
        assert (await manager.evaluate("session", "long question", recorded_duration=2)).reason == "turn_take_skipped"
        assert len(calls) == 1
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_condition", [False, True], ids=["vad_default", "manager_override"])
async def test_manager_fits_the_single_vad_gate_and_uses_outer_condition(clock, explicit_condition):
    calls, notifications = [], []
    child = LocalGate(
        "child", TurnTakingDecision(False, None, "declined"), calls,
        skip_condition=lambda text, duration: True,
    )
    manager = TurnTakingGateManager(
        [child], skip_condition=(lambda text, duration: False) if explicit_condition else None,
    )
    vad = SpeechDetectorDummy(turn_taking_gate=manager, on_recording_started_min_duration=1.5)
    assert vad._init_turn_taking_session("session") is None
    state = manager.get_session("session")
    assert state is not None
    state.start_playback("chunk", "abcdefghij", 10)
    clock.now += 5

    @vad.on_speech_detected
    async def on_speech(*args):
        notifications.append(args)

    try:
        await vad._execute_on_speech_detected(b"audio", "short", {}, 0.3, "session")
        assert notifications == []
        assert len(calls) == 1
        await vad._execute_on_speech_detected(b"audio", "long", {}, 1.5, "session")
        assert len(calls) == (2 if explicit_condition else 1)
        assert [item[1] for item in notifications] == ([] if explicit_condition else ["long"])
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_all_children_use_one_context_snapshot_despite_playback_updates(clock, monkeypatch):
    calls = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_decline(*args, **kwargs):
        entered.set()
        await release.wait()
        return TurnTakingDecision(False, None, "first_declined")

    manager = TurnTakingGateManager([
        LocalGate("first", delayed_decline, calls),
        LocalGate("second", TurnTakingDecision(True, None, "second_accepted"), calls),
    ])
    state = start_playback(manager)
    estimate_calls = []
    original_estimate = state.estimate_playback

    def estimate_once(**kwargs):
        estimate_calls.append(kwargs)
        return original_estimate(**kwargs)

    monkeypatch.setattr(state, "estimate_playback", estimate_once)
    clock.now = 105.0
    pending = asyncio.create_task(manager.evaluate("session", "question"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        clock.now = 108.0
        state.start_playback("next", "KLMNOPQRST", 10, transaction_id="response")
        clock.now = 112.0
        release.set()
        assert (await asyncio.wait_for(pending, 1)).should_take_turn
        assert calls == [
            ("first", "question", "abcde", "session"),
            ("second", "question", "abcde", "session"),
        ]
        assert [child.full_texts for child in manager.gates] == [["abcdefghij"], ["abcdefghij"]]
        assert len(estimate_calls) == 1
        assert original_estimate().assistant_full_text == "abcdefghijKLMNOPQRST"
    finally:
        release.set()
        await close_sessions(manager, "session")
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_new_input_cancels_the_whole_old_chain_without_waiting_for_cleanup(clock):
    calls = []
    entered, cancelled, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed_second(user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None):
        if user_text == "old":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release_cleanup.wait()
                raise
        return TurnTakingDecision(False, None, "second_declined")

    accepted = TurnTakingDecision(True, 0.8, "third_accepted")
    manager = TurnTakingGateManager([
        LocalGate("first", TurnTakingDecision(False, None, "first_declined"), calls),
        LocalGate("second", delayed_second, calls),
        LocalGate("third", accepted, calls),
    ])
    start_playback(manager)
    clock.now += 5
    old = asyncio.create_task(manager.evaluate("session", "old"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert await asyncio.wait_for(manager.evaluate("session", "new"), 1) is accepted
        await asyncio.wait_for(cancelled.wait(), 1)
        assert not old.done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await old
        assert [name for name, text, _, _ in calls if text == "old"] == ["first", "second"]
        assert [name for name, text, _, _ in calls if text == "new"] == ["first", "second", "third"]
    finally:
        release_cleanup.set()
        await close_sessions(manager, "session")
        await asyncio.gather(old, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_joins_the_chain_without_affecting_another_session(clock):
    calls = []
    entered, cancelled, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed_gate(user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None):
        if session_id == "first":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release_cleanup.wait()
                raise
        return TurnTakingDecision(True, None, "accepted")

    child = LocalGate("child", delayed_gate, calls)
    manager = TurnTakingGateManager([child])
    first = start_playback(manager, "first")
    second = start_playback(manager, "second", text="1234567890")
    clock.now += 5
    pending = asyncio.create_task(manager.evaluate("first", "question"))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert manager.close_session("first") is first
        closing = asyncio.create_task(first.aclose())
        await asyncio.wait_for(cancelled.wait(), 1)
        assert not closing.done()
        assert not (await manager.evaluate("first", "late input")).should_take_turn
        assert (await manager.evaluate("second", "another question")).should_take_turn
        assert not second.is_closed
        assert calls[-1] == ("child", "another question", "12345", "second")
        assert child.get_session("first") is None
        assert child.get_session("second") is None
        release_cleanup.set()
        await asyncio.wait_for(closing, 1)
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        release_cleanup.set()
        await first.aclose()
        await close_sessions(manager, "first", "second")
        await asyncio.gather(*[task for task in (pending, closing) if task], return_exceptions=True)


@pytest.mark.asyncio
async def test_child_error_propagates_raw_and_outer_evaluation_fails_open(clock, caplog):
    calls = []
    manager = TurnTakingGateManager([
        LocalGate("first", TurnTakingDecision(False, 0.1, "declined"), calls),
        LocalGate("second", ValueError("private-error-detail"), calls),
        LocalGate("third", TurnTakingDecision(True, 0.8, "accepted"), calls),
    ], debug=True)
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates")
    with pytest.raises(ValueError, match="private-error-detail"):
        await manager.should_take_turn("question", "prefix", session_id="session")
    assert [call[0] for call in calls] == ["first", "second"]
    calls.clear()
    start_playback(manager)
    clock.now += 5
    try:
        result = await manager.evaluate("session", "question")
        assert result == TurnTakingDecision(True, None, "turn_taking_error:ValueError")
        assert [call[0] for call in calls] == ["first", "second"]
        assert "gate_index=2, gate=LocalGate, error_type=ValueError" in caplog.text
        assert "private-error-detail" not in caplog.text
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
async def test_child_cancellation_propagates_without_evaluating_later_children():
    calls = []
    manager = TurnTakingGateManager([
        LocalGate("first", asyncio.CancelledError(), calls),
        LocalGate("second", TurnTakingDecision(True, None, "accepted"), calls),
    ])
    with pytest.raises(asyncio.CancelledError):
        await manager.should_take_turn("question", "prefix", session_id="session")
    assert [call[0] for call in calls] == ["first"]


@pytest.mark.asyncio
@pytest.mark.parametrize("debug", [False, True])
async def test_manager_debug_identifies_each_child_decision(debug, caplog):
    calls = []
    manager = TurnTakingGateManager([
        LocalGate("first", TurnTakingDecision(False, 0.1, "first_declined"), calls),
        LocalGate("second", TurnTakingDecision(True, 0.8, "second_accepted"), calls),
    ], debug=debug)
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates.manager")
    await manager.should_take_turn("question", "prefix", session_id="session")
    messages = [record.message for record in caplog.records if record.name.endswith(".manager")]
    if debug:
        assert len(messages) == 2
        assert "session=session, gate_index=1, gate=LocalGate" in messages[0]
        assert "should_take_turn=False, probability=0.1, reason=first_declined" in messages[0]
        assert "gate_index=2, gate=LocalGate" in messages[1]
        assert "should_take_turn=True, probability=0.8, reason=second_accepted" in messages[1]
    else:
        assert messages == []
