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
from aiavatar.sts.vad.turn_taking_gates.bypass import TurnTakingBypassGate
import aiavatar.sts.vad.turn_taking_gates.session as session_module


class LocalGate(TurnTakingGate):
    def __init__(self, name, outcome, calls, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.outcome = outcome
        self.calls = calls
        self.full_texts = []
        self.contexts = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
        self.calls.append((self.name, user_text, assistant_spoken_text, session_id))
        self.full_texts.append(assistant_full_text)
        self.contexts.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if callable(self.outcome):
            return await self.outcome(
                user_text, assistant_spoken_text, session_id=session_id,
                assistant_full_text=assistant_full_text, **kwargs,
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
@pytest.mark.parametrize("terminal_value", [True, False])
@pytest.mark.parametrize("first_terminal", [0, 1, 2])
async def test_children_run_in_order_until_first_terminal_decision(first_terminal, terminal_value):
    calls = []
    decisions = [
        TurnTakingDecision(
            should_take_turn=terminal_value if index >= first_terminal else None,
            probability=0.8 if index >= first_terminal else None,
            reason=f"gate_{index}",
        )
        for index in range(3)
    ]
    manager = TurnTakingGateManager([
        LocalGate(str(index), decision, calls) for index, decision in enumerate(decisions)
    ])

    result = await manager.should_take_turn(
        "question", "spoken prefix", session_id="session",
        assistant_full_text="spoken prefix plus remainder",
    )

    assert result is decisions[first_terminal]
    assert calls == [
        (str(index), "question", "spoken prefix", "session")
        for index in range(first_terminal + 1)
    ]
    for index, child in enumerate(manager.gates):
        assert child.full_texts == (["spoken prefix plus remainder"] if index <= first_terminal else [])


@pytest.mark.asyncio
async def test_all_children_continuing_allow_only_at_managed_boundary(clock):
    calls = []
    manager = TurnTakingGateManager([
        LocalGate("first", TurnTakingDecision(None, None, "first_continued"), calls),
        LocalGate("second", TurnTakingDecision(None, None, "second_continued"), calls),
    ])

    assert await manager.should_take_turn("yes", "prefix", session_id="session") == TurnTakingDecision(
        None, None, "all_gates_continued"
    )
    assert [call[0] for call in calls] == ["first", "second"]
    start_playback(manager)
    clock.now += 5
    try:
        decision = await manager.evaluate("session", "yes")
        assert decision.should_take_turn is True
        assert [call[0] for call in calls] == ["first", "second", "first", "second"]
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
async def test_empty_manager_continues_raw_and_allows_at_managed_boundary():
    for manager in (TurnTakingGateManager(), TurnTakingGateManager([])):
        assert await manager.should_take_turn("question", "prefix") == TurnTakingDecision(
            None, None, "no_gates"
        )
        manager.get_session("session", create=True)
        try:
            assert (await manager.evaluate("session", "question")).should_take_turn is True
        finally:
            await close_sessions(manager, "session")


@pytest.mark.asyncio
async def test_manager_without_explicit_bypass_allows_idle_and_long_input(clock):
    calls = []
    blocked = TurnTakingDecision(False, None, "child_blocked")
    manager = TurnTakingGateManager([LocalGate("child", blocked, calls)])
    state = manager.get_session(
        "session", create=True, default_skip_condition=lambda text, duration: duration >= 1.5,
    )
    try:
        assert await manager.evaluate("session", "idle") == TurnTakingDecision(True, None, "playback_idle")
        state.start_playback("chunk", "abcdefghij", 10)
        clock.now += 5
        assert await manager.evaluate("session", "long", recorded_duration=1.5) == TurnTakingDecision(
            True, None, "turn_take_skipped"
        )
        assert calls == []
        assert await manager.evaluate("session", "short", recorded_duration=0.3) is blocked
        assert calls == [("child", "short", "abcde", "session")]
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
@pytest.mark.parametrize("bypass_enabled", [False, True])
async def test_explicit_bypass_flag_overrides_manager_inference(bypass_enabled):
    calls = []
    blocked = TurnTakingDecision(False, None, "required_check")
    gates = [LocalGate("required", blocked, calls)]
    if bypass_enabled:
        # A direct bypass would normally disable the automatic policy.
        gates.append(TurnTakingBypassGate())
    manager = TurnTakingGateManager(gates, bypass_enabled=bypass_enabled)
    manager.get_session("session", create=True)
    try:
        decision = await manager.evaluate("session", "idle")
        if bypass_enabled:
            assert decision == TurnTakingDecision(True, None, "playback_idle")
            assert calls == []
        else:
            assert decision is blocked
            assert calls == [("required", "idle", "", "session")]
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
async def test_automatic_manager_bypass_uses_its_skip_and_grace_settings(clock):
    calls = []
    blocked = TurnTakingDecision(False, None, "child_blocked")
    manager = TurnTakingGateManager(
        [LocalGate("child", blocked, calls)],
        skip_condition=lambda text, duration: False,
        response_end_grace_seconds=0.3,
    )
    state = manager.get_session("session", create=True, default_skip_condition=lambda text, duration: True)
    state.start_playback("chunk", "abcdefghij", 10, transaction_id="response")
    state.mark_response_final("chunk", transaction_id="response")
    clock.now += 5
    try:
        assert await manager.evaluate("session", "long", recorded_duration=2) is blocked
        clock.now = 109.8
        assert await manager.evaluate("session", "ending") == TurnTakingDecision(
            True, None, "playback_response_end_grace"
        )
        assert calls == [("child", "long", "abcde", "session")]
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
@pytest.mark.parametrize("inner_value", [True, False, None])
async def test_nested_manager_preserves_continue_for_outer_siblings(inner_value):
    calls = []
    inner_decision = TurnTakingDecision(inner_value, None, "inner_decision")
    outer_decision = TurnTakingDecision(False, None, "outer_veto")
    inner = TurnTakingGateManager([LocalGate("inner", inner_decision, calls)])
    outer = TurnTakingGateManager([inner, LocalGate("outer", outer_decision, calls)])

    result = await outer.should_take_turn("question", "prefix", session_id="session")

    if inner_value is None:
        assert result is outer_decision
        assert [call[0] for call in calls] == ["inner", "outer"]
    else:
        assert result is inner_decision
        assert [call[0] for call in calls] == ["inner"]


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("bypass", ["idle", "no_text", "duration", "grace", "empty_estimate", "skip_error"])
async def test_required_gate_before_bypass_blocks_regardless_of_playback_policy(clock, bypass, nested):
    calls = []
    blocked = TurnTakingDecision(False, None, "mandatory_veto")
    mandatory = LocalGate("mandatory", blocked, calls)
    if nested:
        mandatory = TurnTakingGateManager([mandatory])

    def skip_condition(text, duration):
        if bypass == "skip_error":
            raise RuntimeError("condition unavailable")
        return duration >= 1.5

    manager = TurnTakingGateManager([
        mandatory,
        TurnTakingBypassGate(skip_condition=skip_condition, response_end_grace_seconds=0.3),
        LocalGate("later", TurnTakingDecision(True, None, "later_allowed"), calls),
    ])
    state = manager.get_session("session", create=True)
    if bypass != "idle":
        state.start_playback("chunk", "abcdefghij", 10, transaction_id="response")
        if bypass == "grace":
            state.mark_response_final("chunk", transaction_id="response")
        clock.now += 9.8 if bypass == "grace" else 0 if bypass == "empty_estimate" else 5
    try:
        result = await manager.evaluate(
            "session", None if bypass == "no_text" else "question",
            recorded_duration=1.5 if bypass == "duration" else 0.3,
        )
        assert result is blocked
        assert [call[0] for call in calls] == ["mandatory"]
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
async def test_required_gate_continue_reaches_explicit_bypass():
    calls = []
    manager = TurnTakingGateManager([
        LocalGate("mandatory", TurnTakingDecision(None, None, "active"), calls),
        TurnTakingBypassGate(),
        LocalGate("ordinary", TurnTakingDecision(False, None, "ordinary_blocked"), calls),
    ])
    manager.get_session("session", create=True)
    try:
        decision = await manager.evaluate("session", "question")
        assert decision.should_take_turn is True
        assert [call[0] for call in calls] == ["mandatory"]
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
async def test_manager_uses_explicit_bypass_and_ignores_child_automatic_policy(clock):
    calls = []
    child = LocalGate(
        "child", TurnTakingDecision(False, 0.1, "declined"), calls,
        skip_condition=lambda text, duration: True,
        response_end_grace_seconds=10,
    )
    bypass = TurnTakingBypassGate(skip_condition=lambda text, duration: False)
    manager = TurnTakingGateManager([bypass, child])
    start_playback(manager, final=True)
    clock.now = 109.8
    try:
        # The child is called raw; only the explicit bypass gate applies policy.
        result = await manager.evaluate("session", "yes", recorded_duration=20)
        assert result == TurnTakingDecision(False, 0.1, "declined")
        assert calls == [("child", "yes", "abcdefghi", "session")]
        assert child.get_session("session") is None

        bypass.response_end_grace_seconds = 0.3
        assert (await manager.evaluate("session", "yes")).reason == "playback_response_end_grace"
        bypass.skip_condition = lambda text, duration: duration >= 2
        clock.now = 105.0
        assert (await manager.evaluate("session", "long question", recorded_duration=2)).reason == "turn_take_skipped"
        assert len(calls) == 1
    finally:
        await close_sessions(manager, "session")


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_condition", [False, True], ids=["vad_default", "bypass_override"])
async def test_explicit_bypass_uses_vad_default_unless_it_has_own_condition(clock, explicit_condition):
    calls, notifications = [], []
    child = LocalGate(
        "child", TurnTakingDecision(False, None, "declined"), calls,
    )
    manager = TurnTakingGateManager([
        TurnTakingBypassGate(skip_condition=(lambda text, duration: False) if explicit_condition else None),
        child,
    ])
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

    async def delayed_continue(*args, **kwargs):
        entered.set()
        await release.wait()
        return TurnTakingDecision(None, None, "first_continued")

    manager = TurnTakingGateManager([
        LocalGate("first", delayed_continue, calls),
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
    pending = asyncio.create_task(manager.evaluate("session", "question", recorded_duration=0.3))
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
        contexts = [child.contexts[0] for child in manager.gates]
        assert contexts[0]["playback"] is contexts[1]["playback"]
        assert contexts[0]["playback"].assistant_spoken_text == "abcde"
        assert [context["recorded_duration"] for context in contexts] == [0.3, 0.3]
        assert contexts[0]["is_current"] is contexts[1]["is_current"]
        assert callable(contexts[0]["is_current"])
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

    async def delayed_second(user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
        if user_text == "old":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release_cleanup.wait()
                raise
        return TurnTakingDecision(None, None, "second_continued")

    accepted = TurnTakingDecision(True, 0.8, "third_accepted")
    manager = TurnTakingGateManager([
        LocalGate("first", TurnTakingDecision(None, None, "first_continued"), calls),
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
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("cancelled_outcome", ["continue", "uncancel_continue", "error"])
async def test_new_mandatory_veto_cancels_old_input_without_reviving_it(clock, cancelled_outcome, nested):
    calls = []
    entered, cancelled, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    blocked = TurnTakingDecision(False, None, "missing_wakeword")

    async def mandatory_policy(user_text, *args, **kwargs):
        if user_text != "old":
            return blocked
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            if cancelled_outcome == "uncancel_continue":
                asyncio.current_task().uncancel()
            await release_cleanup.wait()
            if cancelled_outcome == "error":
                raise RuntimeError("obsolete-error")
            return TurnTakingDecision(None, None, "obsolete-continue")

    gates = [
        LocalGate("mandatory", mandatory_policy, calls),
        TurnTakingBypassGate(skip_condition=lambda text, duration: duration >= 1.5),
        LocalGate("later", TurnTakingDecision(True, None, "later_allowed"), calls),
    ]
    if nested:
        gates = [
            TurnTakingGateManager(gates),
            LocalGate("outer_later", TurnTakingDecision(True, None, "outer_allowed"), calls),
        ]
    # The outer manager sees no direct bypass when the policy is nested.
    manager = TurnTakingGateManager(gates, bypass_enabled=False if nested else None)
    start_playback(manager)
    clock.now += 5
    old = asyncio.create_task(manager.evaluate("session", "old", recorded_duration=0.3))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert await manager.evaluate("session", "new", recorded_duration=2) is blocked
        await asyncio.wait_for(cancelled.wait(), 1)
        assert not old.done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(old, 1)
        assert [name for name, text, _, _ in calls if text == "old"] == ["mandatory"]
        assert [name for name, text, _, _ in calls if text == "new"] == ["mandatory"]
    finally:
        release_cleanup.set()
        await close_sessions(manager, "session")
        await asyncio.gather(old, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_joins_the_chain_without_affecting_another_session(clock):
    calls = []
    entered, cancelled, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed_gate(user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
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
        LocalGate("first", TurnTakingDecision(None, None, "continued"), calls),
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
        LocalGate("first", TurnTakingDecision(None, None, "first_continued"), calls),
        LocalGate("second", TurnTakingDecision(True, 0.8, "second_accepted"), calls),
    ], debug=debug)
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates.manager")
    await manager.should_take_turn("question", "prefix", session_id="session")
    messages = [record.message for record in caplog.records if record.name.endswith(".manager")]
    if debug:
        assert len(messages) == 2
        assert "session=session, gate_index=1, gate=LocalGate" in messages[0]
        assert "reason=first_continued" in messages[0]
        assert "gate_index=2, gate=LocalGate" in messages[1]
        assert "probability=0.8, reason=second_accepted" in messages[1]
    else:
        assert messages == []
