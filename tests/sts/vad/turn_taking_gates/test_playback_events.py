"""Playback event routing against real session state, with no external services."""

import logging
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.turn_taking_gates import TurnTakingDecision, TurnTakingGate, TurnTakingGateManager
import aiavatar.sts.vad.turn_taking_gates.session as session_module


class LocalGate(TurnTakingGate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.full_texts = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None):
        self.calls.append((user_text, assistant_spoken_text, session_id))
        self.full_texts.append(assistant_full_text)
        return TurnTakingDecision(False, None, "local_decline")


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


@pytest.fixture
def gate():
    gate = LocalGate()
    gate.get_session("session", create=True)
    yield gate
    gate.close_session("session")


def start_event(gate, **overrides):
    metadata = {
        "event": "start", "playback_id": "chunk", "text": "abcdefghij",
        "duration_seconds": 10.0, "transaction_id": "response",
    }
    metadata.update(overrides)
    return gate.handle_playback_event("session", **metadata)


def test_start_final_end_control_actual_playback_progress(gate, clock):
    state = gate.get_session("session")
    assert start_event(gate) is True
    clock.now += 5
    estimate = state.estimate_playback()
    assert estimate.assistant_spoken_text == "abcde"
    assert estimate.assistant_full_text == "abcdefghij"
    assert estimate.transaction_id == "response"
    assert estimate.is_playing
    assert gate.handle_playback_event(
        "session", event="final", playback_id="chunk", transaction_id="response",
    ) is True
    assert state.estimate_playback().is_final_chunk
    clock.now += 5
    assert gate.handle_playback_event("session", event="end", playback_id="chunk", completed=True) is True
    finished = state.estimate_playback()
    assert not finished.is_playing
    assert finished.assistant_spoken_text == "abcdefghij"
    assert finished.assistant_full_text == "abcdefghij"
    assert finished.reason == "completed"
    assert finished.is_final_chunk


def test_started_chunks_accumulate_only_within_same_transaction(gate, clock):
    state = gate.get_session("session")
    assert start_event(gate, text="abc", duration_seconds=3) is True
    clock.now += 3
    assert gate.handle_playback_event("session", event="end", playback_id="chunk", completed=True) is True
    clock.now += 1
    assert start_event(gate, playback_id="next", text="def", duration_seconds=3) is True
    at_start = state.estimate_playback()
    assert at_start.assistant_spoken_text == "abc"
    assert at_start.continuation_hint == "de…"
    assert at_start.assistant_full_text == "abcdef"
    clock.now += 1
    assert state.estimate_playback().assistant_spoken_text == "abcd"
    assert state.estimate_playback().assistant_full_text == "abcdef"
    assert start_event(gate, playback_id="new", text="xyz", duration_seconds=3, transaction_id="new_response") is True
    clock.now += 1
    estimate = state.estimate_playback()
    assert estimate.assistant_spoken_text == "x"
    assert estimate.assistant_full_text == "xyz"
    assert estimate.transaction_id == "new_response"


def test_keyword_session_and_unknown_metadata_are_supported(gate, clock):
    metadata = {
        "event": "start", "playback_id": "chunk", "text": "abcdefghij",
        "duration_seconds": 10, "transaction_id": "response",
        "client_trace_id": "trace", "future_field": {"nested": True},
    }
    assert gate.handle_playback_event(session_id="session", **metadata) is True
    clock.now += 3
    assert gate.get_session("session").estimate_playback().assistant_spoken_text == "abc"


@pytest.mark.parametrize("invalid", [{}, {"playback_id": "invalid", "text": "text", "duration_seconds": 0}])
def test_invalid_start_preserves_existing_context_clearing_behavior(gate, clock, invalid):
    assert start_event(gate) is True
    clock.now += 2
    assert gate.get_session("session").estimate_playback().assistant_spoken_text == "ab"
    assert gate.handle_playback_event("session", event="start", **invalid) is False
    estimate = gate.get_session("session").estimate_playback()
    assert estimate.playback_id is None
    assert estimate.assistant_spoken_text == ""
    assert estimate.assistant_full_text == ""
    assert estimate.reason == "idle"


def test_stale_final_end_and_missing_events_do_not_change_playback(gate, clock):
    assert start_event(gate) is True
    clock.now += 2
    state = gate.get_session("session")
    before = state.estimate_playback()
    for metadata in (
        {"event": "end", "playback_id": "stale", "completed": True},
        {"event": "final", "playback_id": "chunk", "transaction_id": "stale"},
        {"event": "final", "playback_id": "stale", "transaction_id": "response"},
        {"event": "final"}, {"event": "end"}, {"event": "unknown"}, {},
    ):
        assert gate.handle_playback_event("session", **metadata) is False
        assert state.estimate_playback() == before


@pytest.mark.parametrize("state_kind", ["missing", "closed", "removed"])
def test_events_never_create_or_reopen_missing_or_closed_sessions(clock, state_kind):
    gate = LocalGate()
    state = None
    if state_kind != "missing":
        state = gate.get_session("session", create=True)
        if state_kind == "closed":
            state.close()
        else:
            gate.close_session("session")
    assert start_event(gate) is False
    assert gate.get_session("session") is (state if state_kind == "closed" else None)
    if state is not None:
        assert state.is_closed
        assert state.estimate_playback().playback_id is None
        assert state.estimate_playback().assistant_full_text == ""
    gate.close_session("session")


def test_completed_requires_literal_true(gate, clock):
    for completed in (True, False, 1, "true", None):
        assert start_event(gate, transaction_id=None) is True
        clock.now += 2
        assert gate.handle_playback_event("session", event="end", playback_id="chunk", completed=completed) is True
        estimate = gate.get_session("session").estimate_playback()
        assert estimate.reason == ("completed" if completed is True else "stopped")
        assert estimate.assistant_spoken_text == ("abcdefghij" if completed is True else "ab")
        assert estimate.assistant_full_text == "abcdefghij"


def test_debug_reports_ignored_events_only_when_enabled(clock, caplog):
    gate = LocalGate(debug=True)
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad.turn_taking_gates.base")
    assert start_event(gate) is False
    gate.get_session("session", create=True)
    assert gate.handle_playback_event("session", event="unknown") is False
    gate.get_session("session").close()
    assert start_event(gate) is False
    messages = [record for record in caplog.records if record.name == "aiavatar.sts.vad.turn_taking_gates.base"]
    assert len(messages) == 3
    assert all(record.levelno == logging.INFO for record in messages)
    assert all("session=session" in record.message for record in messages)
    caplog.clear()
    gate.debug = False
    assert start_event(gate) is False
    assert caplog.records == []
    gate.close_session("session")


@pytest.mark.asyncio
async def test_manager_routes_controls_to_its_own_state_and_passes_prefix_to_children(clock):
    child = LocalGate()
    manager = TurnTakingGateManager([child])
    state = manager.get_session("session", create=True)
    try:
        assert start_event(manager) is True
        clock.now += 5
        assert manager.handle_playback_event(
            "session", event="final", playback_id="chunk", transaction_id="response",
        ) is True
        assert state.estimate_playback().is_final_chunk
        assert (await manager.evaluate("session", "question")).should_take_turn is False
        assert child.calls == [("question", "abcde", "session")]
        assert child.full_texts == ["abcdefghij"]
        assert child.get_session("session") is None
    finally:
        manager.close_session("session")
        await state.aclose()
