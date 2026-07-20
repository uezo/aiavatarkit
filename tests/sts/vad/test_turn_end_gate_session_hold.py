import pytest

from aiavatar.sts.vad.turn_end_gates.session_hold import SessionHoldTurnEndGate


async def evaluate(gate: SessionHoldTurnEndGate, session_id: str):
    return await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.5,
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_session_hold_is_consumed_by_next_candidate():
    gate = SessionHoldTurnEndGate()
    gate.hold("session", timeout=0.8, reason="observer_hold")

    held = await evaluate(gate, "session")
    passed = await evaluate(gate, "session")

    assert held.should_end is False
    assert held.timeout == 0.8
    assert held.reason == "observer_hold"
    assert passed.should_end is True
    assert gate.get_hold("session") is None


def test_session_hold_replace_overwrites_pending_hold():
    gate = SessionHoldTurnEndGate()
    gate.hold("session", timeout=0.8, reason="first")
    gate.hold("session", timeout=0.3, reason="second")

    hold = gate.get_hold("session")

    assert hold is not None
    assert hold.timeout == 0.3
    assert hold.reason == "second"


def test_session_hold_without_replace_keeps_longest_timeout():
    gate = SessionHoldTurnEndGate()
    gate.hold("session", timeout=0.8, reason="first")
    gate.hold("session", timeout=0.3, reason="second", replace=False)

    hold = gate.get_hold("session")

    assert hold is not None
    assert hold.timeout == 0.8
    assert hold.reason == "second"


@pytest.mark.asyncio
async def test_session_hold_release_clears_pending_hold():
    gate = SessionHoldTurnEndGate()
    gate.hold("session", timeout=0.8)

    gate.release("session")
    decision = await evaluate(gate, "session")

    assert decision.should_end is True


@pytest.mark.asyncio
async def test_session_hold_passes_when_pending_hold_has_expired(monkeypatch):
    now = 100.0
    monkeypatch.setattr(
        "aiavatar.sts.vad.turn_end_gates.session_hold.time.monotonic",
        lambda: now,
    )
    gate = SessionHoldTurnEndGate(default_expires_in=5.0)
    gate.hold("session", timeout=0.8)

    now = 105.0
    decision = await evaluate(gate, "session")

    assert decision.should_end is True
    assert decision.reason == "session_hold_expired"
    assert gate.get_hold("session") is None


def test_session_hold_purges_other_expired_sessions_when_registering(monkeypatch):
    now = 100.0
    monkeypatch.setattr(
        "aiavatar.sts.vad.turn_end_gates.session_hold.time.monotonic",
        lambda: now,
    )
    gate = SessionHoldTurnEndGate(default_expires_in=5.0)
    gate.hold("expired", timeout=0.8)

    now = 105.0
    gate.hold("active", timeout=0.8)

    assert "expired" not in gate._holds
    assert gate.get_hold("active") is not None


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_session_hold_rejects_invalid_timeout(timeout):
    gate = SessionHoldTurnEndGate()

    with pytest.raises(ValueError):
        gate.hold("session", timeout=timeout)
