import asyncio
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.turn_end_gates.manager import TurnEndGateManager

from _turn_end_gate_helpers import AlwaysHoldGate, BackgroundPassGate


@pytest.mark.asyncio
async def test_turn_end_gate_manager_does_not_build_audio_while_hold_is_active():
    gate = AlwaysHoldGate(timeout=0.3)
    manager = TurnEndGateManager([gate])
    session = SimpleNamespace(
        session_id="manager_lazy_audio",
        turn_end_gate_hold_active=False,
        turn_end_gate_hold_timeout=None,
        turn_end_gate_hold_reasons=[],
    )
    audio_factory_calls = 0

    def audio_factory():
        nonlocal audio_factory_calls
        audio_factory_calls += 1
        return b"\x01\x00"

    should_end = await manager.should_end_turn(
        session=session,
        audio_factory=audio_factory,
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.3,
        silence_duration_threshold=0.2,
    )

    assert should_end is False
    assert audio_factory_calls == 1

    should_end = await manager.should_end_turn(
        session=session,
        audio_factory=audio_factory,
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.4,
        silence_duration_threshold=0.2,
    )

    assert should_end is False
    assert audio_factory_calls == 1

@pytest.mark.asyncio
async def test_background_only_turn_end_gate_latches_pending_timeout():
    future = asyncio.get_running_loop().create_future()
    gate = BackgroundPassGate(future, timeout=0.6)
    manager = TurnEndGateManager([gate])
    session = SimpleNamespace(
        session_id="manager_background_only",
        turn_end_gate_hold_active=False,
        turn_end_gate_hold_timeout=None,
        turn_end_gate_hold_reasons=[],
    )

    should_end = await manager.should_end_turn(
        session=session,
        audio=b"\x01\x00",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.3,
        silence_duration_threshold=0.2,
    )
    await asyncio.sleep(0)

    assert should_end is False
    assert gate.calls
    assert session.turn_end_gate_hold_active is True
    assert session.turn_end_gate_hold_timeout == 0.6
    assert session.turn_end_gate_hold_reasons == ["background_pass_pending"]

    future.set_result(None)
    await asyncio.sleep(0)

    should_end = await manager.should_end_turn(
        session=session,
        audio=b"\x01\x00",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.4,
        silence_duration_threshold=0.2,
    )

    assert should_end is True
    assert session.turn_end_gate_hold_active is False
