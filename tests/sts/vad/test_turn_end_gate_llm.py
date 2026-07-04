import pytest

from aiavatar.sts.vad.turn_end_gates import TurnEndDecision, TurnEndGateContext
from aiavatar.sts.vad.turn_end_gates.llm import LLMTurnEndGate

from _turn_end_gate_helpers import create_openai_client_for_test


@pytest.mark.asyncio
async def test_llm_turn_end_gate_runs_only_when_dependency_waits():
    gate = LLMTurnEndGate(
        openai_client=create_openai_client_for_test(),
        depends_on="filler",
        timeout=10.0,
    )

    skipped = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.5,
        session_id="llm",
        text="口座番号ですね、えっと",
    )

    assert skipped.should_end is True
    assert skipped.reason == "llm_skipped"

@pytest.mark.asyncio
async def test_llm_turn_end_gate_waits_with_long_timeout_when_dependency_waits():
    context = TurnEndGateContext()
    context.add_decision("filler", TurnEndDecision(should_end=False, reason="trailing_filler", timeout=5.0))
    gate = LLMTurnEndGate(
        openai_client=create_openai_client_for_test(),
        depends_on="filler",
        timeout=10.0,
        request_timeout=10.0,
        temperature=0,
        system_prompt="Answer exactly WAIT and nothing else.",
    )

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.5,
        session_id="llm",
        text="口座番号ですね、えっと",
        context=context,
    )

    assert decision.should_end is False
    assert decision.reason == "llm_incomplete"
    assert decision.timeout == 10.0

@pytest.mark.asyncio
async def test_llm_turn_end_gate_marks_complete_with_real_openai_client():
    context = TurnEndGateContext()
    context.add_decision("filler", TurnEndDecision(should_end=False, reason="trailing_filler", timeout=5.0))
    gate = LLMTurnEndGate(
        openai_client=create_openai_client_for_test(),
        depends_on="filler",
        request_timeout=10.0,
        system_prompt="Answer exactly END and nothing else.",
    )

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.5,
        session_id="llm",
        text="口座番号ですね、えっと",
        context=context,
    )

    assert decision.should_end is True
    assert decision.reason == "llm_complete"
    assert decision.timeout is None
