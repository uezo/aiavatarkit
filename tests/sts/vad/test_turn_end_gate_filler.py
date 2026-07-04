import pytest

from aiavatar.sts.vad.turn_end_gates import FillerOnlyTurnEndGate
from aiavatar.sts.vad.turn_end_gates.filler import FillerPhrase


@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_ignores_punctuation():
    gate = FillerOnlyTurnEndGate(fillers=["えっと"], timeout=5.0)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text=" えっと。 ",
    )

    assert decision.should_end is False
    assert decision.timeout == 5.0
    assert decision.reason == "filler_only"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_ignores_prolonged_sound_mark():
    gate = FillerOnlyTurnEndGate(fillers=[FillerPhrase("あの", match="suffix")], timeout=5.0)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="あのー。",
    )

    assert decision.should_end is False
    assert decision.timeout == 5.0
    assert decision.reason == "filler_only"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_passes_non_filler_text():
    gate = FillerOnlyTurnEndGate(fillers=["えっと"], timeout=5.0)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="えっと、今日は行きます。",
    )

    assert decision.should_end is True
    assert decision.timeout is None
    assert decision.reason == "not_filler"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_custom_suffix_phrase():
    gate = FillerOnlyTurnEndGate(
        fillers=[FillerPhrase("えっと", match="suffix")],
        timeout=5.0,
    )

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="口座番号ですね、えっと",
    )

    assert decision.should_end is False
    assert decision.timeout == 5.0
    assert decision.reason == "trailing_filler"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_uses_phrase_timeout():
    gate = FillerOnlyTurnEndGate(
        fillers=[FillerPhrase("えっと", match="suffix", timeout=8.0)],
        timeout=5.0,
    )

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="口座番号ですね、えっと",
    )

    assert decision.should_end is False
    assert decision.timeout == 8.0
    assert decision.reason == "trailing_filler"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_prefers_longest_suffix_timeout():
    gate = FillerOnlyTurnEndGate(
        fillers=[
            FillerPhrase("あの", match="suffix", timeout=3.0),
            FillerPhrase("あのね", match="suffix", timeout=7.0),
        ],
        timeout=5.0,
    )

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="口座番号ですね、あのね",
    )

    assert decision.should_end is False
    assert decision.timeout == 7.0
    assert decision.reason == "trailing_filler"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_waits_for_trailing_filler():
    gate = FillerOnlyTurnEndGate(timeout=5.0)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="口座番号ですね、えっと",
    )

    assert decision.should_end is False
    assert decision.timeout == 5.0
    assert decision.reason == "trailing_filler"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_does_not_wait_for_single_char_trailing_filler():
    gate = FillerOnlyTurnEndGate(timeout=5.0)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="今日は行きます、あ",
    )

    assert decision.should_end is True
    assert decision.timeout is None
    assert decision.reason == "not_filler"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_default_does_not_treat_un_as_filler():
    gate = FillerOnlyTurnEndGate(timeout=5.0)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="うん。",
    )

    assert decision.should_end is True
    assert decision.reason == "not_filler"

@pytest.mark.asyncio
async def test_filler_only_turn_end_gate_default_english_fillers():
    gate = FillerOnlyTurnEndGate(timeout=5.0)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=0.5,
        silence_duration=0.5,
        session_id="filler",
        text="Um...",
    )

    assert decision.should_end is False
    assert decision.timeout == 5.0
    assert decision.reason == "filler_only"
