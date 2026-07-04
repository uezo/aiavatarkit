import pytest

from _turn_end_gate_helpers import (
    FakeNamoSession,
    FakeNamoTokenizer,
    import_namo_turn_with_fake_dependencies,
)


@pytest.mark.asyncio
async def test_namo_turn_end_gate_marks_complete_from_label_one(monkeypatch):
    NamoTurnEndGate, _ = import_namo_turn_with_fake_dependencies(monkeypatch)
    tokenizer = FakeNamoTokenizer()
    session = FakeNamoSession([0.1, 2.0])
    gate = NamoTurnEndGate(tokenizer=tokenizer, session=session, threshold=0.5)

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.5,
        session_id="namo_turn",
        text="こんにちは。",
    )

    assert decision.should_end is True
    assert decision.confidence > 0.5
    assert decision.reason == "namo_end_of_turn"
    assert tokenizer.calls[0][0] == "こんにちは。"
    assert "input_ids" in session.inputs[0]

@pytest.mark.asyncio
async def test_namo_turn_end_gate_marks_incomplete_from_label_zero(monkeypatch):
    NamoTurnEndGate, _ = import_namo_turn_with_fake_dependencies(monkeypatch)
    gate = NamoTurnEndGate(
        tokenizer=FakeNamoTokenizer(),
        session=FakeNamoSession([2.0, 0.1]),
        threshold=0.5,
    )

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.5,
        session_id="namo_turn",
        text="えっと、この間さ",
    )

    assert decision.should_end is False
    assert decision.confidence > 0.5
    assert decision.reason == "namo_not_end_of_turn"

@pytest.mark.asyncio
async def test_namo_turn_end_gate_no_text_defaults_to_complete(monkeypatch):
    NamoTurnEndGate, _ = import_namo_turn_with_fake_dependencies(monkeypatch)
    gate = NamoTurnEndGate(
        tokenizer=FakeNamoTokenizer(),
        session=FakeNamoSession([2.0, 0.1]),
    )

    decision = await gate.should_end_turn(
        audio=b"",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.0,
        silence_duration=0.5,
        session_id="namo_turn",
        text=None,
    )

    assert decision.should_end is True
    assert decision.confidence is None
    assert decision.reason == "namo_no_text"

def test_namo_turn_end_gate_tokenizer_truncates_from_left(monkeypatch):
    NamoTurnEndGate, tokenizer_calls = import_namo_turn_with_fake_dependencies(monkeypatch)
    NamoTurnEndGate(session=FakeNamoSession([0.0, 1.0]))

    assert tokenizer_calls
    assert tokenizer_calls[0][1]["truncation_side"] == "left"

def test_namo_turn_end_gate_tokenizer_path_loads_local_tokenizer(monkeypatch):
    NamoTurnEndGate, tokenizer_calls = import_namo_turn_with_fake_dependencies(monkeypatch)
    monkeypatch.setattr("aiavatar.sts.vad.turn_end_gates.namo_turn.os.path.isdir", lambda path: True)

    gate = NamoTurnEndGate(
        model_path="/models/namo/model_quant.onnx",
        tokenizer_path="/models/namo/tokenizer",
    )

    assert gate.session is not None
    assert tokenizer_calls
    assert tokenizer_calls[0][0] == "/models/namo/tokenizer"
    assert tokenizer_calls[0][1]["truncation_side"] == "left"

def test_namo_turn_end_gate_missing_tokenizer_path_raises(monkeypatch):
    NamoTurnEndGate, _ = import_namo_turn_with_fake_dependencies(monkeypatch)
    monkeypatch.setattr("aiavatar.sts.vad.turn_end_gates.namo_turn.os.path.isdir", lambda path: False)

    with pytest.raises(FileNotFoundError):
        NamoTurnEndGate(
            model_path="/models/namo/model_quant.onnx",
            tokenizer_path="/missing/namo/tokenizer",
        )
