import sys

import numpy as np
import pytest

from _turn_end_gate_helpers import (
    FakeFeatureExtractor,
    FakeOnnxSession,
    import_smart_turn_with_fake_dependencies,
)


@pytest.mark.asyncio
async def test_smart_turn_end_gate_converts_audio_and_returns_decision(monkeypatch):
    SmartTurnEndGate = import_smart_turn_with_fake_dependencies(monkeypatch)
    feature_extractor = FakeFeatureExtractor()
    onnx_session = FakeOnnxSession(probability=0.75)
    gate = SmartTurnEndGate(
        threshold=0.5,
        feature_extractor=feature_extractor,
        session=onnx_session,
    )

    audio = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
    decision = await gate.should_end_turn(
        audio=audio,
        sample_rate=16000,
        channels=1,
        recorded_duration=0.1,
        silence_duration=0.5,
        session_id="smart_turn",
    )

    assert decision.should_end is True
    assert decision.confidence == pytest.approx(0.75)
    assert decision.reason == "smart_turn_complete"
    assert len(feature_extractor.waveforms) == 1
    waveform, kwargs = feature_extractor.waveforms[0]
    assert waveform.dtype == np.float32
    assert len(waveform) == 8 * 16000
    assert waveform[-1] == pytest.approx(1000 / 32768.0)
    assert kwargs["sampling_rate"] == 16000
    assert "input_features" in onnx_session.inputs[0]

@pytest.mark.asyncio
async def test_smart_turn_end_gate_marks_incomplete_below_threshold(monkeypatch):
    SmartTurnEndGate = import_smart_turn_with_fake_dependencies(monkeypatch)
    gate = SmartTurnEndGate(
        threshold=0.5,
        feature_extractor=FakeFeatureExtractor(),
        session=FakeOnnxSession(probability=0.25),
    )

    decision = await gate.should_end_turn(
        audio=b"\x00\x00" * 1600,
        sample_rate=16000,
        channels=1,
        recorded_duration=0.1,
        silence_duration=0.5,
        session_id="smart_turn",
    )

    assert decision.should_end is False
    assert decision.confidence == pytest.approx(0.25)
    assert decision.reason == "smart_turn_incomplete"

def test_smart_turn_end_gate_model_path_skips_download(monkeypatch):
    SmartTurnEndGate = import_smart_turn_with_fake_dependencies(monkeypatch)
    module = sys.modules["aiavatar.sts.vad.turn_end_gates.smart_turn"]

    def fail_download(repo_id, filename):
        raise AssertionError("hf_hub_download should not be called when model_path is set")

    monkeypatch.setattr(module, "hf_hub_download", fail_download)

    gate = SmartTurnEndGate(
        model_path="/models/smart-turn.onnx",
        feature_extractor=FakeFeatureExtractor(),
    )

    assert gate.session is not None
