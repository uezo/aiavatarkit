import asyncio
import importlib
import sys
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from aiavatar.sts.stt.base import SpeechRecognizer
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_end_gates import TurnEndDecision, TurnEndGate


class DummyVADIterator:
    def __init__(self, *args, **kwargs):
        pass

    def reset_states(self):
        pass


def fake_init_silero_model(self, model_path=None):
    self.model_pool = [object()]
    self.model_locks = [threading.Lock()]
    self.VADIteratorClass = DummyVADIterator


class AlwaysHoldGate(TurnEndGate):
    def __init__(self):
        self.calls = []

    async def should_end_turn(self, **kwargs):
        self.calls.append(kwargs)
        return TurnEndDecision(should_end=False, confidence=0.1, reason="hold")


class HoldThenEndGate(TurnEndGate):
    def __init__(self):
        self.calls = 0

    async def should_end_turn(self, **kwargs):
        self.calls += 1
        return TurnEndDecision(
            should_end=self.calls > 1,
            confidence=0.9 if self.calls > 1 else 0.1,
            reason="jitter",
        )


class DummySpeechRecognizer(SpeechRecognizer):
    async def transcribe(self, data: bytes) -> str:
        return "こんにちは"


def detect_non_silent(audio_bytes: bytes, session) -> bool:
    return any(audio_bytes)


class FakeFeatureExtractor:
    def __init__(self):
        self.waveforms = []

    def __call__(self, waveform, **kwargs):
        self.waveforms.append((waveform, kwargs))
        return SimpleNamespace(input_features=np.ones((1, 80, 3000), dtype=np.float32))


class FakeOnnxSession:
    def __init__(self, probability: float):
        self.probability = probability
        self.inputs = []

    def run(self, output_names, inputs):
        self.inputs.append(inputs)
        return [np.asarray([[self.probability]], dtype=np.float32)]


class FakeNamoTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {
            "input_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
        }


class FakeNamoSession:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.inputs = []

    def run(self, output_names, inputs):
        self.inputs.append(inputs)
        return [self.logits]


def import_smart_turn_with_fake_dependencies(monkeypatch):
    fake_ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="ORT_SEQUENTIAL"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL"),
        InferenceSession=lambda *args, **kwargs: FakeOnnxSession(probability=0.5),
    )
    fake_transformers = SimpleNamespace(WhisperFeatureExtractor=FakeFeatureExtractor)
    fake_huggingface_hub = SimpleNamespace(hf_hub_download=lambda repo_id, filename: filename)

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    sys.modules.pop("aiavatar.sts.vad.turn_end_gates.smart_turn", None)
    return importlib.import_module("aiavatar.sts.vad.turn_end_gates.smart_turn").SmartTurnEndGate


def import_namo_turn_with_fake_dependencies(monkeypatch):
    tokenizer_calls = []

    fake_ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="ORT_SEQUENTIAL"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL"),
        InferenceSession=lambda *args, **kwargs: FakeNamoSession([0.0, 1.0]),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda repo_id, **kwargs: (
                tokenizer_calls.append((repo_id, kwargs)) or FakeNamoTokenizer()
            )
        )
    )
    fake_huggingface_hub = SimpleNamespace(hf_hub_download=lambda repo_id, filename: filename)

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    sys.modules.pop("aiavatar.sts.vad.turn_end_gates.namo_turn", None)
    NamoTurnEndGate = importlib.import_module("aiavatar.sts.vad.turn_end_gates.namo_turn").NamoTurnEndGate
    return NamoTurnEndGate, tokenizer_calls


@pytest.mark.asyncio
async def test_silero_turn_end_gate_holds_until_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldGate()
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gate=gate,
        turn_end_gate_timeout=0.3,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append((recorded_data, recorded_duration, session_id))

    session_id = "silero_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)

    await asyncio.sleep(0)
    assert detected == []
    assert len(gate.calls) >= 1
    assert detector.get_session(session_id).is_recording is True

    await detector.process_samples(silence_chunk, session_id)
    await asyncio.sleep(0.05)

    assert len(detected) == 1
    assert detected[0][2] == session_id
    assert detector.get_session(session_id).is_recording is False


@pytest.mark.asyncio
async def test_stream_turn_end_gate_holds_until_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldGate()
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=10.0,
        turn_end_gate=gate,
        turn_end_gate_timeout=0.3,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append((recorded_data, text, recorded_duration, session_id))

    session_id = "stream_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)

    await asyncio.sleep(0)
    assert detected == []
    assert len(gate.calls) >= 1
    assert detector.get_session(session_id).is_recording is True

    await detector.process_samples(silence_chunk, session_id)
    await asyncio.sleep(0.05)

    assert len(detected) == 1
    assert detected[0][1] == "こんにちは"
    assert detected[0][3] == session_id
    assert detector.get_session(session_id).is_recording is False


@pytest.mark.asyncio
async def test_stream_turn_end_gate_receives_recognized_text(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldGate()
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=0.1,
        turn_end_gate=gate,
        turn_end_gate_timeout=0.3,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    session_id = "stream_gate_text"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(3):
        await detector.process_samples(silence_chunk, session_id)

    assert gate.calls
    assert gate.calls[0]["text"] == "こんにちは"


@pytest.mark.asyncio
async def test_stream_wait_pending_recognition_ignores_cancelled_recognition_task(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        sample_rate=16000,
        chunk_size=1,
    )
    session = detector.get_session("cancelled_pending_recognition")

    async def wait_forever():
        await asyncio.Event().wait()

    session.pending_recognition_task = asyncio.create_task(wait_forever())
    session.pending_recognition_task.cancel()

    await detector._wait_pending_recognition_task(session, "in test")
    assert session.pending_recognition_task.cancelled()


@pytest.mark.asyncio
async def test_silero_turn_end_gate_latches_first_hold_until_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = HoldThenEndGate()
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gate=gate,
        turn_end_gate_timeout=0.3,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append((recorded_data, recorded_duration, session_id))

    session_id = "silero_latched_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)

    await asyncio.sleep(0)
    assert detected == []
    assert gate.calls == 1
    assert detector.get_session(session_id).is_recording is True

    await detector.process_samples(silence_chunk, session_id)
    await asyncio.sleep(0.05)

    assert len(detected) == 1
    assert gate.calls == 1
    assert detected[0][2] == session_id


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
