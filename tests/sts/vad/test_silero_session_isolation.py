"""Hermetic regression tests for state isolation with a shared Silero engine.

The fake model follows Silero's ONNX wrapper ownership: ``session`` is the
stateless engine; recurrent state and audio context belong to the wrapper.
No model files, network requests, or speech recognition services are used.
"""

import numpy as np
import pytest
import torch

from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector


class FakeInferenceSession:
    """Deterministic inference that depends on both kinds of audio history."""

    def run(self, _, inputs):
        audio = inputs["input"]
        state = inputs["state"]
        context = float(audio[0, 0])
        amplitude = float(audio[0, -1])
        previous_state = float(state[0, 0, 0])
        probability = 0.6 * amplitude + 0.25 * previous_state + 0.15 * context
        next_state = np.full_like(state, 0.6 * previous_state + 0.4 * amplitude)
        return np.array([[probability]], dtype=np.float32), next_state


class FakeOnnxWrapper:
    def __init__(self):
        self.session = FakeInferenceSession()
        self.sample_rates = [8000, 16000]
        self.reset_states()

    def reset_states(self, batch_size=1):
        self._state = torch.zeros((2, batch_size, 128), dtype=torch.float32)
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0
        self.last_probability = None

    def __call__(self, audio, sample_rate):
        audio = audio.unsqueeze(0) if audio.dim() == 1 else audio
        context_size = 64 if sample_rate == 16000 else 32
        if not self._context.numel():
            self._context = torch.zeros((audio.shape[0], context_size))
        with_context = torch.cat((self._context, audio), dim=1)
        probability, state = self.session.run(
            None,
            {"input": with_context.numpy(), "state": self._state.numpy()},
        )
        self._state = torch.from_numpy(state)
        self._context = with_context[:, -context_size:]
        self._last_sr = sample_rate
        self._last_batch_size = audio.shape[0]
        self.last_probability = float(probability.item())
        return torch.from_numpy(probability)


class FakeVADIterator:
    def __init__(self, model, threshold=0.5, sampling_rate=16000):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.current_sample = 0

    def __call__(self, audio, return_seconds=False):
        self.current_sample += audio.numel()
        probability = self.model(audio, self.sampling_rate).item()
        if probability >= self.threshold and not self.triggered:
            self.triggered = True
            return {"start": self.current_sample - audio.numel()}
        if probability < self.threshold - 0.15 and self.triggered:
            self.triggered = False
            return {"end": self.current_sample}
        return None


class UnusedRecognizer:
    async def recognize(self, session_id, audio):
        raise AssertionError("These VAD tests must not invoke STT")


def pcm(amplitude):
    return np.full(512, round(amplitude * 32768), dtype="<i2").tobytes()


@pytest.fixture(params=[SileroSpeechDetector, SileroStreamSpeechDetector], ids=["batch", "stream"])
def detector_class(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["direct", "iterator"])
def use_vad_iterator(request):
    return request.param


@pytest.fixture
def make_detector(monkeypatch, detector_class, use_vad_iterator):
    hub_calls = []
    detectors = []

    def fake_hub_load(**kwargs):
        hub_calls.append(kwargs)
        return FakeOnnxWrapper(), (None, None, None, FakeVADIterator, None)

    monkeypatch.setattr("aiavatar.sts.vad.silero.torch.hub.load", fake_hub_load)

    def create(**kwargs):
        if detector_class is SileroStreamSpeechDetector:
            kwargs["speech_recognizer"] = UnusedRecognizer()
        detector = detector_class(use_vad_iterator=use_vad_iterator, **kwargs)
        detectors.append(detector)
        return detector

    create.hub_calls = hub_calls
    yield create
    for detector in detectors:
        for session_id in list(detector.recording_sessions):
            detector.delete_session(session_id)


def infer(detector, session, amplitude):
    decision = detector._detect_speech_silero(pcm(amplitude), session)
    return decision, session.vad_iterator.model.last_probability


def test_default_model_is_loaded_as_onnx(make_detector):
    make_detector()
    assert len(make_detector.hub_calls) == 1
    assert make_detector.hub_calls[0]["onnx"] is True


def test_sessions_share_engine_but_own_state_and_context(make_detector):
    detector = make_detector()
    first = detector.get_session("first")
    second = detector.get_session("second")
    first_model = first.vad_iterator.model
    second_model = second.vad_iterator.model
    template = detector.model_pool[0]

    assert first_model is not second_model
    assert first_model is not template
    assert second_model is not template
    assert first_model.session is second_model.session is template.session
    assert first_model._state.data_ptr() != second_model._state.data_ptr()
    assert first_model._state.data_ptr() != template._state.data_ptr()

    infer(detector, first, 0.2)
    infer(detector, second, 0.9)

    assert not torch.equal(first_model._state, second_model._state)
    assert first_model._context.data_ptr() != second_model._context.data_ptr()
    assert first_model._context.numel() == second_model._context.numel() == 64
    assert template._context.numel() == 0
    assert torch.count_nonzero(template._state) == 0


def test_interleaved_sessions_match_isolated_probabilities_and_decisions(make_detector):
    audio = {
        "first": [0.45, 0.45, 0.1, 0.7, 0.0, 0.45, 0.0, 0.0],
        "second": [0.95, 0.9, 0.95, 0.0, 0.95, 0.9, 0.0, 0.0],
    }
    expected = {}
    for session_id, amplitudes in audio.items():
        isolated = make_detector()
        session = isolated.get_session(session_id)
        expected[session_id] = [infer(isolated, session, amplitude) for amplitude in amplitudes]

    shared = make_detector()
    sessions = {session_id: shared.get_session(session_id) for session_id in audio}
    actual = {session_id: [] for session_id in audio}
    for offset in range(len(audio["first"])):
        # Reverse the order on alternate frames, preserving each session's order.
        session_ids = ("first", "second") if offset % 2 == 0 else ("second", "first")
        for session_id in session_ids:
            actual[session_id].append(infer(shared, sessions[session_id], audio[session_id][offset]))

    for session_id in audio:
        assert [decision for decision, _ in actual[session_id]] == [
            decision for decision, _ in expected[session_id]
        ]
        assert [probability for _, probability in actual[session_id]] == [
            probability for _, probability in expected[session_id]
        ]


@pytest.mark.parametrize("operation", ["create", "reset", "reset_audio", "reset_vad", "delete"])
def test_other_session_lifecycle_preserves_active_history(make_detector, operation):
    isolated = make_detector()
    isolated_session = isolated.get_session("first")
    for amplitude in (0.8, 0.6):
        infer(isolated, isolated_session, amplitude)
    expected = infer(isolated, isolated_session, 0.2)

    shared = make_detector()
    first = shared.get_session("first")
    if operation != "create":
        shared.get_session("second")
    for amplitude in (0.8, 0.6):
        infer(shared, first, amplitude)
    first_model = first.vad_iterator.model
    state_before = first_model._state.clone()
    context_before = first_model._context.clone()

    if operation == "create":
        shared.get_session("second")
    elif operation == "reset":
        shared.reset_session("second")
    elif operation == "reset_audio":
        shared.reset_session_audio_state("second")
    elif operation == "reset_vad":
        shared.reset_vad_state("second")
    else:
        shared.delete_session("second")

    torch.testing.assert_close(first_model._state, state_before, rtol=0, atol=0)
    torch.testing.assert_close(first_model._context, context_before, rtol=0, atol=0)
    assert infer(shared, first, 0.2) == expected


def test_threshold_change_preserves_session_isolation(make_detector):
    detector = make_detector()
    first = detector.get_session("first")
    second = detector.get_session("second")
    engine = detector.model_pool[0].session
    infer(detector, first, 0.3)
    infer(detector, second, 0.9)

    detector.set_speech_probability_threshold(0.7)

    assert first.vad_iterator.threshold == second.vad_iterator.threshold == 0.7
    first_model = first.vad_iterator.model
    second_model = second.vad_iterator.model
    assert first_model is not second_model
    assert first_model is not detector.model_pool[0]
    assert second_model is not detector.model_pool[0]
    assert first_model.session is second_model.session is engine
    assert first_model._state.data_ptr() != second_model._state.data_ptr()
    infer(detector, first, 0.3)
    state_before = first_model._state.clone()
    context_before = first_model._context.clone()
    detector.reset_session("second")
    torch.testing.assert_close(first_model._state, state_before, rtol=0, atol=0)
    torch.testing.assert_close(first_model._context, context_before, rtol=0, atol=0)
