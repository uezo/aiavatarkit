"""Stream length regressions without audio models, providers, or credentials."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_end_gates import TurnEndDecision


VOICE = b"\x01\x00" * 1600  # 0.1 seconds at 16 kHz, mono PCM16
SILENCE = b"\x00\x00" * 1600


class Recognizer:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    async def recognize(self, session_id, data):
        self.calls += 1
        return SimpleNamespace(text=self.text)


@pytest_asyncio.fixture
async def make_detector(monkeypatch):
    monkeypatch.setattr(SileroStreamSpeechDetector, "_init_silero_model", lambda *args: None)
    monkeypatch.setattr(SileroStreamSpeechDetector, "_create_vad_iterator", lambda *args: None)
    monkeypatch.setattr(
        SileroStreamSpeechDetector, "_detect_speech_silero",
        lambda self, data, session: any(data),
    )
    detectors = []

    def make(text="はい", recognizer=None, **options):
        recognizer = recognizer or Recognizer(text)
        config = dict(segment_silence_threshold=0.1, silence_duration_threshold=0.4)
        config.update(options)
        detector = SileroStreamSpeechDetector(speech_recognizer=recognizer, **config)
        detectors.append(detector)
        observed = SimpleNamespace(
            detector=detector, recognizer=recognizer, partial=[], started=[], final=[],
        )

        @detector.on_speech_detecting
        async def on_partial(text, session):
            observed.partial.append(text)

        @detector.on_recording_started
        async def on_started(session_id):
            observed.started.append(session_id)

        @detector.on_speech_detected
        async def on_final(audio, text, metadata, duration, session_id):
            observed.final.append((text, duration, session_id))

        return observed

    yield make

    for detector in detectors:
        for session_id, session in list(detector.recording_sessions.items()):
            task = session.pending_recognition_task
            detector.reset_session_audio_state(session_id)
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await detector.finalize_session(session_id)


async def utterance(detector, voice_frames=1):
    for _ in range(voice_frames):
        await detector.process_samples(VOICE, "s")
    await detector.process_samples(SILENCE, "s")
    if task := detector.get_session("s").pending_recognition_task:
        await task
    for _ in range(3):
        await detector.process_samples(SILENCE, "s")
    await asyncio.sleep(0)  # Drain recording-started and speech-detected callbacks.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "voice_frames,text,options,accepted,started",
    [
        (1, "はい", {}, True, True),
        (1, "あ", {}, False, False),
        (3, "あ", {}, True, False),
        (1, "あ", {"min_text_length": 1}, True, False),
        (1, "はい", {"min_text_length": 3, "on_recording_started_min_text_length": 3}, False, False),
        (1, "はいよ", {"min_text_length": 3, "on_recording_started_min_text_length": 3}, True, True),
        (1, "", {}, False, False),
        (3, "", {}, False, False),
    ],
)
async def test_accepts_duration_or_text_length(make_detector, voice_frames, text, options, accepted, started):
    observed = make_detector(text, **options)

    await utterance(observed.detector, voice_frames)

    assert observed.partial == ([text] if text else [])
    assert observed.started == (["s"] if started else [])
    assert observed.final == ([(text, pytest.approx(voice_frames * 0.1), "s")] if accepted else [])
    assert not observed.detector.get_session("s").is_recording


@pytest.mark.asyncio
@pytest.mark.parametrize("text,accepted", [("はい", True), ("あ", False)])
async def test_waits_for_pending_partial_before_length_decision(make_detector, text, accepted):
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedRecognizer(Recognizer):
        async def recognize(self, session_id, data):
            entered.set()
            await release.wait()
            return await super().recognize(session_id, data)

    observed = make_detector(recognizer=DelayedRecognizer(text))
    detector = observed.detector
    await detector.process_samples(VOICE, "s")
    await detector.process_samples(SILENCE, "s")
    await asyncio.wait_for(entered.wait(), timeout=1)

    async def finish():
        for _ in range(3):
            await detector.process_samples(SILENCE, "s")

    finishing = asyncio.create_task(finish())
    try:
        await asyncio.sleep(0)
        assert not finishing.done()
        release.set()
        await asyncio.wait_for(finishing, timeout=1)
        await asyncio.sleep(0)
        assert observed.partial == [text]
        assert observed.final == ([(text, pytest.approx(0.1), "s")] if accepted else [])
        assert not detector.get_session("s").is_recording
    finally:
        release.set()
        finishing.cancel()
        await asyncio.gather(finishing, return_exceptions=True)


@pytest.mark.asyncio
async def test_short_audio_without_a_partial_is_rejected_without_recognition(make_detector):
    observed = make_detector(segment_silence_threshold=1.0)

    await utterance(observed.detector)

    assert observed.partial == []
    assert observed.recognizer.calls == 0
    assert observed.final == []
    assert not observed.detector.get_session("s").is_recording


@pytest.mark.asyncio
async def test_min_text_length_config_can_be_updated(make_detector):
    observed = make_detector()
    assert observed.detector.get_config()["min_text_length"] == 2
    assert observed.detector.set_config({"min_text_length": 3}) == {"min_text_length": 3}
    assert observed.detector.get_config()["min_text_length"] == 3

    await utterance(observed.detector)

    assert observed.final == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_short_text_still_respects_gate_and_validation(make_detector, invalid):
    gate_texts, validated = [], []

    class HoldGate:
        async def should_end_turn(self, **kwargs):
            gate_texts.append(kwargs["text"])
            return TurnEndDecision(should_end=False, timeout=0.15)

    observed = make_detector(turn_end_gates=[HoldGate()])
    detector = observed.detector

    @detector.validate_recognized_text
    def validate(text):
        validated.append(text)
        return "invalid text" if invalid else None

    await utterance(detector)

    assert gate_texts == ["はい"]
    assert observed.final == []
    assert validated == []
    assert detector.get_session("s").is_recording

    for _ in range(2):
        await detector.process_samples(SILENCE, "s")
    await asyncio.sleep(0)

    assert observed.recognizer.calls == 1
    assert validated == ["はい"]
    assert observed.final == ([] if invalid else [("はい", pytest.approx(0.1), "s")])
    assert not detector.get_session("s").is_recording
