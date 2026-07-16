import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector

from _turn_end_gate_helpers import (
    AlwaysHoldForeverGate,
    AlwaysHoldGate,
    DummySpeechRecognizer,
    detect_non_silent,
    fake_init_silero_model,
)


SPEECH_CHUNK = b"\x01\x00" * 1600
SILENCE_CHUNK = b"\x00\x00" * 1600


@pytest.mark.asyncio
async def test_silero_records_speech_end_threshold_and_gate_hold(monkeypatch):
    monkeypatch.setattr(
        SileroSpeechDetector,
        "_init_silero_model",
        fake_init_silero_model,
    )
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[AlwaysHoldGate(timeout=0.3)],
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(data, text, metadata, duration, session_id):
        detected.append(metadata)

    await detector.process_samples(SPEECH_CHUNK, "silero-performance")
    await detector.process_samples(SPEECH_CHUNK, "silero-performance")
    for _ in range(6):
        await detector.process_samples(SILENCE_CHUNK, "silero-performance")
    await asyncio.sleep(0)

    timing = detected[0]["vad_performance"]
    assert isinstance(timing["speech_end_at"], datetime)
    assert timing["speech_end_at"].tzinfo is not None
    assert timing["silence_threshold_time"] == pytest.approx(0.2)
    assert timing["stt_after_threshold_time"] is None
    assert timing["turn_end_gate_time"] is not None
    assert timing["turn_end_gate_time"] >= 0
    assert timing["turn_end_gate_held"] is True


@pytest.mark.asyncio
async def test_silero_resets_candidate_timing_when_speech_resumes(monkeypatch):
    monkeypatch.setattr(
        SileroSpeechDetector,
        "_init_silero_model",
        fake_init_silero_model,
    )
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[AlwaysHoldForeverGate()],
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    session_id = "silero-performance-reset"
    await detector.process_samples(SPEECH_CHUNK, session_id)
    await detector.process_samples(SPEECH_CHUNK, session_id)
    await detector.process_samples(SILENCE_CHUNK, session_id)
    await detector.process_samples(SILENCE_CHUNK, session_id)

    session = detector.get_session(session_id)
    assert session.silence_threshold_reached_at is not None
    assert session.turn_end_gate_held is True

    await detector.process_samples(SPEECH_CHUNK, session_id)

    assert session.silence_threshold_reached_at is None
    assert session.speech_end_at is None
    assert session.turn_end_gate_time is None
    assert session.turn_end_gate_held is None


@pytest.mark.asyncio
async def test_silero_finalizes_gate_timing_when_max_duration_ends_hold(monkeypatch):
    monkeypatch.setattr(
        SileroSpeechDetector,
        "_init_silero_model",
        fake_init_silero_model,
    )
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[AlwaysHoldForeverGate()],
        max_duration=0.5,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)
    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(data, text, metadata, duration, session_id):
        detected.append(metadata)

    session_id = "silero-performance-max-duration"
    await detector.process_samples(SPEECH_CHUNK, session_id)
    await detector.process_samples(SPEECH_CHUNK, session_id)
    await detector.process_samples(SILENCE_CHUNK, session_id)
    await detector.process_samples(SILENCE_CHUNK, session_id)
    await detector.process_samples(SILENCE_CHUNK, session_id)
    await asyncio.sleep(0)

    timing = detected[0]["vad_performance"]
    assert timing["turn_end_gate_held"] is True
    assert timing["turn_end_gate_time"] is not None
    assert timing["turn_end_gate_time"] >= 0


@pytest.mark.asyncio
async def test_silero_stream_records_only_stt_wait_after_threshold(monkeypatch):
    monkeypatch.setattr(
        SileroSpeechDetector,
        "_init_silero_model",
        fake_init_silero_model,
    )
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=10.0,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    async def delayed_recognition(session, data):
        await asyncio.sleep(0.02)
        return SimpleNamespace(text="こんにちは")

    monkeypatch.setattr(detector, "_recognize_audio", delayed_recognition)
    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(data, text, metadata, duration, session_id):
        detected.append(metadata)

    await detector.process_samples(SPEECH_CHUNK, "stream-performance")
    await detector.process_samples(SPEECH_CHUNK, "stream-performance")
    await detector.process_samples(SILENCE_CHUNK, "stream-performance")
    await detector.process_samples(SILENCE_CHUNK, "stream-performance")
    await asyncio.sleep(0)

    timing = detected[0]["vad_performance"]
    assert timing["silence_threshold_time"] == pytest.approx(0.2)
    assert timing["stt_after_threshold_time"] >= 0.015
    assert timing["turn_end_gate_time"] is None
    assert timing["turn_end_gate_held"] is None


@pytest.mark.asyncio
async def test_silero_stream_does_not_emit_performance_without_recognized_text(monkeypatch):
    monkeypatch.setattr(
        SileroSpeechDetector,
        "_init_silero_model",
        fake_init_silero_model,
    )
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=10.0,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    async def empty_recognition(session, data):
        return SimpleNamespace(text="")

    monkeypatch.setattr(detector, "_recognize_audio", empty_recognition)
    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(data, text, metadata, duration, session_id):
        detected.append(metadata)

    await detector.process_samples(SPEECH_CHUNK, "stream-empty-recognition")
    await detector.process_samples(SPEECH_CHUNK, "stream-empty-recognition")
    await detector.process_samples(SILENCE_CHUNK, "stream-empty-recognition")
    await detector.process_samples(SILENCE_CHUNK, "stream-empty-recognition")
    await asyncio.sleep(0)

    assert detected == []
