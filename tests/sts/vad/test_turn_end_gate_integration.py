import asyncio

import pytest

from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector

from _turn_end_gate_helpers import (
    AlwaysHoldForeverGate,
    AlwaysHoldGate,
    AlwaysPassGate,
    BackgroundHoldGate,
    DummySpeechRecognizer,
    HoldThenEndGate,
    detect_non_silent,
    fake_init_silero_model,
)


@pytest.mark.asyncio
async def test_silero_turn_end_gate_holds_until_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldGate()
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[gate],
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
    assert session_id not in detector.turn_end_gate_manager._states

@pytest.mark.asyncio
async def test_stream_turn_end_gate_holds_until_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldGate()
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=10.0,
        turn_end_gates=[gate],
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
    assert session_id not in detector.turn_end_gate_manager._states

@pytest.mark.asyncio
async def test_stream_turn_end_gate_receives_recognized_text(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldGate()
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=0.1,
        turn_end_gates=[gate],
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
    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)

    assert gate.calls
    assert gate.calls[0]["text"] == "こんにちは"

@pytest.mark.asyncio
async def test_silero_max_duration_overrides_gate_hold_without_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldForeverGate()
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[gate],
        max_duration=0.5,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append((recorded_data, recorded_duration, session_id))

    session_id = "silero_max_duration_overrides_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)
    await asyncio.sleep(0.05)

    assert len(detected) == 1
    assert detected[0][2] == session_id
    assert detector.get_session(session_id).is_recording is False

@pytest.mark.asyncio
async def test_stream_max_duration_overrides_gate_hold_without_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldForeverGate()
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=0.1,
        turn_end_gates=[gate],
        max_duration=0.5,
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append((recorded_data, text, recorded_duration, session_id))

    session_id = "stream_max_duration_overrides_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)
    await asyncio.sleep(0.05)

    assert len(detected) == 1
    assert detected[0][1] == "こんにちは"
    assert detected[0][3] == session_id
    assert detector.get_session(session_id).is_recording is False

@pytest.mark.asyncio
async def test_silero_mute_resets_turn_end_gate_manager(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldForeverGate()
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[gate],
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    session_id = "silero_mute_resets_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(3):
        await detector.process_samples(silence_chunk, session_id)

    session = detector.get_session(session_id)
    assert session.turn_end_gate_hold_active is True
    assert session_id in detector.turn_end_gate_manager._states

    detector.should_mute = lambda: True
    await detector.process_samples(silence_chunk, session_id)

    assert session.turn_end_gate_hold_active is False
    assert session_id not in detector.turn_end_gate_manager._states

@pytest.mark.asyncio
async def test_stream_mute_resets_turn_end_gate_manager(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    gate = AlwaysHoldForeverGate()
    detector = SileroStreamSpeechDetector(
        speech_recognizer=DummySpeechRecognizer(),
        silence_duration_threshold=0.2,
        segment_silence_threshold=0.1,
        turn_end_gates=[gate],
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    session_id = "stream_mute_resets_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(3):
        await detector.process_samples(silence_chunk, session_id)

    session = detector.get_session(session_id)
    assert session.turn_end_gate_hold_active is True
    assert session_id in detector.turn_end_gate_manager._states

    detector.should_mute = lambda: True
    await detector.process_samples(silence_chunk, session_id)

    assert session.turn_end_gate_hold_active is False
    assert session_id not in detector.turn_end_gate_manager._states

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
        turn_end_gates=[gate],
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
async def test_silero_turn_end_gates_wait_if_any_gate_waits(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    pass_gate = AlwaysPassGate()
    hold_gate = AlwaysHoldGate(timeout=0.3)
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[pass_gate, hold_gate],
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    session_id = "silero_multiple_gates"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)

    assert pass_gate.calls
    assert hold_gate.calls
    session = detector.get_session(session_id)
    assert session.is_recording is True
    assert session.turn_end_gate_hold_reasons == ["hold"]

@pytest.mark.asyncio
async def test_background_turn_end_gate_can_extend_latched_timeout(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    future = asyncio.get_running_loop().create_future()
    hold_gate = AlwaysHoldGate(timeout=0.3)
    background_gate = BackgroundHoldGate(future, timeout=0.6)
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[hold_gate, background_gate],
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append((recorded_data, recorded_duration, session_id))

    session_id = "silero_background_gate"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(silence_chunk, session_id)
    await detector.process_samples(silence_chunk, session_id)
    await asyncio.sleep(0)

    assert detected == []
    assert background_gate.calls
    future.set_result(None)
    await asyncio.sleep(0)

    for _ in range(3):
        await detector.process_samples(silence_chunk, session_id)

    assert detected == []
    session = detector.get_session(session_id)
    assert session.is_recording is True
    assert session.turn_end_gate_hold_timeout == 0.6
    assert "background_hold" in session.turn_end_gate_hold_reasons

    for _ in range(4):
        await detector.process_samples(silence_chunk, session_id)
    await asyncio.sleep(0.05)

    assert len(detected) == 1
    assert detected[0][2] == session_id

@pytest.mark.asyncio
async def test_turn_end_gate_context_is_passed_between_gates(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", fake_init_silero_model)
    filler_gate = AlwaysHoldGate(timeout=0.3)
    filler_gate.name = "filler"
    observer_gate = AlwaysPassGate()
    observer_gate.name = "observer"
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.2,
        turn_end_gates=[filler_gate, observer_gate],
        min_duration=0.1,
        sample_rate=16000,
        chunk_size=1,
    )
    monkeypatch.setattr(detector, "_detect_speech_silero", detect_non_silent)

    session_id = "silero_gate_context"
    speech_chunk = b"\x01\x00" * 1600
    silence_chunk = b"\x00\x00" * 1600

    await detector.process_samples(speech_chunk, session_id)
    await detector.process_samples(speech_chunk, session_id)
    for _ in range(3):
        await detector.process_samples(silence_chunk, session_id)

    assert observer_gate.calls
    context = observer_gate.calls[0]["context"]
    assert context.is_waiting("filler") is True
