import asyncio
import os
import struct
import wave
import pytest
from pathlib import Path

from aiavatar.sts.vad.stream import SileroStreamSpeechDetector, RecordingSession
from aiavatar.sts.stt.azure import AzureSpeechRecognizer


AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")


def extend_audio(wave_data: bytes, repeat: int) -> bytes:
    """Extend audio by repeating it multiple times."""
    return wave_data * repeat


def scale_audio_amplitude(wave_data: bytes, scale: float) -> bytes:
    """Scale the amplitude of audio samples."""
    samples = struct.unpack("<" + "h" * (len(wave_data) // 2), wave_data)
    scaled = [max(-32768, min(32767, int(s * scale))) for s in samples]
    return struct.pack("<" + "h" * len(scaled), *scaled)


@pytest.fixture
def test_output_dir(tmp_path: Path):
    """Temporary directory to store the file that is created in the each test case."""
    return tmp_path


@pytest.fixture
def stt_wav_path() -> Path:
    """Returns the path to the hello.wav file containing 'こんにちは。'"""
    return Path(__file__).parent.parent / "stt" / "data" / "hello.wav"


@pytest.fixture
def speech_recognizer():
    """Create Azure speech recognizer."""
    return AzureSpeechRecognizer(
        azure_api_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        language="ja-JP",
        debug=True
    )


@pytest.fixture
def detector(test_output_dir, speech_recognizer):
    """Create SileroStreamSpeechDetector with Azure speech recognizer."""
    detector = SileroStreamSpeechDetector(
        speech_recognizer=speech_recognizer,
        silence_duration_threshold=0.5,
        segment_silence_threshold=0.2,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        on_recording_started_min_duration=0.3,
        debug=True
    )

    detected_results = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        detected_results.append({
            "data": recorded_data,
            "text": text,
            "duration": recorded_duration,
            "session_id": session_id
        })

    detector._detected_results = detected_results

    yield detector

    # Cleanup: delete all sessions
    for session_id in list(detector.recording_sessions.keys()):
        detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_process_samples_speech_detection_with_text(detector, stt_wav_path):
    """
    Test that when speech is detected, Azure recognizes the text
    and on_speech_detected receives the recognized text.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    session_id = "test_session"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        wave_data = wav_file.readframes(wav_file.getnframes())

    assert sample_rate == 16000, f"Expected 16000Hz, got {sample_rate}Hz"

    chunk_size = 3200  # 100ms at 16kHz, 16-bit mono

    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence to trigger end of speech detection
    silence_chunk = b'\x00' * chunk_size
    for _ in range(20):  # 2 seconds of silence
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    # Wait for callback to complete
    await asyncio.sleep(0.5)

    # Check results
    assert len(detector._detected_results) >= 1, "No speech detected"
    result = detector._detected_results[0]
    assert result["text"] is not None and len(result["text"]) > 0, "No text recognized"
    assert "こんにちは" in result["text"], f"Expected 'こんにちは', got: {result['text']}"
    assert result["session_id"] == session_id
    assert result["duration"] > 0


@pytest.mark.asyncio
async def test_on_speech_detecting_callback(detector, stt_wav_path):
    """
    Test that on_speech_detecting callback is triggered during recording
    when segment silence is detected.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    detecting_calls = []

    @detector.on_speech_detecting
    async def on_detecting(text: str, session: RecordingSession):
        detecting_calls.append({
            "text": text,
            "session_id": session.session_id,
            "duration": session.segment_duration
        })

    session_id = "test_detecting"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send short silence to trigger segment detection (but not end of recording)
    silence_chunk = b'\x00' * chunk_size
    for _ in range(5):  # ~500ms of silence, enough for segment_silence_threshold
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    # Wait for recognition callback
    await asyncio.sleep(0.5)

    # on_speech_detecting should have been called at least once
    assert len(detecting_calls) >= 1, "on_speech_detecting was not triggered"
    assert "こんにちは" in detecting_calls[0]["text"], f"Expected 'こんにちは', got: {detecting_calls[0]['text']}"
    assert detecting_calls[0]["session_id"] == session_id

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_validate_recognized_text_rejects_invalid(detector, stt_wav_path):
    """
    Test that validate_recognized_text can reject text and prevent on_speech_detected.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    validation_calls = []

    @detector.validate_recognized_text
    def validate_text(text: str):
        validation_calls.append(text)
        # Reject all text
        return "Rejected for testing"

    session_id = "test_validation"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence to end recording
    silence_chunk = b'\x00' * chunk_size
    for _ in range(20):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # Validation should have been called
    assert len(validation_calls) >= 1, "validate_recognized_text was not called"

    # But on_speech_detected should NOT have been called
    assert len(detector._detected_results) == 0, "on_speech_detected should not be called when validation rejects"


@pytest.mark.asyncio
async def test_validate_recognized_text_allows_valid(detector, stt_wav_path):
    """
    Test that validate_recognized_text allows valid text to proceed.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    validation_calls = []

    @detector.validate_recognized_text
    def validate_text(text: str):
        validation_calls.append(text)
        # Allow all text (return None)
        return None

    session_id = "test_validation_allow"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence
    silence_chunk = b'\x00' * chunk_size
    for _ in range(20):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # Validation should have been called
    assert len(validation_calls) >= 1

    # on_speech_detected should have been called
    assert len(detector._detected_results) >= 1
    assert "こんにちは" in detector._detected_results[0]["text"]


@pytest.mark.asyncio
async def test_on_recording_started_callback(stt_wav_path, speech_recognizer):
    """
    Test that on_recording_started callback is triggered when recording duration
    exceeds the threshold.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    callback_calls = []

    async def mock_callback(session_id: str):
        callback_calls.append(session_id)

    detector = SileroStreamSpeechDetector(
        speech_recognizer=speech_recognizer,
        silence_duration_threshold=0.5,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        on_recording_started=mock_callback,
        on_recording_started_min_duration=0.3,
        debug=True
    )

    session_id = "test_callback"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # Callback should have been triggered
    assert len(callback_calls) >= 1, "on_recording_started callback was not triggered"
    assert callback_calls[0] == session_id

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_on_recording_started_with_text_condition(stt_wav_path, speech_recognizer):
    """
    Test that on_recording_started can be triggered by recognized text length.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    callback_calls = []

    async def mock_callback(session_id: str):
        callback_calls.append(session_id)

    detector = SileroStreamSpeechDetector(
        speech_recognizer=speech_recognizer,
        silence_duration_threshold=0.5,
        segment_silence_threshold=0.2,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        on_recording_started=mock_callback,
        on_recording_started_min_duration=10.0,  # Very long - won't trigger by duration
        on_recording_started_min_text_length=2,  # Trigger by text length
        debug=True
    )

    session_id = "test_text_trigger"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send short silence to trigger segment recognition
    silence_chunk = b'\x00' * chunk_size
    for _ in range(5):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # Callback should have been triggered by text length
    assert len(callback_calls) >= 1, "on_recording_started should be triggered by text length"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_should_trigger_recording_started_custom(stt_wav_path, speech_recognizer):
    """
    Test custom should_trigger_recording_started condition.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    callback_calls = []
    trigger_check_calls = []

    detector = SileroStreamSpeechDetector(
        speech_recognizer=speech_recognizer,
        silence_duration_threshold=0.5,
        segment_silence_threshold=0.2,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        on_recording_started_min_duration=0.1,
        debug=True
    )

    @detector.on_recording_started
    async def on_recording_started(session_id: str):
        callback_calls.append(session_id)

    @detector.should_trigger_recording_started
    def custom_trigger(text, session):
        trigger_check_calls.append((text, session.record_duration))
        # Only trigger if text contains specific word
        return text is not None and "こんにちは" in text

    session_id = "test_custom_trigger"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence to trigger segment recognition
    silence_chunk = b'\x00' * chunk_size
    for _ in range(5):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # Custom trigger function should have been called
    assert len(trigger_check_calls) > 0, "Custom trigger function should be called"

    # Callback should have been triggered after custom condition met
    assert len(callback_calls) >= 1, "Callback should be triggered after custom condition met"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_silence_only_no_detection(detector):
    """
    Test that silence-only audio does not trigger speech detection.
    """
    session_id = "test_silence"

    # Generate silence (16kHz, 16-bit mono, 2 seconds)
    silence_samples = 16000 * 2
    silence = b'\x00\x00' * silence_samples

    chunk_size = 3200
    for i in range(0, len(silence), chunk_size):
        chunk = silence[i:i + chunk_size]
        await detector.process_samples(chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # No speech should be detected
    assert len(detector._detected_results) == 0, "Speech was incorrectly detected in silence"


@pytest.mark.asyncio
async def test_multiple_sessions(detector, stt_wav_path):
    """
    Test handling multiple concurrent sessions.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    session_ids = ["session_1", "session_2"]

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send audio for both sessions
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            for session_id in session_ids:
                await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence
    silence_chunk = b'\x00' * chunk_size
    for _ in range(20):
        for session_id in session_ids:
            await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # Both sessions should have detected speech
    results_by_session = {r["session_id"]: r for r in detector._detected_results}

    for session_id in session_ids:
        assert session_id in results_by_session, f"No result for session {session_id}"
        result = results_by_session[session_id]
        assert "こんにちは" in result["text"], f"Expected 'こんにちは' for {session_id}, got: {result['text']}"


@pytest.mark.asyncio
async def test_segment_tracking_reset_on_session_reset(detector, stt_wav_path):
    """
    Test that segment tracking variables are properly reset.
    """
    session_id = "test_segment_reset"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Extract middle portion to avoid silence at beginning/end
    skip_bytes = 16000  # 0.5 sec at 16kHz, 16-bit mono
    middle_portion = wave_data[skip_bytes:-skip_bytes] if len(wave_data) > skip_bytes * 2 else wave_data

    chunk_size = 3200

    # Send audio chunks and check state when recording is active
    recording_started = False
    segment_duration_captured = 0
    for i in range(0, len(middle_portion), chunk_size):
        chunk = middle_portion[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            session = detector.get_session(session_id)
            if session.is_recording:
                recording_started = True
                # Capture segment_duration while recording is active
                if session.segment_duration > 0:
                    segment_duration_captured = session.segment_duration
            await asyncio.sleep(0.01)

    assert recording_started is True, "Recording should have started"
    assert segment_duration_captured > 0, "Segment buffer should have been populated during recording"

    # Reset session
    detector.reset_session(session_id)
    session = detector.get_session(session_id)

    # Check segment tracking is reset
    assert session.is_recording is False
    assert len(session.segment_buffer) == 0
    assert session.segment_duration == 0
    assert session.segment_silence_duration == 0
    assert session.segment_fired is False
    assert session.pending_recognition_task is None
    assert session.recognition_sequence == 0


def test_session_inherits_from_silero_session():
    """
    Test that RecordingSession properly inherits from SileroRecordingSession.
    """
    from aiavatar.sts.vad.silero import RecordingSession as SileroRecordingSession
    from aiavatar.sts.vad.base import RecordingSessionBase

    session = RecordingSession("test_session", preroll_buffer_count=5)

    # Check inheritance chain
    assert isinstance(session, SileroRecordingSession)
    assert isinstance(session, RecordingSessionBase)

    # Check base attributes exist
    assert hasattr(session, 'session_id')
    assert hasattr(session, 'is_recording')
    assert hasattr(session, 'buffer')
    assert hasattr(session, 'silence_duration')
    assert hasattr(session, 'record_duration')

    # Check silero-specific attributes
    assert hasattr(session, 'amplitude_threshold')
    assert hasattr(session, 'vad_buffer')
    assert hasattr(session, 'vad_iterator')

    # Check stream-specific attributes
    assert hasattr(session, 'segment_buffer')
    assert hasattr(session, 'segment_duration')
    assert hasattr(session, 'segment_silence_duration')
    assert hasattr(session, 'segment_fired')
    assert hasattr(session, 'pending_recognition_task')
    assert hasattr(session, 'recognition_sequence')


def test_detector_inherits_from_silero_detector(speech_recognizer):
    """
    Test that SileroStreamSpeechDetector properly inherits from SileroSpeechDetector.
    """
    from aiavatar.sts.vad.silero import SileroSpeechDetector
    from aiavatar.sts.vad.base import SpeechDetector

    detector = SileroStreamSpeechDetector(speech_recognizer=speech_recognizer)

    # Check inheritance chain
    assert isinstance(detector, SileroSpeechDetector)
    assert isinstance(detector, SpeechDetector)

    # Check parent class methods are available
    assert hasattr(detector, 'process_stream')
    assert hasattr(detector, 'finalize_session')
    assert hasattr(detector, 'reset_session')
    assert hasattr(detector, 'delete_session')
    assert hasattr(detector, 'set_speech_probability_threshold')
    assert hasattr(detector, 'reset_vad_state')

    # Check stream-specific attributes
    assert hasattr(detector, 'speech_recognizer')
    assert hasattr(detector, 'segment_silence_threshold')
    assert hasattr(detector, '_on_speech_detecting')
    assert hasattr(detector, '_validate_recognized_text')


@pytest.mark.asyncio
async def test_process_stream(detector, stt_wav_path):
    """
    Test stream processing via process_stream.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    session_id = "test_stream"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    async def async_audio_stream():
        # Send speech audio
        for i in range(0, len(wave_data), chunk_size):
            chunk = wave_data[i:i + chunk_size]
            if chunk:
                yield chunk
                await asyncio.sleep(0.01)

        # Send silence
        silence_chunk = b'\x00' * chunk_size
        for _ in range(20):
            yield silence_chunk
            await asyncio.sleep(0.01)

    await detector.process_stream(async_audio_stream(), session_id)

    await asyncio.sleep(0.5)

    # Check results
    assert len(detector._detected_results) >= 1, "No speech detected in stream"
    result = detector._detected_results[0]
    assert "こんにちは" in result["text"], f"Expected 'こんにちは', got: {result['text']}"

    # Session should be deleted after stream
    assert session_id not in detector.recording_sessions


@pytest.mark.asyncio
async def test_max_duration_exceeded(stt_wav_path, speech_recognizer):
    """
    Test that recording is stopped when max_duration is exceeded.
    Uses extended audio data by repeating the middle portion of the WAV file
    to avoid gaps caused by silence at the beginning/end.
    """
    detector = SileroStreamSpeechDetector(
        speech_recognizer=speech_recognizer,
        silence_duration_threshold=0.5,
        max_duration=3.0,  # 3 seconds max
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        debug=True
    )

    detected_results = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        detected_results.append({"text": text})

    session_id = "test_max_duration"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Extract middle portion of audio (skip first and last 0.5 sec to avoid silence)
    # 16000 samples/sec * 2 bytes * 0.5 sec = 16000 bytes
    skip_bytes = 16000
    middle_portion = wave_data[skip_bytes:-skip_bytes] if len(wave_data) > skip_bytes * 2 else wave_data

    # Repeat the middle portion to create ~10 seconds of continuous speech
    extended_wave_data = extend_audio(middle_portion, 5)

    chunk_size = 3200

    # Send extended audio
    recording_was_active = False
    max_duration_exceeded = False
    for i in range(0, len(extended_wave_data), chunk_size):
        chunk = extended_wave_data[i:i + chunk_size]
        if chunk:
            was_recording = detector.get_session(session_id).is_recording
            is_recording = await detector.process_samples(chunk, session_id)
            if is_recording:
                recording_was_active = True
            # Check if recording was stopped due to max_duration (was recording, now not, no silence sent)
            if was_recording and not is_recording and detector.get_session(session_id).record_duration == 0:
                max_duration_exceeded = True
                break
            await asyncio.sleep(0.01)

    # Recording should have started at some point
    assert recording_was_active is True, "Recording never started"

    # Recording should have been stopped due to max_duration
    assert max_duration_exceeded is True, "Recording should have been stopped due to max_duration"

    await asyncio.sleep(0.3)

    # on_speech_detected should NOT have been called (exceeded max_duration)
    assert len(detected_results) == 0, "on_speech_detected should not be called when max_duration is exceeded"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_volume_db_threshold_filtering(stt_wav_path, speech_recognizer):
    """
    Test that volume_db_threshold properly filters out low volume audio.
    """
    detector = SileroStreamSpeechDetector(
        speech_recognizer=speech_recognizer,
        volume_db_threshold=-10.0,  # High threshold
        silence_duration_threshold=0.5,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        debug=True
    )

    session_id = "test_volume_filter"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Scale down audio to very low volume
    low_volume_data = scale_audio_amplitude(wave_data, 0.01)

    chunk_size = 3200

    # Send low volume audio - should NOT trigger recording
    for i in range(0, len(low_volume_data), chunk_size):
        chunk = low_volume_data[i:i + chunk_size]
        if chunk:
            is_recording = await detector.process_samples(chunk, session_id)
            assert is_recording is False, "Low volume audio should not trigger recording"
            await asyncio.sleep(0.01)

    # Cleanup
    detector.delete_session(session_id)


def test_session_data(detector):
    """
    Test session data storage functionality.
    """
    session_id_1 = "session_id_1"
    session_id_2 = "session_id_2"

    detector.set_session_data(session_id_1, "key", "val")
    assert detector.recording_sessions.get(session_id_1) is None

    detector.set_session_data(session_id_1, "key1", "val1", create_session=True)
    assert detector.recording_sessions.get(session_id_1).data == {"key1": "val1"}
    detector.set_session_data(session_id_1, "key2", "val2")
    assert detector.recording_sessions.get(session_id_1).data == {"key1": "val1", "key2": "val2"}

    assert detector.recording_sessions.get(session_id_2) is None

    # Cleanup
    detector.delete_session(session_id_1)


@pytest.mark.asyncio
async def test_recognition_sequence_ordering(stt_wav_path, speech_recognizer):
    """
    Test that recognition sequence numbers are incremented properly.
    """
    detector = SileroStreamSpeechDetector(
        speech_recognizer=speech_recognizer,
        silence_duration_threshold=1.0,  # Longer threshold to prevent early termination
        segment_silence_threshold=0.1,  # Quick segment detection
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        debug=True
    )

    session_id = "test_sequence"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Extract middle portion to avoid silence at beginning/end
    skip_bytes = 16000  # 0.5 sec at 16kHz, 16-bit mono
    middle_portion = wave_data[skip_bytes:-skip_bytes] if len(wave_data) > skip_bytes * 2 else wave_data

    chunk_size = 3200

    # Send audio to start recording
    recording_started = False
    for i in range(0, len(middle_portion), chunk_size):
        chunk = middle_portion[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            session = detector.get_session(session_id)
            if session.is_recording:
                recording_started = True
                break
            await asyncio.sleep(0.01)

    assert recording_started is True, "Recording should have started"

    # Continue sending more audio to ensure we're recording
    for i in range(chunk_size, len(middle_portion) // 2, chunk_size):
        chunk = middle_portion[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Now insert silence to trigger segment recognition (segment_silence_threshold=0.1 sec)
    # Need at least 0.1 sec of silence = 1600 samples = 3200 bytes
    silence_chunk = b'\x00' * chunk_size
    for _ in range(5):  # ~500ms of silence (well above 0.1 sec threshold)
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    # Wait for recognition task to complete
    await asyncio.sleep(0.5)

    session = detector.get_session(session_id)
    # Recognition sequence should have been incremented
    assert session.recognition_sequence >= 1, f"Recognition sequence should be incremented, got {session.recognition_sequence}"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_process_samples_return_value(detector, stt_wav_path):
    """
    Test that process_samples returns correct boolean value indicating recording status.
    """
    session_id = "test_return_value"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Start recording with speech
    chunk = wave_data[:chunk_size * 3]
    is_recording = await detector.process_samples(chunk, session_id)
    assert is_recording is True, "Should return True when recording starts"

    # Continue recording
    more_chunk = wave_data[chunk_size * 3:chunk_size * 6]
    is_recording = await detector.process_samples(more_chunk, session_id)
    assert is_recording is True, "Should return True while recording continues"

    # Stop with silence
    silence_chunk = b'\x00' * chunk_size
    for _ in range(15):
        is_recording = await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    # After silence, recording should have stopped
    assert is_recording is False, "Should return False after recording stops"


@pytest.mark.asyncio
async def test_muted_detector(detector, stt_wav_path):
    """
    Test that detector properly handles muted state.
    """
    detector.should_mute = lambda: True

    session_id = "test_muted"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200
    chunk = wave_data[:chunk_size * 3]

    # Try to record while muted
    is_recording = await detector.process_samples(chunk, session_id)
    assert is_recording is False, "Should return False when muted"

    session = detector.get_session(session_id)
    assert session.is_recording is False

    # Unmute and try again
    detector.should_mute = lambda: False
    is_recording = await detector.process_samples(chunk, session_id)
    assert is_recording is True, "Should return True when unmuted and speech detected"
