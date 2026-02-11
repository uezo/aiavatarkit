import asyncio
import wave
import pytest
import os
from pathlib import Path

from aiavatar.sts.vad.azure_stream import AzureStreamSpeechDetector

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")


@pytest.fixture
def test_output_dir(tmp_path: Path):
    """
    Temporary directory to store the file that is created in the each test case
    """
    return tmp_path


@pytest.fixture
def stt_wav_path() -> Path:
    """
    Returns the path to the hello.wav file containing "こんにちは。"
    """
    return Path(__file__).parent.parent / "stt" / "data" / "hello.wav"


@pytest.fixture
def detector(test_output_dir):
    """
    Create AzureStreamSpeechDetector with real Azure connection
    """
    detector = AzureStreamSpeechDetector(
        azure_subscription_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        azure_language="ja-JP",
        silence_duration_threshold=0.5,
        max_duration=10.0,
        sample_rate=16000,
        channels=1,
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


async def send_audio_and_wait_for_recognition(detector, session_id: str, wave_data: bytes, chunk_size: int = 3200, timeout: float = 5.0):
    """
    Send audio data and then send silence while waiting for Azure recognition.
    This helper ensures we keep calling process_samples to detect the azure_recognized_event.
    """
    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.05)

    # Send silence and wait for recognition
    silence_chunk = b'\x00' * chunk_size * 10   # Longer than 500ms (32000 is 2sec for 16000Hz)
    start_time = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start_time < timeout:
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.1)

        # Check if we got a result
        if len(detector._detected_results) > 0:
            break


@pytest.mark.asyncio
async def test_process_samples_speech_detection_with_text(detector, stt_wav_path):
    """
    Test to verify that when speech is detected, Azure recognizes the text
    and on_speech_detected is called with the recognized text.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    session_id = "test_session"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    assert sample_rate == 16000, f"Expected 16000Hz, got {sample_rate}Hz"

    # Send audio and wait for recognition
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data)

    # Check results
    assert len(detector._detected_results) >= 1, "No speech detected"
    result = detector._detected_results[0]
    assert result["text"] is not None and len(result["text"]) > 0, "No text recognized"
    assert "こんにちは" in result["text"], f"Expected 'こんにちは', got: {result['text']}"
    assert result["session_id"] == session_id


@pytest.mark.asyncio
async def test_process_stream_with_text(detector, stt_wav_path):
    """
    Test stream processing via process_stream with real Azure recognition.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    session_id = "test_stream"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200  # 100ms at 16kHz, 16-bit mono

    async def async_audio_stream():
        # Send speech audio
        for i in range(0, len(wave_data), chunk_size):
            chunk = wave_data[i:i + chunk_size]
            if chunk:
                yield chunk
                await asyncio.sleep(0.05)

        # Send silence to trigger recognition and keep checking for result
        silence_chunk = b'\x00' * chunk_size
        for _ in range(50):  # Up to 5 seconds of silence
            yield silence_chunk
            await asyncio.sleep(0.1)
            if len(detector._detected_results) > 0:
                break

    await detector.process_stream(async_audio_stream(), session_id)

    # Wait a bit more for the callback to complete
    await asyncio.sleep(0.5)

    # Check results
    assert len(detector._detected_results) >= 1, "No speech detected"
    result = detector._detected_results[0]
    assert result["text"] is not None and len(result["text"]) > 0, "No text recognized"
    assert "こんにちは" in result["text"], f"Expected 'こんにちは', got: {result['text']}"

    # Session should be deleted after stream
    assert session_id not in detector.recording_sessions


@pytest.mark.asyncio
async def test_on_recording_started_callback(stt_wav_path):
    """
    Test that on_recording_started callback is triggered when Azure starts recognizing.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    callback_calls = []

    async def mock_callback(session_id: str):
        callback_calls.append(session_id)

    detector = AzureStreamSpeechDetector(
        azure_subscription_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        azure_language="ja-JP",
        silence_duration_threshold=0.5,
        max_duration=10.0,
        sample_rate=16000,
        channels=1,
        on_recording_started=mock_callback,
        debug=True
    )

    detected_results = []
    detector._detected_results = detected_results

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, recorded_duration: float, session_id: str):
        detected_results.append({"text": text, "session_id": session_id})

    session_id = "test_callback"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio and wait for recognition
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data)

    # Callback should have been triggered
    assert len(callback_calls) >= 1, "on_recording_started callback was not triggered"
    assert callback_calls[0] == session_id

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_session_reset_and_delete(detector, stt_wav_path):
    """
    Test the operation of reset / delete for a session.
    """
    session_id = "test_session_reset"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio to create session and populate preroll buffer
    chunk_size = 3200
    chunk = wave_data[:chunk_size * 3]
    await detector.process_samples(chunk, session_id)
    await asyncio.sleep(0.1)

    # Verify session was created with Azure recognition started
    session = detector.get_session(session_id)
    assert session.azure_push_stream is not None
    assert session.azure_recognizer is not None
    assert len(session.preroll_buffer) > 0

    # Manually set recording state to test reset
    session.is_recording = True
    session.buffer.extend(b'\x00' * 100)
    session.record_duration = 1.0

    # Reset recording session
    detector.reset_session(session_id)
    assert session.is_recording is False
    assert len(session.buffer) == 0
    assert session.record_duration == 0
    # preroll_buffer should NOT be cleared by reset
    assert len(session.preroll_buffer) > 0
    # Azure recognition should still be active
    assert session.azure_push_stream is not None
    assert session.azure_recognizer is not None

    # Delete session
    detector.delete_session(session_id)
    assert session_id not in detector.recording_sessions


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
            await asyncio.sleep(0.05)

    # Send silence and wait for recognition for both sessions
    silence_chunk = b'\x00' * chunk_size
    for _ in range(50):  # Up to 5 seconds
        for session_id in session_ids:
            await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.1)

        # Check if we got results for all sessions
        results_by_session = {r["session_id"]: r for r in detector._detected_results}
        if all(sid in results_by_session for sid in session_ids):
            break

    # Both sessions should have detected speech
    results_by_session = {r["session_id"]: r for r in detector._detected_results}

    for session_id in session_ids:
        assert session_id in results_by_session, f"No result for session {session_id}"
        result = results_by_session[session_id]
        assert "こんにちは" in result["text"], f"Expected 'こんにちは' for {session_id}, got: {result['text']}"

    # Cleanup
    for session_id in session_ids:
        if session_id in detector.recording_sessions:
            detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_silence_only_no_detection(detector):
    """
    Test that silence-only audio does not trigger speech detection.
    """
    session_id = "test_silence"

    # Generate silence (16kHz, 16-bit mono, 2 seconds)
    silence_samples = 16000 * 2  # 2 seconds
    silence = b'\x00\x00' * silence_samples

    # Process silence in chunks
    chunk_size = 3200
    for i in range(0, len(silence), chunk_size):
        chunk = silence[i:i + chunk_size]
        await detector.process_samples(chunk, session_id)
        await asyncio.sleep(0.05)

    # Wait a bit
    await asyncio.sleep(1.0)

    # No speech should be detected
    assert len(detector._detected_results) == 0, "Speech was incorrectly detected in silence"

    # Cleanup
    if session_id in detector.recording_sessions:
        detector.delete_session(session_id)


def test_session_data(detector):
    """Test session data storage functionality"""
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
async def test_validate_recognized_text_rejects_short_text(detector, stt_wav_path):
    """
    Test that validate_recognized_text can reject short text before pipeline processing.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    validation_calls = []
    original_detected_results = detector._detected_results

    @detector.validate_recognized_text
    def validate_text(text: str) -> str | None:
        validation_calls.append(text)
        # Reject all text (simulating validation failure)
        return "Rejected for testing"

    # Use validation_calls as the trigger for send_audio_and_wait_for_recognition
    detector._detected_results = validation_calls
    session_id = "test_validation_reject"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio and wait for validation to be called
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data, timeout=5.0)

    # Wait a bit more for any pending callbacks
    await asyncio.sleep(0.5)

    # Validation should have been called
    assert len(validation_calls) >= 1, "validate_recognized_text was not called"
    assert "こんにちは" in validation_calls[0], f"Expected 'こんにちは', got: {validation_calls[0]}"

    # But on_speech_detected should NOT have been called (rejected by validation)
    assert len(original_detected_results) == 0, "on_speech_detected should not be called when validation rejects"


@pytest.mark.asyncio
async def test_validate_recognized_text_allows_valid_text(detector, stt_wav_path):
    """
    Test that validate_recognized_text allows valid text to proceed.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    validation_calls = []

    @detector.validate_recognized_text
    def validate_text(text: str) -> str | None:
        validation_calls.append(text)
        # Allow all text (return None means valid)
        return None

    session_id = "test_validation_allow"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio and wait for recognition
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data)

    # Validation should have been called
    assert len(validation_calls) >= 1, "validate_recognized_text was not called"

    # on_speech_detected should also have been called (validation passed)
    assert len(detector._detected_results) >= 1, "on_speech_detected should be called when validation passes"
    assert "こんにちは" in detector._detected_results[0]["text"]


@pytest.mark.asyncio
async def test_validate_recognized_text_session_reset_on_reject(detector, stt_wav_path):
    """
    Test that session is properly reset even when validation rejects the text.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    validation_calls = []

    @detector.validate_recognized_text
    def validate_text(text: str) -> str | None:
        validation_calls.append(text)
        return "Rejected"

    # Use validation_calls as the trigger for send_audio_and_wait_for_recognition
    detector._detected_results = validation_calls
    session_id = "test_reset_on_reject"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio and wait for validation to be called
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data, timeout=5.0)

    # Wait a bit more
    await asyncio.sleep(0.5)

    # Validation should have been called
    assert len(validation_calls) >= 1, "validate_recognized_text was not called"

    # Session should be reset (buffer cleared, not recording)
    session = detector.get_session(session_id)
    assert session.is_recording is False, "Session should not be recording after rejection"
    assert len(session.buffer) == 0, "Session buffer should be cleared after rejection"


@pytest.mark.asyncio
async def test_on_speech_detecting_callback(detector, stt_wav_path):
    """
    Test that on_speech_detecting callback is triggered during recognition.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    from aiavatar.sts.vad.azure_stream import RecordingSession

    detecting_calls = []

    @detector.on_speech_detecting
    async def on_detecting(text: str, session: RecordingSession):
        detecting_calls.append({
            "text": text,
            "session_id": session.session_id,
            "is_recording": session.is_recording
        })

    session_id = "test_detecting"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio and wait for recognition
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data)

    # on_speech_detecting should have been called at least once (during recognizing events)
    assert len(detecting_calls) >= 1, "on_speech_detecting was not triggered"
    # Check that text was captured during recognizing phase
    assert detecting_calls[0]["session_id"] == session_id


@pytest.mark.asyncio
async def test_on_recording_started_decorator(stt_wav_path):
    """
    Test that on_recording_started decorator works correctly.
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    callback_calls = []

    detector = AzureStreamSpeechDetector(
        azure_subscription_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        azure_language="ja-JP",
        silence_duration_threshold=0.5,
        max_duration=10.0,
        sample_rate=16000,
        channels=1,
        debug=True
    )

    @detector.on_recording_started
    async def on_started(session_id: str):
        callback_calls.append(session_id)

    detected_results = []
    detector._detected_results = detected_results

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        detected_results.append({"text": text, "session_id": session_id})

    session_id = "test_decorator"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio and wait for recognition
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data)

    # Callback should have been triggered
    assert len(callback_calls) >= 1, "on_recording_started decorator callback was not triggered"
    assert callback_calls[0] == session_id

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_max_duration_exceeded(stt_wav_path):
    """
    Test that recording handles max_duration properly.
    NOTE: Azure has a minimum of 20 seconds for max_duration, so this test
    verifies that the setting is applied correctly.
    """
    detector = AzureStreamSpeechDetector(
        azure_subscription_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        azure_language="ja-JP",
        silence_duration_threshold=0.5,
        max_duration=3.0,  # Will be capped to 20 seconds by Azure
        sample_rate=16000,
        channels=1,
        debug=True
    )

    detected_results = []
    detector._detected_results = detected_results

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        detected_results.append({"text": text, "duration": recorded_duration})

    session_id = "test_max_duration"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    # Send audio and wait for recognition
    await send_audio_and_wait_for_recognition(detector, session_id, wave_data)

    # Should still detect speech (max_duration doesn't prevent detection)
    assert len(detected_results) >= 1, "Speech should still be detected"
    assert "こんにちは" in detected_results[0]["text"]

    # Cleanup
    detector.delete_session(session_id)


def test_detector_inherits_from_speech_detector():
    """
    Test that AzureStreamSpeechDetector properly inherits from SpeechDetector.
    """
    from aiavatar.sts.vad.base import SpeechDetector

    detector = AzureStreamSpeechDetector(
        azure_subscription_key="test_key",
        azure_region="test_region",
        azure_language="ja-JP"
    )

    # Check inheritance chain
    assert isinstance(detector, SpeechDetector)

    # Check parent class methods are available
    assert hasattr(detector, 'process_stream')
    assert hasattr(detector, 'finalize_session')
    assert hasattr(detector, 'reset_session')
    assert hasattr(detector, 'delete_session')
    assert hasattr(detector, 'on_speech_detected')
    assert hasattr(detector, 'on_recording_started')

    # Check Azure-specific attributes
    assert hasattr(detector, 'azure_subscription_key')
    assert hasattr(detector, 'azure_region')
    assert hasattr(detector, 'azure_language')
    assert hasattr(detector, '_on_speech_detecting')
    assert hasattr(detector, '_validate_recognized_text')


def test_recording_session_attributes():
    """
    Test that RecordingSession has all required attributes.
    """
    from aiavatar.sts.vad.azure_stream import RecordingSession

    session = RecordingSession("test_session", preroll_buffer_size=10)

    # Check required attributes
    assert hasattr(session, 'session_id')
    assert hasattr(session, 'is_recording')
    assert hasattr(session, 'buffer')
    assert hasattr(session, 'record_duration')
    assert hasattr(session, 'preroll_buffer')
    assert hasattr(session, 'data')
    assert hasattr(session, 'on_recording_started_triggered')

    # Check Azure-specific attributes
    assert hasattr(session, 'azure_push_stream')
    assert hasattr(session, 'azure_recognizer')
    assert hasattr(session, 'event_loop')

    # Check initial values
    assert session.session_id == "test_session"
    assert session.is_recording is False
    assert len(session.buffer) == 0
    assert session.record_duration == 0
    assert session.on_recording_started_triggered is False
    assert session.azure_push_stream is None
    assert session.azure_recognizer is None


def test_recording_session_reset():
    """
    Test that RecordingSession.reset() properly clears state.
    """
    from aiavatar.sts.vad.azure_stream import RecordingSession

    session = RecordingSession("test_session", preroll_buffer_size=10)

    # Set some state
    session.is_recording = True
    session.buffer.extend(b'\x00' * 100)
    session.record_duration = 2.5
    session.on_recording_started_triggered = True
    session.preroll_buffer.append(b'\x00' * 50)

    # Reset
    session.reset()

    # Check that recording state is reset
    assert session.is_recording is False
    assert len(session.buffer) == 0
    assert session.record_duration == 0
    assert session.on_recording_started_triggered is False

    # preroll_buffer should NOT be cleared by reset
    assert len(session.preroll_buffer) > 0


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
    assert len(session.preroll_buffer) == 0, "Preroll buffer should be cleared when muted"


def test_get_config(detector):
    """
    Test that get_config returns expected configuration.
    """
    config = detector.get_config()

    assert "azure_language" in config
    assert "silence_duration_threshold" in config
    assert "max_duration" in config
    assert "sample_rate" in config
    assert "channels" in config
    assert "preroll_buffer_sec" in config
    assert "debug" in config

    assert config["azure_language"] == "ja-JP"
    assert config["silence_duration_threshold"] == 0.5
    assert config["max_duration"] == 10.0
    assert config["sample_rate"] == 16000
    assert config["channels"] == 1


@pytest.mark.asyncio
async def test_finalize_session(detector, stt_wav_path):
    """
    Test that finalize_session properly cleans up a session.
    """
    session_id = "test_finalize"

    # Create a session by sending some audio
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk = wave_data[:3200]
    await detector.process_samples(chunk, session_id)

    # Verify session exists
    assert session_id in detector.recording_sessions

    # Finalize session
    await detector.finalize_session(session_id)

    # Session should be deleted
    assert session_id not in detector.recording_sessions


def test_calculate_preroll_buffer_size():
    """
    Test that preroll buffer size is calculated correctly.
    """
    detector = AzureStreamSpeechDetector(
        azure_subscription_key="test_key",
        azure_region="test_region",
        azure_language="ja-JP",
        sample_rate=16000,
        channels=1,
        preroll_buffer_sec=2.0
    )

    # Calculate expected size
    # bytes_per_sec = 16000 * 1 * 2 = 32000
    # preroll_buffer_size = (2.0 * 32000) / 512 = 125
    expected_size = int((2.0 * 32000) / 512)
    actual_size = detector._calculate_preroll_buffer_size()

    assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size}"
