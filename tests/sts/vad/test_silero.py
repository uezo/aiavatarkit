import asyncio
import struct
import wave
import pytest
from pathlib import Path

from aiavatar.sts.vad.silero import SileroSpeechDetector


def extend_audio(wave_data: bytes, repeat: int) -> bytes:
    """Extend audio by repeating it multiple times."""
    return wave_data * repeat


def scale_audio_amplitude(wave_data: bytes, scale: float) -> bytes:
    """Scale the amplitude of audio samples.

    Args:
        wave_data: Raw PCM audio data (16-bit signed)
        scale: Scale factor (0.0 to 1.0 to reduce, >1.0 to increase)

    Returns:
        Scaled audio data
    """
    # Unpack 16-bit signed samples
    samples = struct.unpack("<" + "h" * (len(wave_data) // 2), wave_data)
    # Scale and clamp to int16 range
    scaled = [max(-32768, min(32767, int(s * scale))) for s in samples]
    # Pack back to bytes
    return struct.pack("<" + "h" * len(scaled), *scaled)


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
    Create SileroSpeechDetector with real Silero VAD model
    """
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.5,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,  # Use direct model for simpler probability-based detection
        debug=True
    )

    detected_results = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        detected_results.append({
            "data": recorded_data,
            "duration": recorded_duration,
            "session_id": session_id
        })

    detector._detected_results = detected_results

    yield detector

    # Cleanup: delete all sessions
    for session_id in list(detector.recording_sessions.keys()):
        detector.delete_session(session_id)


@pytest.fixture
def detector_with_vad_iterator(test_output_dir):
    """
    Create SileroSpeechDetector with real Silero VAD model using VAD iterator
    """
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.5,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=True,  # Use VAD iterator for stateful processing
        debug=True
    )

    detected_results = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        detected_results.append({
            "data": recorded_data,
            "duration": recorded_duration,
            "session_id": session_id
        })

    detector._detected_results = detected_results

    yield detector

    # Cleanup: delete all sessions
    for session_id in list(detector.recording_sessions.keys()):
        detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_process_samples_speech_detection(detector, stt_wav_path, test_output_dir):
    """
    Test to verify that when speech audio is provided, recording starts,
    and after silence, recording ends and on_speech_detected is called.
    Uses real Silero VAD model.
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
    assert result["session_id"] == session_id
    assert result["duration"] > 0, "Recording duration should be greater than 0"
    assert len(result["data"]) > 0, "Recorded data should not be empty"


@pytest.mark.asyncio
async def test_process_samples_with_vad_iterator(detector_with_vad_iterator, stt_wav_path):
    """
    Test speech detection using VAD iterator mode.
    Uses real Silero VAD model with stateful processing.
    """
    session_id = "test_vad_iterator"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send speech audio
    for i in range(0, len(wave_data), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector_with_vad_iterator.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence to trigger end of speech detection
    silence_chunk = b'\x00' * chunk_size
    for _ in range(20):
        await detector_with_vad_iterator.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.5)

    # Check results
    assert len(detector_with_vad_iterator._detected_results) >= 1, "No speech detected with VAD iterator"
    result = detector_with_vad_iterator._detected_results[0]
    assert result["duration"] > 0


@pytest.mark.asyncio
async def test_silence_only_no_detection(detector):
    """
    Test that silence-only audio does not trigger speech detection.
    Uses real Silero VAD model.
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
async def test_on_recording_started_callback(stt_wav_path):
    """
    Test that on_recording_started callback is triggered when speech is detected.
    Uses real Silero VAD model.
    """
    callback_calls = []

    async def mock_callback(session_id: str):
        callback_calls.append(session_id)

    detector = SileroSpeechDetector(
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
        on_recording_started_min_duration=0.3,  # Trigger quickly
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

    # Wait for callback
    await asyncio.sleep(0.5)

    # Callback should have been triggered
    assert len(callback_calls) >= 1, "on_recording_started callback was not triggered"
    assert callback_calls[0] == session_id

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_process_stream(detector, stt_wav_path):
    """
    Test stream processing via process_stream.
    Uses real Silero VAD model.
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
    assert result["duration"] > 0

    # Session should be deleted after stream
    assert session_id not in detector.recording_sessions


@pytest.mark.asyncio
async def test_multiple_sessions(detector, stt_wav_path):
    """
    Test handling multiple concurrent sessions.
    Uses real Silero VAD model.
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
        assert result["duration"] > 0


@pytest.mark.asyncio
async def test_short_recording_not_detected(detector):
    """
    Test that very short speech is not detected (below min_duration).
    Uses real Silero VAD model.
    """
    session_id = "test_short"

    # Create a very short "speech" by using a small portion of audio
    # This may or may not trigger based on actual content, so we use generated data
    # that looks like speech but is very short
    import struct

    # Generate a short burst (50ms = 800 samples at 16kHz)
    short_samples = struct.pack("<" + "h" * 800, *([10000] * 800))
    await detector.process_samples(short_samples, session_id)

    # Immediately send silence
    silence = b'\x00' * 16000  # 500ms of silence
    await detector.process_samples(silence, session_id)

    await asyncio.sleep(0.3)

    # Recording should not be detected due to min_duration
    session = detector.get_session(session_id)
    assert session.is_recording is False


def test_model_initialization(detector):
    """
    Test that Silero VAD model is properly initialized.
    """
    assert len(detector.model_pool) == 1
    assert len(detector.model_locks) == 1
    assert detector.model_pool[0] is not None


def test_model_pool_size():
    """
    Test model pool initialization with different sizes.
    """
    detector = SileroSpeechDetector(model_pool_size=2)
    assert len(detector.model_pool) == 2
    assert len(detector.model_locks) == 2


def test_speech_probability_threshold_change(detector):
    """
    Test that speech_probability_threshold can be changed.
    """
    detector.set_speech_probability_threshold(0.7)
    assert detector.speech_probability_threshold == 0.7

    detector.set_speech_probability_threshold(0.5)
    assert detector.speech_probability_threshold == 0.5


def test_volume_db_threshold_property(detector):
    """
    Test that volume_db_threshold property can be updated dynamically.
    """
    detector.volume_db_threshold = -10.0
    assert detector.volume_db_threshold == -10.0
    expected_threshold = 32767 * (10 ** (-10.0 / 20.0))
    assert abs(detector.amplitude_threshold - expected_threshold) < 1

    detector.volume_db_threshold = None
    assert detector.volume_db_threshold is None
    assert detector.amplitude_threshold is None


@pytest.mark.asyncio
async def test_session_reset_and_delete(detector, stt_wav_path):
    """
    Test the operation of reset / delete for a session.
    Uses real Silero VAD model.
    """
    session_id = "test_session_reset"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Start recording with some audio
    chunk = wave_data[:chunk_size * 3]
    await detector.process_samples(chunk, session_id)
    await asyncio.sleep(0.1)

    session = detector.get_session(session_id)
    assert session.is_recording is True
    assert len(session.buffer) > 0

    # Reset recording session
    detector.reset_session(session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is False
    assert len(session.buffer) == 0

    # Start again
    await detector.process_samples(chunk, session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is True
    assert len(session.buffer) > 0

    # Delete session
    detector.delete_session(session_id)
    assert session_id not in detector.recording_sessions


@pytest.mark.asyncio
async def test_on_recording_started_callback_error_handling(stt_wav_path):
    """
    Test that errors in on_recording_started callback are handled gracefully.
    Recording should complete and on_speech_detected should be called even if
    on_recording_started callback raises an error.
    Uses real Silero VAD model.
    """
    callback_calls = []
    detected_results = []

    async def failing_callback(session_id: str):
        callback_calls.append(session_id)
        raise ValueError("Test error in callback")

    detector = SileroSpeechDetector(
        silence_duration_threshold=0.5,
        max_duration=10.0,
        min_duration=0.1,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        on_recording_started=failing_callback,
        on_recording_started_min_duration=0.3,  # Trigger quickly
        debug=True
    )

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
        detected_results.append({
            "data": recorded_data,
            "duration": recorded_duration,
            "session_id": session_id
        })

    session_id = "test_error_callback"

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

    # Send silence to trigger end of recording
    silence_chunk = b'\x00' * chunk_size
    for _ in range(20):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    # Wait for async callbacks to complete
    await asyncio.sleep(0.3)

    # Verify on_recording_started callback was called (even though it raised an error)
    assert len(callback_calls) >= 1, "on_recording_started callback was not called"

    # Verify recording completed successfully despite callback error
    assert len(detected_results) >= 1, "on_speech_detected was not called - recording did not complete"
    assert detected_results[0]["duration"] > 0, "Recording duration should be greater than 0"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_process_samples_short_recording(stt_wav_path):
    """
    Verify that if recording starts but falls silent before min_duration,
    on_speech_detected is not called.
    Uses real Silero VAD model with longer min_duration.
    """
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.3,
        max_duration=10.0,
        min_duration=3.0,  # Set min_duration longer than the WAV file
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
        detected_results.append({
            "duration": recorded_duration,
            "session_id": session_id
        })

    session_id = "test_short"

    # Load WAV file (hello.wav is about 3.5 seconds)
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send only a portion of the audio (less than min_duration of 3 seconds)
    # 16000 samples/sec * 2 bytes * 2 sec = 64000 bytes
    short_wave_data = wave_data[:64000]

    for i in range(0, len(short_wave_data), chunk_size):
        chunk = short_wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence to trigger end of recording
    silence_chunk = b'\x00' * chunk_size
    for _ in range(10):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.3)

    # on_speech_detected should NOT have been called (recording too short)
    assert len(detected_results) == 0, "Speech was detected even though recording was shorter than min_duration"

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
async def test_process_samples_return_value(detector, stt_wav_path):
    """
    Test that process_samples returns correct boolean value indicating recording status.
    Uses real Silero VAD model.
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
    for _ in range(15):  # Send enough silence to trigger end
        is_recording = await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    # After silence, recording should have stopped
    assert is_recording is False, "Should return False after recording stops"


@pytest.mark.asyncio
async def test_process_samples_return_value_when_muted(detector, stt_wav_path):
    """
    Test that process_samples returns False when detector is muted.
    Uses real Silero VAD model.
    """
    # Mute the detector
    detector.should_mute = lambda: True

    session_id = "test_muted_return"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200
    chunk = wave_data[:chunk_size * 3]

    # Try to record while muted
    is_recording = await detector.process_samples(chunk, session_id)
    assert is_recording is False, "Should return False when muted"

    # Session should not be recording
    session = detector.get_session(session_id)
    assert session.is_recording is False

    # Unmute and try again
    detector.should_mute = lambda: False
    is_recording = await detector.process_samples(chunk, session_id)
    assert is_recording is True, "Should return True when unmuted and speech detected"


@pytest.mark.asyncio
async def test_process_samples_max_duration(stt_wav_path):
    """
    Verify that when sound continues beyond max_duration,
    recording is automatically stopped and on_speech_detected is not called.
    Uses extended audio data by repeating the middle portion of the WAV file
    to avoid gaps caused by silence at the beginning/end.
    """
    detector = SileroSpeechDetector(
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
        detected_results.append({
            "duration": recorded_duration,
            "session_id": session_id
        })

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
async def test_volume_db_threshold_filtering(stt_wav_path):
    """
    Test that volume_db_threshold properly filters out low volume audio
    even when speech probability is high.
    Uses scaled audio data to simulate low volume.
    """
    detector = SileroSpeechDetector(
        volume_db_threshold=-10.0,  # Only sounds louder than -10 dB will be considered
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

    # Scale down audio to very low volume (1% of original)
    low_volume_data = scale_audio_amplitude(wave_data, 0.01)

    chunk_size = 3200

    # Send low volume audio - should NOT trigger recording due to volume threshold
    for i in range(0, len(low_volume_data), chunk_size):
        chunk = low_volume_data[i:i + chunk_size]
        if chunk:
            is_recording = await detector.process_samples(chunk, session_id)
            # Recording should not start with low volume
            assert is_recording is False, "Low volume audio should not trigger recording"
            await asyncio.sleep(0.01)

    session = detector.get_session(session_id)
    assert session.is_recording is False, "Recording should not have started with low volume audio"

    # Now send normal volume audio - should trigger recording
    detector.reset_session(session_id)
    for i in range(0, min(len(wave_data), chunk_size * 5), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            is_recording = await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    session = detector.get_session(session_id)
    assert session.is_recording is True, "Normal volume audio should trigger recording"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_on_recording_started_min_duration(stt_wav_path):
    """
    Test that on_recording_started callback is triggered based on on_recording_started_min_duration.
    """
    callback_calls = []

    async def mock_callback(session_id: str):
        callback_calls.append(session_id)

    # Set min_duration to 0.5 sec so callback triggers quickly
    detector = SileroSpeechDetector(
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
        on_recording_started_min_duration=0.5,  # Trigger after 0.5 sec
        debug=True
    )

    session_id = "test_min_duration"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200  # 100ms at 16kHz

    # Send enough audio to exceed min_duration (0.5 sec = 5 chunks)
    for i in range(0, min(len(wave_data), chunk_size * 10), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    await asyncio.sleep(0.3)

    # Callback should have been triggered after min_duration
    assert len(callback_calls) >= 1, "on_recording_started should be triggered after min_duration"
    assert callback_calls[0] == session_id

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_on_recording_started_not_triggered_for_short_recording(stt_wav_path):
    """
    Test that on_recording_started callback is NOT triggered if recording
    ends before on_recording_started_min_duration.
    """
    callback_calls = []

    async def mock_callback(session_id: str):
        callback_calls.append(session_id)

    # Set min_duration very long so callback won't trigger
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.3,
        max_duration=10.0,
        min_duration=0.1,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        on_recording_started=mock_callback,
        on_recording_started_min_duration=10.0,  # Very long - won't be reached
        debug=True
    )

    session_id = "test_no_trigger"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send short audio
    for i in range(0, min(len(wave_data), chunk_size * 5), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    # Send silence to end recording
    silence_chunk = b'\x00' * chunk_size
    for _ in range(10):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.3)

    # Callback should NOT have been triggered
    assert len(callback_calls) == 0, "on_recording_started should not trigger for short recording"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_on_recording_started_decorator(stt_wav_path):
    """
    Test on_recording_started decorator functionality.
    """
    callback_calls = []

    detector = SileroSpeechDetector(
        silence_duration_threshold=0.5,
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

    @detector.on_recording_started
    async def on_recording_started(session_id: str):
        callback_calls.append(session_id)

    session_id = "test_decorator"

    # Load WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        wave_data = wav_file.readframes(wav_file.getnframes())

    chunk_size = 3200

    # Send audio
    for i in range(0, min(len(wave_data), chunk_size * 10), chunk_size):
        chunk = wave_data[i:i + chunk_size]
        if chunk:
            await detector.process_samples(chunk, session_id)
            await asyncio.sleep(0.01)

    await asyncio.sleep(0.3)

    # Callback should have been triggered
    assert len(callback_calls) >= 1, "Decorator-registered callback should be triggered"
    assert callback_calls[0] == session_id

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_should_trigger_recording_started_custom(stt_wav_path):
    """
    Test custom should_trigger_recording_started condition.
    """
    callback_calls = []
    trigger_check_calls = []

    detector = SileroSpeechDetector(
        silence_duration_threshold=0.5,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        channels=1,
        preroll_buffer_count=5,
        speech_probability_threshold=0.5,
        chunk_size=512,
        use_vad_iterator=False,
        on_recording_started_min_duration=0.1,  # Low default
        debug=True
    )

    @detector.on_recording_started
    async def on_recording_started(session_id: str):
        callback_calls.append(session_id)

    @detector.should_trigger_recording_started
    def custom_trigger(text, session):
        trigger_check_calls.append(session.record_duration)
        # Only trigger if recording duration >= 0.5 sec
        return session.record_duration - session.silence_duration >= 0.5

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

    await asyncio.sleep(0.3)

    # Custom trigger function should have been called
    assert len(trigger_check_calls) > 0, "Custom trigger function should be called"

    # Callback should have been triggered after custom condition met
    assert len(callback_calls) >= 1, "Callback should be triggered after custom condition met"

    # Cleanup
    detector.delete_session(session_id)


@pytest.mark.asyncio
async def test_should_trigger_recording_started_blocks_callback(stt_wav_path):
    """
    Test that custom should_trigger_recording_started can prevent callback.
    """
    callback_calls = []

    detector = SileroSpeechDetector(
        silence_duration_threshold=0.3,
        max_duration=10.0,
        min_duration=0.1,
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
    def never_trigger(text, session):
        # Always return False - never trigger
        return False

    session_id = "test_block_trigger"

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

    # Send silence to end recording
    silence_chunk = b'\x00' * chunk_size
    for _ in range(10):
        await detector.process_samples(silence_chunk, session_id)
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.3)

    # Callback should NOT have been triggered due to custom condition
    assert len(callback_calls) == 0, "Callback should not trigger when custom condition returns False"

    # Cleanup
    detector.delete_session(session_id)


def test_on_recording_started_triggered_flag():
    """
    Test that on_recording_started_triggered flag is set correctly in session.
    """
    detector = SileroSpeechDetector(
        silence_duration_threshold=0.5,
        max_duration=10.0,
        min_duration=0.2,
        sample_rate=16000,
        debug=True
    )

    session_id = "test_flag"
    session = detector.get_session(session_id)

    # Initially False
    assert session.on_recording_started_triggered is False

    # After reset, still False
    session.on_recording_started_triggered = True
    session.reset()
    assert session.on_recording_started_triggered is False

    # Cleanup
    detector.delete_session(session_id)


def test_session_inherits_from_base():
    """
    Test that RecordingSession properly inherits from RecordingSessionBase.
    """
    from aiavatar.sts.vad.silero import RecordingSession
    from aiavatar.sts.vad.base import RecordingSessionBase

    session = RecordingSession("test_session", preroll_buffer_count=5)

    # Check inheritance
    assert isinstance(session, RecordingSessionBase)

    # Check base attributes exist
    assert hasattr(session, 'session_id')
    assert hasattr(session, 'is_recording')
    assert hasattr(session, 'buffer')
    assert hasattr(session, 'silence_duration')
    assert hasattr(session, 'record_duration')
    assert hasattr(session, 'preroll_buffer')
    assert hasattr(session, 'data')
    assert hasattr(session, 'on_recording_started_triggered')
    assert hasattr(session, 'last_recognized_text')

    # Check silero-specific attributes
    assert hasattr(session, 'amplitude_threshold')
    assert hasattr(session, 'vad_buffer')
    assert hasattr(session, 'vad_iterator')
    assert hasattr(session, 'is_speech_active')
