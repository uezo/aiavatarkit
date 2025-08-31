import asyncio
import struct
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from aiavatar.sts.vad.silero import SileroSpeechDetector


@pytest.fixture
def test_output_dir(tmp_path: Path):
    """
    Temporary directory to store the file that is created in the each test case
    """
    return tmp_path


@pytest.fixture
def mock_silero_model():
    """
    Mock Silero VAD model to avoid downloading during tests
    """
    model = MagicMock()
    model.return_value = torch.tensor(0.7)  # Mock speech probability
    
    # Create a mock VAD iterator class that returns new instances
    vad_iterator_class = MagicMock()
    vad_iterator_class.side_effect = lambda *_args, **_kwargs: MagicMock()
    
    utils = (None, None, None, vad_iterator_class, None)
    
    return model, utils


@pytest.fixture
def detector(test_output_dir, mock_silero_model):
    """
    Create SileroSpeechDetector with mocked model
    """
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            debug=True
        )

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, recorded_duration: float, session_id: str):
        output_file = test_output_dir / f"speech_{session_id}.pcm"
        with open(output_file, "wb") as f:
            f.write(recorded_data)

    return detector


@pytest.fixture
def detector_with_model_pool(test_output_dir, mock_silero_model):
    """
    Create SileroSpeechDetector with model pool (size=2)
    """
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            model_pool_size=2,
            debug=True
        )

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data: bytes, recorded_duration: float, session_id: str):
        output_file = test_output_dir / f"speech_{session_id}.pcm"
        with open(output_file, "wb") as f:
            f.write(recorded_data)

    return detector


def generate_samples(amplitude: int, num_samples: int) -> bytes:
    """Generate audio samples for testing"""
    data = [amplitude] * num_samples
    return struct.pack("<" + "h" * num_samples, *data)


@pytest.mark.asyncio
async def test_process_samples_speech_detection(detector, test_output_dir):
    """
    Test to verify that when data exceeding the speech probability threshold is provided, 
    recording starts, and after silence, recording ends, and on_speech_detected is called.
    """
    session_id = "test_session"

    # Mock high speech probability
    detector.model_pool[0].return_value = torch.tensor(0.8)  # Above threshold (0.5)

    assert detector.min_duration == 0.5

    # Start with loud samples (0.5 sec, same as min_duration)
    # Need enough samples to trigger VAD (chunk_size * 2)
    speech_samples = generate_samples(amplitude=1200, num_samples=8000)
    await detector.process_samples(speech_samples, session_id=session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is True

    # Mock low speech probability for silence
    detector.model_pool[0].return_value = torch.tensor(0.2)  # Below threshold
    
    # Stop with silent samples
    silent_samples = generate_samples(amplitude=0, num_samples=16000)
    await detector.process_samples(silent_samples, session_id=session_id)

    # Wait for on_speech_detected invoked
    await asyncio.sleep(0.2)

    # Check whether the file that is created on_speech_detected exists
    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert output_file.exists(), "Recorded file doesn't exist"
    file_size = output_file.stat().st_size
    assert file_size > 0, "No data in the recorded file"


@pytest.mark.asyncio
async def test_process_samples_short_recording(detector, test_output_dir):
    """
    Verify that if recording starts but falls silent before min_duration, 
    on_speech_detected is not called, and no file is created.
    """
    session_id = "test_short"

    # Mock high speech probability initially
    detector.model_pool[0].return_value = torch.tensor(0.8)

    assert detector.min_duration == 0.5

    # Speech samples slightly shorter than 0.5 sec (0.5 = 8000 samples)
    speech_samples = generate_samples(amplitude=1000, num_samples=7999)
    await detector.process_samples(speech_samples, session_id=session_id)

    # Mock low speech probability for silence
    detector.model_pool[0].return_value = torch.tensor(0.2)

    # Stop with silent samples
    silent_samples = generate_samples(amplitude=0, num_samples=8000)
    await detector.process_samples(silent_samples, session_id=session_id)

    await asyncio.sleep(0.2)

    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert not output_file.exists(), "File exists even the samples are shorter than min_duration"


@pytest.mark.asyncio
async def test_process_samples_max_duration(detector, test_output_dir):
    """
    Verify that when sound continues beyond max_duration (3 seconds), 
    recording is automatically stopped, and on_speech_detected is not called.
    """
    session_id = "test_max_duration"

    # Mock high speech probability throughout
    detector.model_pool[0].return_value = torch.tensor(0.8)

    assert detector.max_duration == 3.0

    # Start recording
    speech_samples = generate_samples(amplitude=2000, num_samples=16000)
    await detector.process_samples(speech_samples, session_id=session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is True

    # Continue beyond max_duration (3.0 = 48000 samples total)
    more_speech_samples = generate_samples(amplitude=2000, num_samples=32000)
    await detector.process_samples(more_speech_samples, session_id=session_id)
    session = detector.get_session(session_id)
    # Recording should be stopped due to max_duration
    assert session.is_recording is False

    await asyncio.sleep(0.2)

    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert not output_file.exists(), "File exists even the samples exceeded max_duration"


@pytest.mark.asyncio
async def test_process_stream(detector, test_output_dir):
    """
    Test stream processing via process_stream.
    """
    session_id = "test_stream"

    # Mock speech probability sequence
    speech_probs = [0.8, 0.8, 0.2]  # speech, speech, silence
    speech_prob_iter = iter(speech_probs)
    detector.model_pool[0].side_effect = lambda *_: torch.tensor(next(speech_prob_iter, 0.2))

    assert detector.min_duration == 0.5

    async def async_audio_stream():
        # Start recording with speech samples
        yield generate_samples(amplitude=1500, num_samples=16000)
        await asyncio.sleep(0.1)
        # More speech samples
        yield generate_samples(amplitude=1500, num_samples=3200)
        # Stop with silent samples
        yield generate_samples(amplitude=0, num_samples=16000)
        return

    await detector.process_stream(async_audio_stream(), session_id=session_id)

    # Wait for on_speech_detected invoked
    await asyncio.sleep(0.2)

    output_file = test_output_dir / f"speech_{session_id}.pcm"
    assert output_file.exists(), "Recorded file doesn't exist"
    file_size = output_file.stat().st_size
    assert file_size > 0, "No data in the recorded file"

    # Session is deleted after stream
    assert session_id not in detector.recording_sessions


@pytest.mark.asyncio
async def test_session_reset_and_delete(detector, test_output_dir):
    """
    Test the operation of reset / delete for a session.
    """
    session_id = "test_session_reset"

    # Mock high speech probability
    detector.model_pool[0].return_value = torch.tensor(0.8)

    assert detector.min_duration == 0.5
    speech_samples = generate_samples(amplitude=1000, num_samples=8000)

    # Start recording
    await detector.process_samples(speech_samples, session_id=session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is True
    assert len(session.buffer) > 0

    # Reset recording session
    detector.reset_session(session_id)
    session = detector.get_session(session_id)
    assert session.is_recording is False
    assert len(session.buffer) == 0

    # Start again
    await detector.process_samples(speech_samples, session_id=session_id)
    assert session.is_recording is True
    assert len(session.buffer) > 0

    # Delete session
    session = detector.get_session(session_id)
    detector.delete_session(session_id)
    session.is_recording = False
    assert len(session.buffer) == 0
    assert session_id not in detector.recording_sessions


@pytest.mark.asyncio
async def test_speech_probability_threshold_change(detector):
    """
    Verify that when speech_probability_threshold is changed, 
    the detection behavior changes accordingly.
    """
    session_id = "test_threshold_change"

    # Set threshold to 0.7 (stricter)
    detector.set_speech_probability_threshold(0.7)
    assert detector.speech_probability_threshold == 0.7

    # Mock probability below new threshold
    detector.model_pool[0].return_value = torch.tensor(0.6)  # Below 0.7
    
    samples = generate_samples(amplitude=1000, num_samples=3200)
    await detector.process_samples(samples, session_id=session_id)

    session = detector.get_session(session_id)
    assert session.is_recording is False

    # Mock probability above new threshold
    detector.model_pool[0].return_value = torch.tensor(0.8)  # Above 0.7
    
    samples = generate_samples(amplitude=1000, num_samples=3200)
    await detector.process_samples(samples, session_id=session_id)

    session = detector.get_session(session_id)
    assert session.is_recording is True


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


def test_model_pool_initialization(mock_silero_model):
    """Test model pool initialization with different sizes"""
    model, utils = mock_silero_model
    
    # Test single model (default)
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(model_pool_size=1)
        assert len(detector.model_pool) == 1
        assert len(detector.model_locks) == 1

    # Test multiple models
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(model_pool_size=3)
        assert len(detector.model_pool) == 3
        assert len(detector.model_locks) == 3


def test_get_model_and_lock(detector_with_model_pool):
    """Test model assignment consistency"""
    # Same session should always get same model
    session_id = "test_session"
    model1, lock1 = detector_with_model_pool._get_model_and_lock(session_id)
    model2, lock2 = detector_with_model_pool._get_model_and_lock(session_id)
    
    assert model1 is model2
    assert lock1 is lock2

    # Different sessions might get different models
    session_id_2 = "test_session_2"
    model3, lock3 = detector_with_model_pool._get_model_and_lock(session_id_2)
    
    # Should be one of the models in the pool
    assert model3 in detector_with_model_pool.model_pool
    assert lock3 in detector_with_model_pool.model_locks


@pytest.mark.asyncio
async def test_concurrent_sessions(detector_with_model_pool):
    """Test concurrent session handling"""
    session_ids = ["session_1", "session_2", "session_3"]
    
    # Mock high speech probability for all
    detector_with_model_pool.model_pool[0].return_value = torch.tensor(0.8)
    detector_with_model_pool.model_pool[1].return_value = torch.tensor(0.8)

    # Process samples concurrently
    tasks = []
    for session_id in session_ids:
        samples = generate_samples(amplitude=1500, num_samples=8000)
        task = detector_with_model_pool.process_samples(samples, session_id)
        tasks.append(task)
    
    await asyncio.gather(*tasks)

    # All sessions should be recording
    for session_id in session_ids:
        session = detector_with_model_pool.get_session(session_id)
        assert session.is_recording is True
        assert len(session.buffer) > 0

    # Each session should have its own VAD iterator
    iterators = [detector_with_model_pool.get_session(sid).vad_iterator for sid in session_ids]
    iterator_ids = [id(it) for it in iterators]
    assert len(set(iterator_ids)) == len(session_ids), f"Expected {len(session_ids)} unique iterators, got {len(set(iterator_ids))}"


def test_reset_vad_state(detector_with_model_pool):
    """Test VAD state reset functionality"""
    session_ids = ["session_1", "session_2"]
    
    # Create sessions
    for session_id in session_ids:
        detector_with_model_pool.get_session(session_id)

    # Reset specific session
    detector_with_model_pool.reset_vad_state("session_1")
    session_1_iterator = detector_with_model_pool.get_session("session_1").vad_iterator
    session_1_iterator.reset_states.assert_called_once()

    # Reset all sessions
    detector_with_model_pool.reset_vad_state()
    for session_id in session_ids:
        iterator = detector_with_model_pool.get_session(session_id).vad_iterator
        assert iterator.reset_states.call_count >= 1


@pytest.mark.asyncio
async def test_on_recording_started_callback(mock_silero_model):
    """
    Test that on_recording_started callback is triggered when recording exceeds min_duration
    """
    callback_calls = []
    
    async def mock_callback(session_id: str):
        callback_calls.append(session_id)
    
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            on_recording_started=mock_callback,
            debug=True
        )
    
    # Mock high speech probability
    detector.model_pool[0].return_value = torch.tensor(0.8)
    
    session_id = "test_callback"
    
    # Start recording with speech samples
    speech_samples1 = generate_samples(amplitude=1200, num_samples=4000)  # 0.25 sec
    is_recording = await detector.process_samples(speech_samples1, session_id)
    assert is_recording is True
    
    # Callback should not be triggered yet (below min_duration)
    await asyncio.sleep(0.1)
    assert len(callback_calls) == 0
    
    # Add more samples to exceed min_duration
    speech_samples2 = generate_samples(amplitude=1200, num_samples=4000)  # Another 0.25 sec
    is_recording = await detector.process_samples(speech_samples2, session_id)
    assert is_recording is True
    
    # Wait for callback task to complete
    await asyncio.sleep(0.1)
    
    # Verify callback was triggered
    assert len(callback_calls) == 1
    assert callback_calls[0] == session_id
    
    # Verify flag is set
    session = detector.get_session(session_id)
    assert session.on_recording_started_triggered is True
    
    # Additional samples should not trigger callback again
    more_samples = generate_samples(amplitude=1200, num_samples=3200)
    await detector.process_samples(more_samples, session_id)
    await asyncio.sleep(0.1)
    assert len(callback_calls) == 1  # Still only one call


@pytest.mark.asyncio
async def test_on_recording_started_not_triggered_below_min_duration(mock_silero_model):
    """
    Test that on_recording_started callback is NOT triggered for short recordings
    """
    callback_calls = []
    
    async def mock_callback(session_id: str):
        callback_calls.append(session_id)
    
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            on_recording_started=mock_callback,
            debug=True
        )
    
    # Mock high speech probability initially, then low
    detector.model_pool[0].return_value = torch.tensor(0.8)
    
    session_id = "test_short_callback"
    
    # Short samples below min_duration
    short_samples = generate_samples(amplitude=1200, num_samples=7000)  # Well under 0.5 sec (0.4375 sec)
    is_recording = await detector.process_samples(short_samples, session_id)
    assert is_recording is True
    
    # Mock low speech probability for silence
    detector.model_pool[0].return_value = torch.tensor(0.2)
    
    # Stop with silence
    silent_samples = generate_samples(amplitude=0, num_samples=8000)
    is_recording = await detector.process_samples(silent_samples, session_id)
    assert is_recording is False
    
    await asyncio.sleep(0.1)
    
    # Callback should not have been triggered
    assert len(callback_calls) == 0
    
    # Flag should remain False
    session = detector.get_session(session_id)
    assert session.on_recording_started_triggered is False


@pytest.mark.asyncio
async def test_on_recording_started_callback_reset(mock_silero_model):
    """
    Test that on_recording_started_triggered flag is reset properly
    """
    callback_calls = []
    
    async def mock_callback(session_id: str):
        callback_calls.append(session_id)
    
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            on_recording_started=mock_callback,
            debug=True
        )
    
    # Mock high speech probability
    detector.model_pool[0].return_value = torch.tensor(0.8)
    
    session_id = "test_reset_callback"
    
    # First recording
    speech_samples1 = generate_samples(amplitude=1200, num_samples=4000)
    await detector.process_samples(speech_samples1, session_id)
    speech_samples2 = generate_samples(amplitude=1200, num_samples=4000)
    await detector.process_samples(speech_samples2, session_id)
    await asyncio.sleep(0.1)
    assert len(callback_calls) == 1
    
    # Reset session
    detector.reset_session(session_id)
    session = detector.get_session(session_id)
    assert session.on_recording_started_triggered is False
    
    # Second recording after reset
    await detector.process_samples(speech_samples1, session_id)
    await detector.process_samples(speech_samples2, session_id)
    await asyncio.sleep(0.1)
    assert len(callback_calls) == 2  # Callback triggered again


@pytest.mark.asyncio
async def test_process_samples_return_value(mock_silero_model):
    """
    Test that process_samples returns correct boolean value indicating recording status
    """
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            debug=True
        )
    
    session_id = "test_return_value"
    
    # Mock high speech probability for recording
    detector.model_pool[0].return_value = torch.tensor(0.8)
    
    # Start recording
    speech_samples = generate_samples(amplitude=1200, num_samples=8000)
    is_recording = await detector.process_samples(speech_samples, session_id)
    assert is_recording is True
    
    # Continue recording
    more_speech_samples = generate_samples(amplitude=1200, num_samples=3200)
    is_recording = await detector.process_samples(more_speech_samples, session_id)
    assert is_recording is True
    
    # Mock low speech probability for silence
    detector.model_pool[0].return_value = torch.tensor(0.2)
    
    # Stop with silence
    silent_samples = generate_samples(amplitude=0, num_samples=8000)
    is_recording = await detector.process_samples(silent_samples, session_id)
    assert is_recording is False


@pytest.mark.asyncio
async def test_process_samples_return_value_when_muted(mock_silero_model):
    """
    Test that process_samples returns False when detector is muted
    """
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            debug=True
        )
    
    # Mute the detector
    detector.should_mute = lambda: True
    
    session_id = "test_muted_return"
    
    # Try to record
    speech_samples = generate_samples(amplitude=1200, num_samples=8000)
    is_recording = await detector.process_samples(speech_samples, session_id)
    assert is_recording is False


@pytest.mark.asyncio
async def test_on_recording_started_callback_error_handling(mock_silero_model):
    """
    Test that errors in on_recording_started callback are handled gracefully
    """
    async def failing_callback(session_id: str):
        raise ValueError("Test error in callback")
    
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            on_recording_started=failing_callback,
            debug=True
        )
    
    # Mock high speech probability
    detector.model_pool[0].return_value = torch.tensor(0.8)
    
    session_id = "test_error_callback"
    
    # Recording should continue despite callback error
    speech_samples1 = generate_samples(amplitude=1200, num_samples=4000)
    await detector.process_samples(speech_samples1, session_id)
    speech_samples2 = generate_samples(amplitude=1200, num_samples=4000)
    is_recording = await detector.process_samples(speech_samples2, session_id)
    assert is_recording is True
    
    await asyncio.sleep(0.1)
    
    # Flag should still be set even with callback error
    session = detector.get_session(session_id)
    assert session.on_recording_started_triggered is True


@pytest.mark.asyncio
async def test_volume_db_threshold_filtering(mock_silero_model):
    """
    Test that volume_db_threshold properly filters out low volume audio
    even when speech probability is high
    """
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        # Set volume_db_threshold to -20 dB (relatively high threshold)
        detector = SileroSpeechDetector(
            volume_db_threshold=-20.0,  # Only sounds louder than -20 dB will be considered
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            debug=True
        )
    
    # Mock high speech probability (would normally trigger recording)
    detector.model_pool[0].return_value = torch.tensor(0.8)
    
    session_id = "test_volume_filter"
    
    # Calculate amplitude for -20 dB threshold
    # amplitude_threshold = 32767 * 10^(-20/20) ≈ 3277
    expected_threshold = 32767 * (10 ** (-20.0 / 20.0))
    assert abs(detector.amplitude_threshold - expected_threshold) < 1  # Allow small floating point error
    
    # Test 1: Low volume samples (below threshold) - should NOT trigger recording
    low_volume_samples = generate_samples(amplitude=1000, num_samples=8000)  # Below threshold
    is_recording = await detector.process_samples(low_volume_samples, session_id)
    assert is_recording is False, "Low volume samples should not trigger recording despite high speech probability"
    
    session = detector.get_session(session_id)
    assert session.is_recording is False
    
    # Test 2: High volume samples (above threshold) - should trigger recording
    high_volume_samples = generate_samples(amplitude=5000, num_samples=8000)  # Above threshold
    is_recording = await detector.process_samples(high_volume_samples, session_id)
    assert is_recording is True, "High volume samples should trigger recording"
    
    session = detector.get_session(session_id)
    assert session.is_recording is True
    
    # Test 3: Low volume during recording - should continue recording (no immediate stop)
    low_volume_samples2 = generate_samples(amplitude=1000, num_samples=3200)
    is_recording = await detector.process_samples(low_volume_samples2, session_id)
    assert is_recording is True, "Recording should continue even with low volume samples"
    
    # Mock low speech probability for proper stop
    detector.model_pool[0].return_value = torch.tensor(0.2)
    
    # Test 4: Silent samples to stop recording
    silent_samples = generate_samples(amplitude=0, num_samples=8000)
    is_recording = await detector.process_samples(silent_samples, session_id)
    assert is_recording is False, "Recording should stop with silence"


@pytest.mark.asyncio
async def test_volume_db_threshold_property_update(mock_silero_model):
    """
    Test that volume_db_threshold property can be updated dynamically
    and amplitude_threshold is recalculated correctly
    """
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        # Start with default threshold (0 dB)
        detector = SileroSpeechDetector(
            volume_db_threshold=0.0,
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            debug=True
        )
    
    # Check initial values
    assert detector.volume_db_threshold == 0.0
    expected_threshold_0db = 32767 * (10 ** (0.0 / 20.0))  # Should be 32767
    assert abs(detector.amplitude_threshold - expected_threshold_0db) < 1
    
    # Update to -10 dB
    detector.volume_db_threshold = -10.0
    assert detector.volume_db_threshold == -10.0
    expected_threshold_minus10db = 32767 * (10 ** (-10.0 / 20.0))  # ≈ 10362
    assert abs(detector.amplitude_threshold - expected_threshold_minus10db) < 1
    
    # Update to -30 dB
    detector.volume_db_threshold = -30.0
    assert detector.volume_db_threshold == -30.0
    expected_threshold_minus30db = 32767 * (10 ** (-30.0 / 20.0))  # ≈ 1036
    assert abs(detector.amplitude_threshold - expected_threshold_minus30db) < 1
    
    # Test with actual processing
    detector.model_pool[0].return_value = torch.tensor(0.8)  # High speech probability
    session_id = "test_dynamic_threshold"
    
    # With -30 dB threshold, even quiet sounds should trigger
    quiet_samples = generate_samples(amplitude=1500, num_samples=8000)
    is_recording = await detector.process_samples(quiet_samples, session_id)
    assert is_recording is True
    
    # Change threshold to -10 dB
    detector.volume_db_threshold = -10.0
    detector.reset_session(session_id)
    
    # Same quiet samples should NOT trigger with stricter threshold
    is_recording = await detector.process_samples(quiet_samples, session_id)
    assert is_recording is False
    
    # Louder samples should trigger
    loud_samples = generate_samples(amplitude=15000, num_samples=8000)
    is_recording = await detector.process_samples(loud_samples, session_id)
    assert is_recording is True


@pytest.mark.asyncio
async def test_volume_threshold_with_mixed_volume(mock_silero_model):
    """
    Test volume threshold behavior with mixed volume audio
    (quiet start, loud middle, quiet end)
    """
    model, utils = mock_silero_model
    
    with patch('torch.hub.load', return_value=(model, utils)):
        detector = SileroSpeechDetector(
            volume_db_threshold=-15.0,  # Moderate threshold
            silence_duration_threshold=0.5,
            max_duration=3.0,
            min_duration=0.5,
            sample_rate=16000,
            channels=1,
            preroll_buffer_count=5,
            speech_probability_threshold=0.5,
            chunk_size=512,
            debug=True
        )
    
    # Always return high speech probability
    detector.model_pool[0].return_value = torch.tensor(0.8)
    
    session_id = "test_mixed_volume"
    
    # Start with quiet samples (below threshold)
    quiet_start = generate_samples(amplitude=2000, num_samples=3200)
    is_recording = await detector.process_samples(quiet_start, session_id)
    assert is_recording is False
    
    # Loud samples (above threshold) - should start recording
    loud_middle = generate_samples(amplitude=8000, num_samples=8000)
    is_recording = await detector.process_samples(loud_middle, session_id)
    assert is_recording is True
    
    # Continue with more loud samples
    more_loud = generate_samples(amplitude=8000, num_samples=3200)
    is_recording = await detector.process_samples(more_loud, session_id)
    assert is_recording is True
    
    # Quiet end (below threshold but recording continues)
    quiet_end = generate_samples(amplitude=2000, num_samples=3200)
    is_recording = await detector.process_samples(quiet_end, session_id)
    assert is_recording is True  # Recording continues despite low volume
    
    # Verify recording is still active
    session = detector.get_session(session_id)
    assert session.is_recording is True