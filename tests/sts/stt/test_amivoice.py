import pytest
import os
import wave
from pathlib import Path

from aiavatar.sts.stt.amivoice import AmiVoiceSpeechRecognizer

AMIVOICE_API_KEY = os.getenv("AMIVOICE_API_KEY")


@pytest.fixture
def stt_wav_path() -> Path:
    """
    Returns the path to the hello.wav file containing "こんにちは。"
    Make sure the file is placed at tests/data/hello.wav (or an appropriate path).
    """
    return Path(__file__).parent / "data" / "hello.wav"


@pytest.fixture
def stt_wav_path_en() -> Path:
    """
    Returns the path to the hello.wav file containing "hello"
    Make sure the file is placed at tests/data/hello.wav (or an appropriate path).
    """
    return Path(__file__).parent / "data" / "hello_en.wav"


@pytest.mark.asyncio
async def test_amivoice_speech_recognizer_transcribe(stt_wav_path):
    """
    Test to verify that AmiVoiceSpeechRecognizer can transcribe the hello.wav file
    which contains "こんにちは。".
    NOTE: This test actually calls AmiVoice's Speech-to-Text API and consumes credits.
    """
    # 1) Load the WAV file
    with wave.open(str(stt_wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer with Japanese general engine
    recognizer = AmiVoiceSpeechRecognizer(
        amivoice_api_key=AMIVOICE_API_KEY,
        engine="-a2-ja-general",
        sample_rate=sample_rate,
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text
    assert recognized_text is not None, "Recognition should return a result"
    assert "こんにちは" in recognized_text, f"Expected 'こんにちは', got: {recognized_text}"

    # 5) Close the recognizer's http_client
    await recognizer.close()


@pytest.mark.asyncio
async def test_amivoice_speech_recognizer_transcribe_with_downsampling(stt_wav_path):
    """
    Test to verify that AmiVoiceSpeechRecognizer can transcribe with downsampling.
    NOTE: This test actually calls AmiVoice's Speech-to-Text API and consumes credits.
    """
    # 1) Load the WAV file
    with wave.open(str(stt_wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer with downsampling to 8kHz
    recognizer = AmiVoiceSpeechRecognizer(
        amivoice_api_key=AMIVOICE_API_KEY,
        engine="-a2-ja-general",
        sample_rate=sample_rate,
        target_sample_rate=8000,  # Downsample to 8kHz
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text
    assert recognized_text is not None, "Recognition should return a result"
    assert "こんにちは" in recognized_text, f"Expected 'こんにちは', got: {recognized_text}"

    # 5) Close the recognizer's http_client
    await recognizer.close()


@pytest.mark.asyncio
async def test_amivoice_speech_recognizer_transcribe_english(stt_wav_path_en):
    """
    Test to verify that AmiVoiceSpeechRecognizer can transcribe English audio.
    NOTE: This test actually calls AmiVoice's Speech-to-Text API and consumes credits.
    """
    # 1) Load the English WAV file
    with wave.open(str(stt_wav_path_en), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer with English engine
    recognizer = AmiVoiceSpeechRecognizer(
        amivoice_api_key=AMIVOICE_API_KEY,
        engine="-a2-multi-general",  # Multi language engine
        sample_rate=sample_rate,
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text
    assert recognized_text is not None, "Recognition should return a result"
    assert "hello" in recognized_text.lower(), f"Expected 'hello', got: {recognized_text}"

    # 5) Close the recognizer's http_client
    await recognizer.close()


@pytest.mark.asyncio
async def test_amivoice_speech_recognizer_transcribe_business_engine(stt_wav_path):
    """
    Test to verify that AmiVoiceSpeechRecognizer can transcribe with business engine.
    NOTE: This test actually calls AmiVoice's Speech-to-Text API and consumes credits.
    """
    # 1) Load the WAV file
    with wave.open(str(stt_wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer with business engine
    recognizer = AmiVoiceSpeechRecognizer(
        amivoice_api_key=AMIVOICE_API_KEY,
        engine="-a-bizfinance",  # Business/Finance specialized engine
        sample_rate=sample_rate,
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text
    assert recognized_text is not None, "Recognition should return a result"
    assert "こんにちは" in recognized_text, f"Expected 'こんにちは', got: {recognized_text}"

    # 5) Close the recognizer's http_client
    await recognizer.close()


@pytest.mark.asyncio
async def test_amivoice_speech_recognizer_error_handling():
    """
    Test error handling with invalid API key.
    """
    # 1) Create recognizer with invalid API key
    recognizer = AmiVoiceSpeechRecognizer(
        amivoice_api_key="invalid_key",
        engine="-a2-ja-general",
        sample_rate=16000,
        debug=True
    )

    # 2) Create dummy audio data
    dummy_audio = b"\x00" * 1024

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(dummy_audio)

    # 4) Should return None for invalid API key
    assert recognized_text is "", "Should return empty string for invalid API key"

    # 5) Close the recognizer's http_client
    await recognizer.close()