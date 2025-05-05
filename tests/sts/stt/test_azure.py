import pytest
import os
import wave
from pathlib import Path

from aiavatar.sts.stt.azure import AzureSpeechRecognizer

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")

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
async def test_azure_speech_recognizer_transcribe(stt_wav_path):
    """
    Test to verify that AzureSpeechRecognizer can transcribe the hello.wav file
    which contains "こんにちは。".
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    # 1) Load the WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer
    recognizer = AzureSpeechRecognizer(
        azure_api_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        sample_rate=sample_rate,
        language="ja-JP",
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text
    assert "こんにちは" in recognized_text, f"Expected 'こんにちは', got: {recognized_text}"

    # 5) Invoke the transcribe_classic method
    recognized_text_classic = await recognizer.transcribe_classic(wave_data)

    # 6) Check the recognized text
    assert "こんにちは" in recognized_text_classic, f"Expected 'こんにちは', got: {recognized_text_classic}"

    # 7) Close the recognizer's http_client
    await recognizer.close()


@pytest.mark.asyncio
async def test_azure_speech_recognizer_transcribe_autodetect(stt_wav_path, stt_wav_path_en):
    """
    Test to verify that AzureSpeechRecognizer can transcribe the hello.wav file
    which contains "こんにちは。".
    NOTE: This test actually calls Azure's Speech-to-Text API and consumes credits.
    """
    # 1-1) Load the WAV files
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 1-2) Prepare the recognizer
    recognizer = AzureSpeechRecognizer(
        azure_api_key=AZURE_API_KEY,
        azure_region=AZURE_REGION,
        sample_rate=sample_rate,
        language="ja-JP",
        alternative_languages=["en-US"],
        debug=True
    )

    # 1-3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 1-4) Check the recognized text
    assert "こんにちは" in recognized_text, f"Expected 'こんにちは', got: {recognized_text}"

    # 1-5) Invoke the transcribe_classic method
    recognized_text_classic = await recognizer.transcribe_classic(wave_data)

    # 1-6) Check the recognized text
    assert "こんにちは" in recognized_text_classic, f"Expected 'こんにちは', got: {recognized_text_classic}"

    # 2-1) Load the English WAV files
    with wave.open(str(stt_wav_path_en), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data_en = wav_file.readframes(n_frames)

    # 2-2) Prepare the recognizer
    recognizer.sample_rate = sample_rate

    # 2-3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data_en)

    # 2-4) Check the recognized text
    assert "hello" in recognized_text.lower(), f"Expected 'hello', got: {recognized_text}"

    # 2-5) Invoke the transcribe_classic method
    recognized_text_classic = await recognizer.transcribe_classic(wave_data_en)

    # 2-6) Check the recognized text
    assert "hello" in recognized_text_classic.lower(), f"Expected 'hello', got: {recognized_text_classic}"

    # Close the recognizer's http_client
    await recognizer.close()
