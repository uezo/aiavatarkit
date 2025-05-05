import pytest
import os
import wave
from pathlib import Path

from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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
async def test_openai_speech_recognizer_transcribe(stt_wav_path):
    """
    Test to verify that OpenAISpeechRecognizer can transcribe the hello.wav file
    which contains "こんにちは。".
    NOTE: This test actually calls OpenAI's Speech-to-Text API and consumes credits.
    """
    # 1) Load the WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 2) Prepare the recognizer
    recognizer = OpenAISpeechRecognizer(
        openai_api_key=OPENAI_API_KEY,
        sample_rate=sample_rate,
        language="ja",
        debug=True
    )

    # 3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 4) Check the recognized text (Whisper-1 doesn't recognize 'こんにちは' correctly...)
    assert "こんにちわ" in recognized_text, f"Expected 'こんにちわ', got: {recognized_text}"

    # 5) Close the recognizer's http_client
    await recognizer.close()


@pytest.mark.asyncio
async def test_openai_speech_recognizer_transcribe_autodetect(stt_wav_path, stt_wav_path_en):
    """
    Test to verify that OpenAISpeechRecognizer can transcribe the hello.wav file
    which contains "こんにちは。".
    NOTE: This test actually calls OpenAI's Speech-to-Text API and consumes credits.
    """
    # 1-1) Load the WAV file
    with wave.open(str(stt_wav_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        wave_data = wav_file.readframes(n_frames)

    # 1-2) Prepare the recognizer
    recognizer = OpenAISpeechRecognizer(
        openai_api_key=OPENAI_API_KEY,
        sample_rate=sample_rate,
        language="ja-JP",
        alternative_languages=["en-US"],
        debug=True
    )

    # 1-3) Invoke the transcribe method
    recognized_text = await recognizer.transcribe(wave_data)

    # 1-4) Check the recognized text (Whisper-1 doesn't recognize 'こんにちは' correctly...)
    assert "こんにちわ" in recognized_text, f"Expected 'こんにちわ', got: {recognized_text}"

    # 2-1) Load the WAV file
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

    # Close the recognizer's http_client
    await recognizer.close()
