import os
import pytest
from aiavatar.sts.stt.google import GoogleSpeechRecognizer
from aiavatar.sts.tts.speech_gateway import SpeechGatewaySpeechSynthesizer

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@pytest.mark.asyncio
async def test_speech_gateway_synthesizer_with_google_stt():
    """
    Test the SpeechGatewaySpeechSynthesizer by actually calling a TTS server 
    and verifying the synthesized audio with Google STT.
    This test requires:
      - TTS server running at http://127.0.0.1:8000/tts
      - Valid GOOGLE_API_KEY environment variable
    """

    # 1) Create synthesizer instance
    synthesizer = SpeechGatewaySpeechSynthesizer(
        service_name="sbv2",
        speaker="0-0",
        tts_url="http://127.0.0.1:8000/tts",
        audio_format="wav",
        debug=True
    )

    # 2) The text to synthesize
    input_text = "これはテストです。"

    # 3) Call TTS
    tts_data = await synthesizer.synthesize(input_text)
    assert len(tts_data) > 0, "Synthesized audio data is empty."

    # 4) Recognize synthesized speech via GoogleSpeechRecognizer
    recognizer = GoogleSpeechRecognizer(
        google_api_key=GOOGLE_API_KEY,
        sample_rate=44100,  # Sampling rate of TTS service
        language="ja-JP"
    )

    recognized_text = await recognizer.transcribe(tts_data)

    # 5) Verify recognized text
    assert "テスト" in recognized_text, (
        f"Expected 'テスト' in recognized result, but got: {recognized_text}"
    )

    # 6) Cleanup
    await recognizer.close()
    await synthesizer.close()


@pytest.mark.asyncio
async def test_speech_gateway_synthesizer_with_google_stt_english():
    """
    Test the SpeechGatewaySpeechSynthesizer by actually calling a TTS server 
    and verifying the synthesized audio with Google STT.
    This test requires:
      - TTS server running at http://127.0.0.1:8000/tts
      - English routing is already configured on TTS server
      - Valid GOOGLE_API_KEY environment variable
    """

    # 1) Create synthesizer instance
    synthesizer = SpeechGatewaySpeechSynthesizer(
        service_name="sbv2",
        speaker="0-0",
        tts_url="http://127.0.0.1:8000/tts",
        audio_format="wav",
        debug=True
    )

    # 2) The text to synthesize
    input_text = "This is a test for speech synthesizer."

    # 3) Call TTS
    tts_data = await synthesizer.synthesize(input_text, language="en-US")
    assert len(tts_data) > 0, "Synthesized audio data is empty."

    # 4) Recognize synthesized speech via GoogleSpeechRecognizer
    recognizer = GoogleSpeechRecognizer(
        google_api_key=GOOGLE_API_KEY,
        sample_rate=24000,  # Sampling rate of TTS service
        language="en-US"
    )

    recognized_text = await recognizer.transcribe(tts_data)

    # 5) Verify recognized text
    assert "test" in recognized_text, (
        f"Expected 'test' in recognized result, but got: {recognized_text}"
    )
    assert "speech" in recognized_text, (
        f"Expected 'speech' in recognized result, but got: {recognized_text}"
    )

    # 6) Cleanup
    await recognizer.close()
    await synthesizer.close()


@pytest.mark.asyncio
async def test_speech_gateway_synthesizer_empty_text():
    """
    If empty text is provided, speech_gateway should return empty bytes 
    (no synthesis performed).
    """
    synthesizer = SpeechGatewaySpeechSynthesizer(
        service_name="sbv2",
        speaker="0-0",
        tts_url="http://127.0.0.1:8000/tts"
    )

    tts_data = await synthesizer.synthesize("    ")  # Empty or just whitespace
    assert len(tts_data) == 0, "Expected empty bytes for empty text."

    await synthesizer.close()
