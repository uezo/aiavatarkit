import asyncio
import pytest
from time import time
from aiavatar.speech import VoicevoxSpeechController

@pytest.fixture
def voicevox_controller():
    return VoicevoxSpeechController(base_url="http://127.0.0.1:50021", speaker_id=46)

@pytest.mark.asyncio
async def test_prefetch_download_task_started(voicevox_controller):
    text = "こんにちは"
    voice = voicevox_controller.prefetch(text)
    assert voice.text == text
    assert voice.download_task is not None
    assert voice.audio_clip is None

@pytest.mark.asyncio
async def test_prefetch_download_completed(voicevox_controller):
    text = "こんにちは"
    voice = voicevox_controller.prefetch(text)
    await voice.download_task
    assert voice.audio_clip is not None

@pytest.mark.asyncio
async def test_speak_audio_played(voicevox_controller):
    text = "こんにちは。この音声は、テストのために再生されています。"
    start_time = time()
    await voicevox_controller.speak(text)
    assert time() - start_time > 1

@pytest.mark.asyncio
async def test_is_speaking(voicevox_controller):
    text = "こんにちは。この音声は、テストのために再生されています。"
    voice = voicevox_controller.prefetch(text)
    await voice.download_task

    assert voicevox_controller.is_speaking() is False

    speech_task = asyncio.create_task(voicevox_controller.speak(text))
    await asyncio.sleep(0.1)    # wait for starting speech

    while not speech_task.done():
        assert voicevox_controller.is_speaking() is True
        await asyncio.sleep(0.1)
    
    await speech_task

    assert voicevox_controller.is_speaking() is False
