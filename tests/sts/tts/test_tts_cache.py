import base64
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx
from aiavatar.sts.stt.google import GoogleSpeechRecognizer
from aiavatar.sts.tts.azure import AzureSpeechSynthesizer
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.sts.tts.google import GoogleSpeechSynthesizer
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.sts.tts.base import create_instant_synthesizer

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FAKE_AUDIO = b"fake-audio-data-for-test"


# --- Mock tests (no external API) ---

# Azure

@pytest.mark.asyncio
async def test_azure_cache_created():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = AzureSpeechSynthesizer(
            azure_api_key="test-key", azure_region="eastus",
            speaker="ja-JP-MayuNeural", cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        assert len(os.listdir(cache_dir)) == 1
        await synth.close()


@pytest.mark.asyncio
async def test_azure_cache_used():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = AzureSpeechSynthesizer(
            azure_api_key="test-key", azure_region="eastus",
            speaker="ja-JP-MayuNeural", cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        await synth.synthesize("テスト")
        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        await synth.close()


# OpenAI

@pytest.mark.asyncio
async def test_openai_cache_created():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = OpenAISpeechSynthesizer(
            openai_api_key="test-key", speaker="sage",
            cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        assert len(os.listdir(cache_dir)) == 1
        await synth.close()


@pytest.mark.asyncio
async def test_openai_cache_used():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = OpenAISpeechSynthesizer(
            openai_api_key="test-key", speaker="sage",
            cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        await synth.synthesize("テスト")
        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        await synth.close()


# Google

@pytest.mark.asyncio
async def test_google_cache_created():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = GoogleSpeechSynthesizer(
            google_api_key="test-key", speaker="ja-JP-Standard-B",
            cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "audioContent": base64.b64encode(FAKE_AUDIO).decode()
        }
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        assert len(os.listdir(cache_dir)) == 1
        await synth.close()


@pytest.mark.asyncio
async def test_google_cache_used():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = GoogleSpeechSynthesizer(
            google_api_key="test-key", speaker="ja-JP-Standard-B",
            cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "audioContent": base64.b64encode(FAKE_AUDIO).decode()
        }
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        await synth.synthesize("テスト")
        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        await synth.close()


# Voicevox

@pytest.mark.asyncio
async def test_voicevox_cache_created():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = VoicevoxSpeechSynthesizer(
            speaker=46, cache_dir=cache_dir,
        )
        synth.get_audio_query = AsyncMock(return_value={"key": "value"})
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        assert len(os.listdir(cache_dir)) == 1
        await synth.close()


@pytest.mark.asyncio
async def test_voicevox_cache_used():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = VoicevoxSpeechSynthesizer(
            speaker=46, cache_dir=cache_dir,
        )
        synth.get_audio_query = AsyncMock(return_value={"key": "value"})
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        synth.http_client.post = AsyncMock(return_value=mock_resp)

        await synth.synthesize("テスト")
        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.post.call_count == 1
        # get_audio_query should only be called once (skipped on cache hit)
        assert synth.get_audio_query.call_count == 1
        await synth.close()


# InstantSynthesizer

@pytest.mark.asyncio
async def test_instant_cache_created():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = create_instant_synthesizer(
            method="POST",
            url="https://example.com/tts",
            json={"text": "{text}"},
            cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        mock_resp.raise_for_status = MagicMock()
        synth.http_client.send = AsyncMock(return_value=mock_resp)

        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.send.call_count == 1
        assert len(os.listdir(cache_dir)) == 1
        await synth.close()


@pytest.mark.asyncio
async def test_instant_cache_used():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = create_instant_synthesizer(
            method="POST",
            url="https://example.com/tts",
            json={"text": "{text}"},
            cache_dir=cache_dir,
        )
        mock_resp = MagicMock()
        mock_resp.content = FAKE_AUDIO
        mock_resp.raise_for_status = MagicMock()
        synth.http_client.send = AsyncMock(return_value=mock_resp)

        await synth.synthesize("テスト")
        result = await synth.synthesize("テスト")
        assert result == FAKE_AUDIO
        assert synth.http_client.send.call_count == 1
        await synth.close()


# No cache_dir = no caching

@pytest.mark.asyncio
async def test_no_cache_dir_no_caching():
    synth = AzureSpeechSynthesizer(
        azure_api_key="test-key", azure_region="eastus",
        speaker="ja-JP-MayuNeural",
    )
    mock_resp = MagicMock()
    mock_resp.content = FAKE_AUDIO
    synth.http_client.post = AsyncMock(return_value=mock_resp)

    await synth.synthesize("テスト")
    await synth.synthesize("テスト")
    assert synth.http_client.post.call_count == 2
    await synth.close()


# --- Integration tests (real API + cache + STT verification) ---

@pytest.mark.asyncio
async def test_azure_cache_audio_is_valid():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = AzureSpeechSynthesizer(
            azure_api_key=AZURE_API_KEY, azure_region=AZURE_REGION,
            speaker="ja-JP-MayuNeural", cache_dir=cache_dir,
        )

        # First call - API
        result1 = await synth.synthesize("これはキャッシュのテストです。")
        assert len(result1) > 0
        assert len(os.listdir(cache_dir)) == 1

        # Kill API - if second call succeeds, it must be from cache
        synth.http_client.post = AsyncMock(side_effect=Exception("API should not be called"))

        # Second call - from cache
        result2 = await synth.synthesize("これはキャッシュのテストです。")
        assert result1 == result2

        # Verify cached audio is valid via STT
        recognizer = GoogleSpeechRecognizer(
            google_api_key=GOOGLE_API_KEY, language="ja-JP"
        )
        recognized = await recognizer.transcribe(result2)
        assert "キャッシュ" in recognized or "テスト" in recognized

        await recognizer.close()
        await synth.close()


@pytest.mark.asyncio
async def test_openai_cache_audio_is_valid():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = OpenAISpeechSynthesizer(
            openai_api_key=OPENAI_API_KEY, speaker="sage",
            cache_dir=cache_dir,
        )

        result1 = await synth.synthesize("これはキャッシュのテストです。")
        assert len(result1) > 0
        assert len(os.listdir(cache_dir)) == 1

        synth.http_client.post = AsyncMock(side_effect=Exception("API should not be called"))

        result2 = await synth.synthesize("これはキャッシュのテストです。")
        assert result1 == result2

        recognizer = GoogleSpeechRecognizer(
            google_api_key=GOOGLE_API_KEY, sample_rate=24000, language="ja-JP"
        )
        recognized = await recognizer.transcribe(result2)
        assert "キャッシュ" in recognized or "テスト" in recognized

        await recognizer.close()
        await synth.close()


@pytest.mark.asyncio
async def test_google_cache_audio_is_valid():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = GoogleSpeechSynthesizer(
            google_api_key=GOOGLE_API_KEY, speaker="ja-JP-Standard-B",
            cache_dir=cache_dir,
        )

        result1 = await synth.synthesize("これはキャッシュのテストです。")
        assert len(result1) > 0
        assert len(os.listdir(cache_dir)) == 1

        synth.http_client.post = AsyncMock(side_effect=Exception("API should not be called"))

        result2 = await synth.synthesize("これはキャッシュのテストです。")
        assert result1 == result2

        recognizer = GoogleSpeechRecognizer(
            google_api_key=GOOGLE_API_KEY, sample_rate=24000, language="ja-JP"
        )
        recognized = await recognizer.transcribe(result2)
        assert "キャッシュ" in recognized or "テスト" in recognized

        await recognizer.close()
        await synth.close()


@pytest.mark.asyncio
async def test_voicevox_cache_audio_is_valid():
    with tempfile.TemporaryDirectory() as cache_dir:
        synth = VoicevoxSpeechSynthesizer(
            speaker=46, base_url="http://127.0.0.1:50021",
            cache_dir=cache_dir,
        )

        result1 = await synth.synthesize("これはキャッシュのテストです。")
        assert len(result1) > 0
        assert len(os.listdir(cache_dir)) == 1

        synth.get_audio_query = AsyncMock(side_effect=Exception("API should not be called"))
        synth.http_client.post = AsyncMock(side_effect=Exception("API should not be called"))

        result2 = await synth.synthesize("これはキャッシュのテストです。")
        assert result1 == result2

        recognizer = GoogleSpeechRecognizer(
            google_api_key=GOOGLE_API_KEY, sample_rate=24000, language="ja-JP"
        )
        recognized = await recognizer.transcribe(result2)
        assert "キャッシュ" in recognized or "テスト" in recognized

        await recognizer.close()
        await synth.close()
