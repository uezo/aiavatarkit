from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiavatar.sts.tts.speech_gateway import SpeechGatewaySpeechSynthesizer


@pytest.mark.asyncio
async def test_speech_gateway_uses_common_cache_flow(tmp_path):
    synthesizer = SpeechGatewaySpeechSynthesizer(
        service_name="test-service",
        speaker="test-speaker",
        speed=1.1,
        audio_format="wav",
        cache_dir=str(tmp_path),
    )
    synthesizer.http_client.post = AsyncMock(
        return_value=SimpleNamespace(content=b"audio")
    )

    try:
        first = await synthesizer.synthesize("hello")
        second = await synthesizer.synthesize("hello")
    finally:
        await synthesizer.close()

    assert first == second == b"audio"
    synthesizer.http_client.post.assert_awaited_once_with(
        url="http://127.0.0.1:8000/tts",
        json={
            "text": "hello",
            "service_name": "test-service",
            "speaker": "test-speaker",
            "speed": 1.1,
            "audio_format": "wav",
        },
    )


@pytest.mark.asyncio
async def test_speech_gateway_language_routing_does_not_require_default_service():
    synthesizer = SpeechGatewaySpeechSynthesizer()
    synthesizer.http_client.post = AsyncMock(
        return_value=SimpleNamespace(content=b"audio")
    )

    try:
        result = await synthesizer.synthesize("hello", language="en-US")
    finally:
        await synthesizer.close()

    assert result == b"audio"
    assert synthesizer.http_client.post.call_args.kwargs["json"] == {
        "text": "hello",
        "language": "en-US",
    }
