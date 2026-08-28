from types import SimpleNamespace

import pytest

from aiavatar.sts.stt.openai import OpenAISpeechRecognizer


@pytest.mark.asyncio
async def test_openai_stt_uses_configured_base_url():
    recognizer = OpenAISpeechRecognizer(
        openai_api_key="test-key",
        base_url="https://stt.example/v1/",
        min_data_length=1,
    )
    observed = {}

    async def fake_request(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(json=lambda: {"text": "hello"})

    recognizer.http_request_with_retry = fake_request
    try:
        assert await recognizer.transcribe(b"\0\0") == "hello"
    finally:
        await recognizer.close()

    assert observed["url"] == "https://stt.example/v1/audio/transcriptions"
    assert observed["headers"] == {"Authorization": "Bearer test-key"}
    assert observed["data"]["model"] == "gpt-transcribe"
