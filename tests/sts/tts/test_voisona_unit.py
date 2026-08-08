from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiavatar.sts.tts.voisona import VoisonaSpeechSynthesizer


@pytest.mark.asyncio
async def test_voisona_keeps_generation_cleanup_inside_common_cache_flow(tmp_path):
    output_path = tmp_path / "voisona-output.wav"
    output_path.write_bytes(b"audio")
    cache_dir = tmp_path / "cache"

    synthesizer = VoisonaSpeechSynthesizer(
        speaker="test-voice",
        output_dir=str(tmp_path),
        cache_dir=str(cache_dir),
    )
    synthesizer.get_voice_library = AsyncMock(return_value={
        "voice_name": "test-voice",
        "voice_version": "1.0",
        "languages": ["ja_JP"],
    })
    synthesizer._make_output_path = lambda: str(output_path)
    synthesizer.http_client.post = AsyncMock(return_value=SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"uuid": "request-id"},
    ))
    synthesizer._wait_for_synthesis = AsyncMock()
    synthesizer._delete_synthesis_request = AsyncMock()

    try:
        first = await synthesizer.synthesize("hello")
        second = await synthesizer.synthesize("hello")
    finally:
        await synthesizer.close()

    assert first == second == b"audio"
    assert not output_path.exists()
    synthesizer.http_client.post.assert_awaited_once()
    synthesizer._wait_for_synthesis.assert_awaited_once_with("request-id")
    synthesizer._delete_synthesis_request.assert_awaited_once_with("request-id")
