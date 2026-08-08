import io
import struct
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer


def make_wav(sample_rate: int = 24000, frame_count: int = 240) -> bytes:
    frames = b"".join(
        struct.pack("<h", ((frame_index % 40) - 20) * 1000)
        for frame_index in range(frame_count)
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return output.getvalue()


def read_wav_params(audio: bytes) -> tuple[int, int, int, int]:
    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        return (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getnframes(),
        )


@pytest.mark.asyncio
async def test_sample_rate_resamples_openai_wav_response():
    source_audio = make_wav()
    synthesizer = OpenAISpeechSynthesizer(
        openai_api_key="test-key",
        audio_format="wav",
        sample_rate=16000,
    )
    synthesizer.http_client.post = AsyncMock(
        return_value=SimpleNamespace(content=source_audio)
    )

    try:
        result = await synthesizer.synthesize("test")
    finally:
        await synthesizer.close()

    sample_rate, channels, sample_width, frame_count = read_wav_params(result)
    assert sample_rate == 16000
    assert channels == 1
    assert sample_width == 2
    assert abs(frame_count / sample_rate - 240 / 24000) <= 1 / sample_rate
    assert synthesizer.get_config()["sample_rate"] == 16000
    assert synthesizer.http_client.post.call_args.kwargs["json"]["response_format"] == "wav"


@pytest.mark.asyncio
async def test_sample_rate_is_ignored_for_non_wav_audio():
    source_audio = b"fake-mp3-audio"
    synthesizer = OpenAISpeechSynthesizer(
        openai_api_key="test-key",
        audio_format="mp3",
        sample_rate=16000,
    )
    synthesizer.http_client.post = AsyncMock(
        return_value=SimpleNamespace(content=source_audio)
    )

    try:
        result = await synthesizer.synthesize("test")
    finally:
        await synthesizer.close()

    assert result == source_audio
    assert synthesizer.http_client.post.call_args.kwargs["json"]["response_format"] == "mp3"


@pytest.mark.asyncio
async def test_sample_rate_is_part_of_cache_key(tmp_path):
    source_audio = make_wav()
    synthesizers = [
        OpenAISpeechSynthesizer(
            openai_api_key="test-key",
            sample_rate=sample_rate,
            cache_dir=str(tmp_path),
        )
        for sample_rate in (16000, 8000)
    ]
    for synthesizer in synthesizers:
        synthesizer.http_client.post = AsyncMock(
            return_value=SimpleNamespace(content=source_audio)
        )

    try:
        results = [
            await synthesizer.synthesize("same text")
            for synthesizer in synthesizers
        ]
    finally:
        for synthesizer in synthesizers:
            await synthesizer.close()

    assert [read_wav_params(result)[0] for result in results] == [16000, 8000]
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.asyncio
async def test_sample_rate_must_be_positive_before_request():
    synthesizer = OpenAISpeechSynthesizer(
        openai_api_key="test-key",
        sample_rate=0,
    )
    synthesizer.http_client.post = AsyncMock()

    try:
        with pytest.raises(ValueError, match="positive integer"):
            await synthesizer.synthesize("test")
    finally:
        await synthesizer.close()

    synthesizer.http_client.post.assert_not_awaited()
