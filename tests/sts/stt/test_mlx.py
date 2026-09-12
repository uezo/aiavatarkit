import numpy as np
import pytest

import aiavatar.sts.stt.mlx as mlx_stt
from aiavatar.sts.stt.mlx import MLXSpeechRecognizer


@pytest.mark.asyncio
async def test_mlx_recognizer_converts_pcm_and_maps_language(monkeypatch):
    observed = []

    def fake_transcribe(audio, **kwargs):
        observed.append((audio.copy(), kwargs))
        return {"text": " こんにちは。 "}

    monkeypatch.setattr(mlx_stt, "_transcribe", fake_transcribe)
    recognizer = MLXSpeechRecognizer(
        model="test-model",
        min_data_length=2,
        language="ja-JP",
        initial_prompt="固有名詞",
    )

    try:
        result = await recognizer.transcribe(
            np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        )
    finally:
        await recognizer.close()

    assert result == "こんにちは。"
    audio, kwargs = observed[0]
    np.testing.assert_allclose(audio, [-1.0, 0.0, 32767 / 32768])
    assert audio.dtype == np.float32
    assert kwargs == {
        "path_or_hf_repo": "test-model",
        "language": "ja",
        "initial_prompt": "固有名詞",
        "task": "transcribe",
        "verbose": None,
    }


@pytest.mark.asyncio
async def test_mlx_recognizer_skips_short_audio(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mlx_stt,
        "_transcribe",
        lambda audio, **kwargs: calls.append((audio, kwargs)),
    )
    recognizer = MLXSpeechRecognizer(min_data_length=4)

    try:
        assert await recognizer.transcribe(b"\0\0") is None
    finally:
        await recognizer.close()

    assert calls == []
