import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import aiavatar.sts.tts.qwen3_mlx as qwen3_mlx
from aiavatar.sts.tts.qwen3_mlx import Qwen3MLXSpeechSynthesizer


class FakeModel:
    sample_rate = 24000

    def __init__(self):
        self.requests = []

    def generate_custom_voice(self, **kwargs):
        self.requests.append(kwargs)
        return iter([
            SimpleNamespace(audio=np.array([0.0, 0.5], dtype=np.float32)),
            SimpleNamespace(audio=np.array([-0.5], dtype=np.float32)),
        ])


class FakeMLX:
    def __init__(self):
        self.seeds = []
        self.random = SimpleNamespace(seed=self.seeds.append)


@pytest.mark.asyncio
async def test_qwen3_mlx_returns_pcm_wav_and_maps_request_language(monkeypatch):
    model = FakeModel()
    mlx = FakeMLX()
    loads = []
    monkeypatch.setattr(
        qwen3_mlx,
        "_load_runtime",
        lambda model_id: loads.append(model_id) or (model, mlx),
    )
    synthesizer = Qwen3MLXSpeechSynthesizer(
        model="test-model",
        voice="Vivian",
        instruct="Speak evenly.",
        max_tokens=321,
        seed=11,
        style_mapper={"<face name=\"joy\"": "Ryan"},
    )

    try:
        audio = await synthesizer.synthesize(
            "こんにちは",
            style_info={"styled_text": '<face name="joy" />こんにちは'},
            language="ja-JP",
        )
        await synthesizer.synthesize("Hello", language="en-US")
    finally:
        await synthesizer.close()

    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 3

    assert loads == ["test-model"]
    assert mlx.seeds == [11, 11]
    assert model.requests == [
        {
            "text": "こんにちは",
            "speaker": "Ryan",
            "language": "Japanese",
            "instruct": "Speak evenly.",
            "max_tokens": 321,
            "verbose": False,
        },
        {
            "text": "Hello",
            "speaker": "Vivian",
            "language": "English",
            "instruct": "Speak evenly.",
            "max_tokens": 321,
            "verbose": False,
        },
    ]


@pytest.mark.asyncio
async def test_qwen3_mlx_cache_distinguishes_voice(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setattr(
        qwen3_mlx,
        "_load_runtime",
        lambda model_id: (model, FakeMLX()),
    )
    synthesizer = Qwen3MLXSpeechSynthesizer(
        model="test-model",
        voice="Vivian",
        cache_dir=str(tmp_path),
    )

    try:
        await synthesizer.synthesize("same text")
        synthesizer.voice = "Ryan"
        await synthesizer.synthesize("same text")
    finally:
        await synthesizer.close()

    assert [request["speaker"] for request in model.requests] == ["Vivian", "Ryan"]
    assert len(list(tmp_path.iterdir())) == 2
