import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import aiavatar.sts.tts.irodori_mlx as irodori_mlx
from aiavatar.sts.tts.irodori_mlx import IrodoriMLXSpeechSynthesizer


class FakeModel:
    def __init__(self):
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        return iter([
            SimpleNamespace(
                audio=np.array([0.0, 0.5, -0.5], dtype=np.float32),
                sample_rate=48000,
            )
        ])


@pytest.mark.asyncio
async def test_irodori_mlx_returns_pcm_wav_and_maps_style_to_instruction(monkeypatch):
    model = FakeModel()
    loads = []
    monkeypatch.setattr(
        irodori_mlx,
        "_load_model",
        lambda model_id: loads.append(model_id) or model,
    )
    synthesizer = IrodoriMLXSpeechSynthesizer(
        model="test-model",
        instruct="通常の話し方",
        seed=11,
        num_steps=8,
        t_schedule_mode="sway",
        sway_coeff=-0.5,
        duration_scale=0.9,
        max_seconds=12.0,
        style_mapper={"<face name=\"joy\"": "明るく弾むような声"},
    )

    try:
        audio = await synthesizer.synthesize(
            "こんにちは",
            style_info={"styled_text": '<face name="joy" />こんにちは'},
            language="ja-JP",
        )
    finally:
        await synthesizer.close()

    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        assert wav_file.getframerate() == 48000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 3

    assert loads == ["test-model"]
    assert model.requests == [{
        "text": "こんにちは",
        "instruct": "明るく弾むような声",
        "rng_seed": 11,
        "num_steps": 8,
        "t_schedule_mode": "sway",
        "sway_coeff": -0.5,
        "duration_scale": 0.9,
        "max_seconds": 12.0,
    }]


@pytest.mark.asyncio
async def test_irodori_mlx_cache_distinguishes_instructions(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setattr(irodori_mlx, "_load_model", lambda model_id: model)
    synthesizer = IrodoriMLXSpeechSynthesizer(
        model="test-model",
        instruct="calm",
        cache_dir=str(tmp_path),
    )

    try:
        await synthesizer.synthesize("same text")
        synthesizer.instruct = "excited"
        await synthesizer.synthesize("same text")
    finally:
        await synthesizer.close()

    assert [request["instruct"] for request in model.requests] == ["calm", "excited"]
    assert len(list(tmp_path.iterdir())) == 2
