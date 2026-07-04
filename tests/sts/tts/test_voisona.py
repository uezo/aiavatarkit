import configparser
import os
import tempfile
from pathlib import Path

import pytest

from aiavatar.sts.tts.voisona import VoisonaSpeechSynthesizer


def get_pytest_env(name: str, default: str = None) -> str:
    if value := os.getenv(name):
        return value

    config = configparser.ConfigParser()
    config.read(Path(__file__).parents[3] / "pytest.ini")
    for line in config.get("pytest", "env", fallback="").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip()
    return default


VOISONA_URL = get_pytest_env("VOISONA_URL", "http://127.0.0.1:32766/api/talk/v1")
VOISONA_USERNAME = get_pytest_env("VOISONA_USERNAME")
VOISONA_PASSWORD = get_pytest_env("VOISONA_PASSWORD")
VOISONA_SPEAKER = get_pytest_env("VOISONA_SPEAKER")


@pytest.mark.asyncio
async def test_voisona_synthesize_with_local_server_and_cache():
    with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as output_dir:
        synth = VoisonaSpeechSynthesizer(
            base_url=VOISONA_URL,
            username=VOISONA_USERNAME,
            password=VOISONA_PASSWORD,
            speaker=VOISONA_SPEAKER,
            cache_dir=cache_dir,
            output_dir=output_dir,
            timeout=30.0,
        )

        voice_library = await synth.get_voice_library(VOISONA_SPEAKER)
        assert voice_library["voice_name"]
        assert voice_library["voice_version"]

        result1 = await synth.synthesize("これはVoiSona Talkの実機テストです。")
        assert result1.startswith(b"RIFF")
        assert len(result1) > 44
        assert len(os.listdir(cache_dir)) == 1
        assert os.listdir(output_dir) == []

        synth.http_client.post = None
        result2 = await synth.synthesize("これはVoiSona Talkの実機テストです。")
        assert result2 == result1

        await synth.close()
