import pytest

import aiavatar.cli.config as cli_config
import aiavatar.cli.tts as cli_tts


def test_instant_tts_requires_method_and_url(
    monkeypatch,
    clean_builtin_environment,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AIAVATAR_JA_TTS", "instant")
    monkeypatch.setenv("AIAVATAR_JA_TTS_CONFIG", '{}')

    with pytest.raises(
        RuntimeError,
        match="AIAVATAR_JA_TTS_CONFIG.method is required for instant",
    ):
        cli_tts.build_default_tts(cli_config.AppConfig.from_env())


def test_create_instant_tts(monkeypatch):
    observed = {}
    alphabet_to_kana = object()

    monkeypatch.setattr(
        cli_tts,
        "create_instant_synthesizer",
        lambda **kwargs: observed.update(kwargs) or object(),
    )

    cli_tts.create_tts(
        "instant",
        {
            "method": "POST",
            "url": "https://aivis.example/tts",
            "headers": {"Authorization": "Bearer aivis-key"},
            "json": {
                "model_uuid": "model-uuid",
                "text": "{text}",
                "output_format": "wav",
            },
            "cache_dir": None,
        },
        openai_api_key=None,
        preprocessors=[alphabet_to_kana],
        debug=False,
    )

    assert observed["url"] == "https://aivis.example/tts"
    assert observed["headers"]["Authorization"] == "Bearer aivis-key"
    assert observed["json"] == {
        "model_uuid": "model-uuid",
        "text": "{text}",
        "output_format": "wav",
    }
    assert observed["preprocessors"] == [alphabet_to_kana]
    assert observed["cache_dir"] is None
    assert observed["debug"] is False
