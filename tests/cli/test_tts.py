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


def test_qwen3_mlx_routes_share_one_model_instance(
    monkeypatch,
    clean_builtin_environment,
    component_fakes,
):
    monkeypatch.setenv("AIAVATAR_JA_TTS", "qwen3-mlx")
    monkeypatch.setenv("AIAVATAR_MULTI_TTS", "qwen3-mlx")
    monkeypatch.setenv(
        "AIAVATAR_JA_TTS_CONFIG",
        '{"voice":"Vivian","instruct":"Speak evenly."}',
    )
    monkeypatch.setenv(
        "AIAVATAR_MULTI_TTS_CONFIG",
        '{"voice":"Vivian","instruct":"Speak evenly."}',
    )

    router, preprocessor = cli_tts.build_default_tts(
        cli_config.AppConfig.from_env()
    )

    japanese = router.kwargs["synthesizers"]["ja"]
    multilingual = router.kwargs["synthesizers"]["multi"]
    assert japanese is multilingual
    assert japanese.kwargs["voice"] == "Vivian"
    assert japanese.kwargs["instruct"] == "Speak evenly."
    assert japanese.kwargs["language"] == "Auto"
    assert japanese.kwargs["preprocessors"] == []
    assert preprocessor is None
