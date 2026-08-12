import pytest

import aiavatar.cli.config as cli_config


def test_llm_extra_body_requires_json_object(
    monkeypatch,
    clean_builtin_environment,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AIAVATAR_JA_TTS", "voicevox")
    monkeypatch.setenv("AIAVATAR_LLM_EXTRA_BODY", "not-json")

    with pytest.raises(
        RuntimeError,
        match="AIAVATAR_LLM_EXTRA_BODY must be a valid JSON object",
    ):
        cli_config.AppConfig.from_env()


def test_component_openai_config_overrides_common_values(
    monkeypatch,
    clean_builtin_environment,
):
    monkeypatch.setenv("OPENAI_API_KEY", "common-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://common.example/v1")
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_BASE_URL", "https://llm.example/v1")

    stt = cli_config.OpenAIConfig.from_env("stt")
    llm = cli_config.OpenAIConfig.from_env("llm")
    assert (stt.api_key, stt.base_url) == (
        "common-key",
        "https://common.example/v1",
    )
    assert (llm.api_key, llm.base_url) == (
        "llm-key",
        "https://llm.example/v1",
    )


def test_reasoning_effort_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("AIAVATAR_LLM_REASONING_EFFORT", raising=False)
    assert cli_config.resolve_reasoning_effort(None) == "none"
    assert cli_config.resolve_reasoning_effort({"thinking": {}}) is None

    monkeypatch.setenv("AIAVATAR_LLM_REASONING_EFFORT", "")
    assert cli_config.resolve_reasoning_effort(None) == "none"

    monkeypatch.setenv("AIAVATAR_LLM_REASONING_EFFORT", "medium")
    assert cli_config.resolve_reasoning_effort({"thinking": {}}) == "medium"

    monkeypatch.setenv("AIAVATAR_LLM_REASONING_EFFORT", "omit")
    assert cli_config.resolve_reasoning_effort(None) is None


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("AIAVATAR_DEBUG", "sometimes", "must be true or false"),
        (
            "AIAVATAR_VAD_SILENCE_DURATION_THRESHOLD",
            "quickly",
            "must be a number",
        ),
        (
            "AIAVATAR_FILLER_PHRASES",
            '["okay", 1]',
            "must be a JSON array of strings",
        ),
    ],
)
def test_app_config_rejects_invalid_typed_environment(
    monkeypatch,
    clean_builtin_environment,
    name,
    value,
    message,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        cli_config.AppConfig.from_env()
