from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import aiavatar.cli.builtin as builtin
import aiavatar.cli.components as cli_components
import aiavatar.cli.config as cli_config


def test_builtin_app_serves_example_at_root_without_hiding_admin(
    monkeypatch,
    tmp_path,
    clean_builtin_environment,
    component_fakes,
    builtin_fakes,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "index.html").write_text(
        '<link rel="stylesheet" href="style.css">AIAvatar UI',
        encoding="utf-8",
    )
    (tmp_path / "style.css").write_text("body {}", encoding="utf-8")
    monkeypatch.setattr(builtin, "download_example", lambda _: tmp_path)

    def setup_admin(app, **_):
        app.get("/admin/")(lambda: {"admin": True})

    monkeypatch.setattr(builtin, "setup_admin_panel", setup_admin)
    app = builtin.create_app()

    with TestClient(app) as client:
        assert client.get("/").text.endswith("AIAvatar UI")
        assert client.get("/style.css").text == "body {}"
        assert client.get("/admin/").json() == {"admin": True}


@pytest.mark.asyncio
async def test_builtin_app_assembles_pipeline_and_forwards_partial_text(
    monkeypatch,
    clean_builtin_environment,
    component_fakes,
    builtin_fakes,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")
    monkeypatch.setenv("AIAVATAR_STT_OPENAI_API_KEY", "stt-key")
    monkeypatch.setenv("AIAVATAR_STT_OPENAI_BASE_URL", "https://stt.example/v1")
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("AIAVATAR_TTS_OPENAI_API_KEY", "tts-key")
    monkeypatch.setenv("AIAVATAR_TTS_OPENAI_BASE_URL", "https://tts.example/v1")
    monkeypatch.setenv("AIAVATAR_LLM_MODEL", "test-model")
    monkeypatch.setenv("AIAVATAR_LLM_SYSTEM_PROMPT", "test system prompt")
    monkeypatch.setenv("AIAVATAR_LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("AIAVATAR_LLM_VOICE_TEXT_TAGS", '["first", "second"]')
    monkeypatch.setenv(
        "AIAVATAR_LLM_EXTRA_BODY",
        '{"thinking":{"type":"disabled"}}',
    )
    monkeypatch.setenv("AIAVATAR_STT_MODEL", "test-transcribe")
    monkeypatch.setenv("AIAVATAR_STT_LANGUAGE", "en")
    monkeypatch.setenv("AIAVATAR_NEAR_FIELD_INITIAL_AMBIENT_DB", "-24.5")
    monkeypatch.setenv("AIAVATAR_FILLER_PHRASES", '["well", "okay"]')
    monkeypatch.setenv("FILLER_TURN_GATE_TIMEOUT", "4.5")
    monkeypatch.setenv("AIAVATAR_NAMO_LANGUAGE", "en")
    monkeypatch.setenv("NAMO_TURN_THRESHOLD", "0.7")
    monkeypatch.setenv("NAMO_TURN_GATE_TIMEOUT", "1.5")
    monkeypatch.setenv(
        "AIAVATAR_NAMO_FORCE_END_PHRASES",
        '["goodbye", "stop"]',
    )
    monkeypatch.setenv("AIAVATAR_VAD_SILENCE_DURATION_THRESHOLD", "0.8")
    monkeypatch.setenv("AIAVATAR_VAD_SEGMENT_SILENCE_THRESHOLD", "0.1")
    monkeypatch.setenv("AIAVATAR_VAD_USE_ITERATOR", "false")
    monkeypatch.setenv("AIAVATAR_VOICEVOX_URL", "http://voicevox.example:50021")
    monkeypatch.setenv("AIAVATAR_VOICEVOX_SPEAKER", "3")
    monkeypatch.setenv("AIAVATAR_VOICEVOX_CACHE_DIR", "custom/voicevox")
    monkeypatch.setenv("AIAVATAR_OPENAI_TTS_MODEL", "test-tts")
    monkeypatch.setenv("AIAVATAR_OPENAI_TTS_SPEAKER", "coral")
    monkeypatch.setenv("AIAVATAR_OPENAI_TTS_INSTRUCTIONS", "Speak for a test.")
    monkeypatch.setenv("AIAVATAR_OPENAI_TTS_CACHE_DIR", "custom/openai")
    monkeypatch.setenv("AIAVATAR_TIMESTAMP_INTERVAL_SECONDS", "120")
    monkeypatch.setenv("AIAVATAR_TIMESTAMP_TIMEZONE", "UTC")
    monkeypatch.setenv("AIAVATAR_MERGE_REQUEST_THRESHOLD", "1.25")
    monkeypatch.setenv("AIAVATAR_USE_INVOKE_QUEUE", "false")
    monkeypatch.setenv("AIAVATAR_MUTE_ON_BARGE_IN", "false")
    monkeypatch.setenv("AIAVATAR_DEBUG", "false")

    config = cli_config.AppConfig.from_env()
    components = cli_components.build_components(config)
    app = builtin.create_app(
        config=config,
        components=components,
        download_ui=False,
    )
    pipeline = app.state.aiavatar_pipeline
    adapter = app.state.aiavatar_adapter
    tts_routes = components.tts.kwargs["synthesizers"]
    alphabet_to_kana = tts_routes["ja"].kwargs["preprocessors"][0]

    assert app.state.aiavatar_components is components
    assert alphabet_to_kana.kwargs == {
        "openai_api_key": "tts-key",
        "base_url": "https://tts.example/v1",
        "debug": False,
    }
    assert components.stt.kwargs["openai_api_key"] == "stt-key"
    assert components.stt.kwargs["base_url"] == "https://stt.example/v1"
    assert components.stt.kwargs["model"] == "test-transcribe"
    assert components.stt.kwargs["language"] == "en"
    assert components.stt.kwargs["debug"] is False
    near_field_gate = components.vad.kwargs["audio_filters"][0]
    filler_gate, namo_gate = components.vad.kwargs["turn_end_gates"]
    assert near_field_gate.kwargs["initial_ambient_db"] == -24.5
    assert filler_gate.kwargs == {
        "fillers": ["well", "okay"],
        "timeout": 4.5,
    }
    assert namo_gate.kwargs == {
        "language": "en",
        "threshold": 0.7,
        "timeout": 1.5,
        "force_end_phrases": ["goodbye", "stop"],
    }
    assert components.vad.kwargs["silence_duration_threshold"] == 0.8
    assert components.vad.kwargs["segment_silence_threshold"] == 0.1
    assert components.vad.kwargs["use_vad_iterator"] is False
    assert components.vad.kwargs["debug"] is False
    assert tts_routes["ja"].kwargs["base_url"] == "http://voicevox.example:50021"
    assert tts_routes["ja"].kwargs["speaker"] == 3
    assert tts_routes["ja"].kwargs["preprocessors"] == [alphabet_to_kana]
    assert tts_routes["ja"].kwargs["cache_dir"] == "custom/voicevox"
    assert tts_routes["ja"].kwargs["debug"] is False
    assert tts_routes["multi"].kwargs["model"] == "test-tts"
    assert tts_routes["multi"].kwargs["speaker"] == "coral"
    assert tts_routes["multi"].kwargs["openai_api_key"] == "tts-key"
    assert tts_routes["multi"].kwargs["base_url"] == "https://tts.example/v1"
    assert tts_routes["multi"].kwargs["instructions"] == "Speak for a test."
    assert tts_routes["multi"].kwargs["cache_dir"] == "custom/openai"
    assert tts_routes["multi"].kwargs["debug"] is False
    assert components.tts.route_handler("こんにちは", None, None) == "ja"
    assert components.tts.route_handler("こんにちは", None, "JA-jp") == "ja"
    assert components.tts.route_handler("Hello", None, "en-US") == "multi"
    assert components.llm.kwargs["model"] == "test-model"
    assert components.llm.kwargs["openai_api_key"] == "llm-key"
    assert components.llm.kwargs["system_prompt"] == "test system prompt"
    assert components.llm.kwargs["reasoning_effort"] == "low"
    assert components.llm.kwargs["voice_text_tag"] == ["first", "second"]
    assert components.llm.kwargs["debug"] is False
    assert components.llm.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"},
    }
    assert components.llm.kwargs["ws_url"] == "wss://llm.example"
    assert pipeline.timestamp_interval_seconds == 120.0
    assert pipeline.timestamp_timezone == "UTC"
    assert pipeline.merge_request_threshold == 1.25
    assert pipeline.use_invoke_queue is False
    assert pipeline.debug is False
    assert adapter.sts is pipeline
    assert adapter.kwargs == {
        "mute_on_barge_in": False,
        "debug": False,
    }
    await components.vad.partial_handler(
        "こんにちは",
        SimpleNamespace(session_id="session-1"),
    )

    response = pipeline.responses[0]
    assert response.type == "info"
    assert response.session_id == "session-1"
    assert response.metadata == {"partial_request_text": "こんにちは"}

    async with app.router.lifespan_context(app):
        pass
    assert pipeline.shutdown_calls == 1
    assert components._closed is True
