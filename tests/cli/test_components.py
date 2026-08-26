from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import aiavatar.cli.components as cli_components
import aiavatar.cli.config as cli_config
import aiavatar.cli.tts as cli_tts


def test_create_chat_completions_llm(
    monkeypatch,
    clean_builtin_environment,
):
    class FakeChatGPTService:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(cli_components, "ChatGPTService", FakeChatGPTService)
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv(
        "AIAVATAR_LLM_OPENAI_BASE_URL",
        "https://llm.example/v1",
    )
    monkeypatch.setenv("AIAVATAR_LLM_API", "chat-completions")
    monkeypatch.setenv("AIAVATAR_LLM_MODEL", "provider/model")
    monkeypatch.setenv("AIAVATAR_LLM_SYSTEM_PROMPT", "test prompt")
    monkeypatch.setenv("AIAVATAR_LLM_EXTRA_BODY", '{"thinking":{}}')
    monkeypatch.setenv("AIAVATAR_LLM_VOICE_TEXT_TAGS", '["ack","answer"]')
    monkeypatch.setenv("AIAVATAR_DEBUG", "false")

    llm = cli_components.create_default_llm(cli_config.AppConfig.from_env())

    assert llm.kwargs["openai_api_key"] == "llm-key"
    assert llm.kwargs["base_url"] == "https://llm.example/v1"
    assert llm.kwargs["model"] == "provider/model"
    assert llm.kwargs["reasoning_effort"] is None
    assert llm.kwargs["extra_body"] == {"thinking": {}}
    assert llm.kwargs["voice_text_tag"] == ["ack", "answer"]
    assert llm.kwargs["debug"] is False


@pytest.mark.asyncio
async def test_component_set_closes_owned_resources_once():
    llm_client = SimpleNamespace(close=AsyncMock())
    stt = SimpleNamespace(close=AsyncMock())
    tts = SimpleNamespace(close=AsyncMock())
    preprocessor_client = SimpleNamespace(close=AsyncMock())
    vad = object()
    llm = SimpleNamespace(openai_client=llm_client)
    components = cli_components.ComponentSet(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        _managed_resources=[preprocessor_client],
    )

    assert tuple(components) == (vad, stt, llm, tts)
    await components.close()
    await components.close()

    llm_client.close.assert_awaited_once()
    stt.close.assert_awaited_once()
    tts.close.assert_awaited_once()
    preprocessor_client.close.assert_awaited_once()


def test_build_components_preserves_defaults(
    monkeypatch,
    clean_builtin_environment,
    component_fakes,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    components = cli_components.build_components()

    assert components.stt.kwargs == {
        "openai_api_key": "test-key",
        "base_url": None,
        "model": "gpt-transcribe",
        "language": None,
        "debug": True,
    }
    near_field_gate = components.vad.kwargs["audio_filters"][0]
    filler_gate, namo_gate = components.vad.kwargs["turn_end_gates"]
    assert near_field_gate.kwargs == {"initial_ambient_db": -30.0}
    assert filler_gate.kwargs == {
        "fillers": cli_config.DEFAULT_JA_FILLERS + cli_config.DEFAULT_EN_FILLERS,
        "timeout": 3.0,
    }
    assert namo_gate.kwargs == {
        "language": None,
        "threshold": 0.5,
        "timeout": 1.0,
        "force_end_phrases": ["こんにちは。"],
    }
    assert components.vad.kwargs["silence_duration_threshold"] == 0.5
    assert components.vad.kwargs["segment_silence_threshold"] == 0.05
    assert components.vad.kwargs["use_vad_iterator"] is True
    assert components.vad.kwargs["debug"] is True

    assert components.llm.kwargs["model"] == "gpt-5.6-terra"
    assert components.llm.kwargs["reasoning_effort"] == "none"
    assert components.llm.kwargs["voice_text_tag"] == ["ack", "answer"]
    assert components.llm.kwargs["system_prompt"] == cli_config.SYSTEM_PROMPT
    assert components.llm.kwargs["debug"] is True

    tts_routes = components.tts.kwargs["synthesizers"]
    alphabet_to_kana = tts_routes["ja"].kwargs["preprocessors"][0]
    assert alphabet_to_kana.kwargs == {
        "openai_api_key": "test-key",
        "base_url": None,
        "debug": True,
    }
    assert tts_routes["ja"].kwargs["base_url"] == "http://127.0.0.1:50021"
    assert tts_routes["ja"].kwargs["speaker"] == 46
    assert tts_routes["ja"].kwargs["cache_dir"] == "ttscache/voicevox"
    assert tts_routes["ja"].kwargs["preprocessors"] == [alphabet_to_kana]
    assert tts_routes["ja"].kwargs["debug"] is True
    assert tts_routes["multi"].kwargs["model"] == "gpt-4o-mini-tts"
    assert tts_routes["multi"].kwargs["speaker"] == "sage"
    assert tts_routes["multi"].kwargs["audio_format"] == "wav"
    assert tts_routes["multi"].kwargs["preprocessors"] == []
    assert (
        tts_routes["multi"].kwargs["instructions"]
        == cli_config.OPENAI_TTS_INSTRUCTIONS
    )
    assert tts_routes["multi"].kwargs["cache_dir"] == "ttscache/openai"
    assert tts_routes["multi"].kwargs["debug"] is True


def test_build_components_allows_non_openai_tts_without_tts_key(
    monkeypatch,
    clean_builtin_environment,
    component_fakes,
):
    monkeypatch.setenv("AIAVATAR_STT_OPENAI_API_KEY", "stt-key")
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("AIAVATAR_JA_TTS", "voicevox")
    monkeypatch.setenv(
        "AIAVATAR_JA_TTS_CONFIG",
        '{"speaker":3,"alphabet_to_kana":false}',
    )
    monkeypatch.setenv("AIAVATAR_MULTI_TTS", "instant")
    monkeypatch.setenv(
        "AIAVATAR_MULTI_TTS_CONFIG",
        '{"method":"POST","url":"https://tts.example/speech"}',
    )
    monkeypatch.setattr(
        cli_tts,
        "create_instant_synthesizer",
        lambda **kwargs: component_fakes.component(**kwargs),
    )

    components = cli_components.build_components()
    routes = components.tts.kwargs["synthesizers"]

    assert components._managed_resources == []
    assert routes["ja"].kwargs["speaker"] == 3
    assert routes["ja"].kwargs["preprocessors"] == []
    assert routes["multi"].kwargs["method"] == "POST"
    assert routes["multi"].kwargs["url"] == "https://tts.example/speech"
    assert routes["multi"].kwargs["preprocessors"] == []


def test_build_components_applies_instant_aivis_config(
    monkeypatch,
    clean_builtin_environment,
    component_fakes,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AIAVATAR_JA_TTS", "instant")
    monkeypatch.setenv(
        "AIAVATAR_JA_TTS_CONFIG",
        """{
            "method": "POST",
            "url": "https://aivis.example/tts",
            "headers": {
                "Authorization": "Bearer aivis-key",
                "Content-Type": "application/json"
            },
            "json": {
                "model_uuid": "model-uuid",
                "text": "{text}",
                "output_format": "wav",
                "output_sampling_rate": 24000,
                "output_audio_channels": "stereo",
                "use_ssml": true
            },
            "sample_rate": 16000,
            "cache_dir": null,
            "alphabet_to_kana": false
        }""",
    )
    monkeypatch.setenv("AIAVATAR_DEBUG", "false")
    monkeypatch.setattr(
        cli_tts,
        "create_instant_synthesizer",
        lambda **kwargs: component_fakes.component(**kwargs),
    )

    components = cli_components.build_components()
    tts_ja = components.tts.kwargs["synthesizers"]["ja"]

    assert tts_ja.kwargs["url"] == "https://aivis.example/tts"
    assert tts_ja.kwargs["headers"]["Authorization"] == "Bearer aivis-key"
    assert tts_ja.kwargs["json"] == {
        "model_uuid": "model-uuid",
        "text": "{text}",
        "output_format": "wav",
        "output_sampling_rate": 24000,
        "output_audio_channels": "stereo",
        "use_ssml": True,
    }
    assert tts_ja.kwargs["sample_rate"] == 16000
    assert tts_ja.kwargs["preprocessors"] == []
    assert tts_ja.kwargs["cache_dir"] is None
    assert tts_ja.kwargs["debug"] is False


def test_build_components_uses_overrides_and_injects_stt_into_default_vad(
    clean_builtin_environment,
    component_fakes,
):
    custom_stt = object()
    custom_llm = object()
    custom_tts = object()

    components = cli_components.build_components(
        stt=custom_stt,
        llm=custom_llm,
        tts=custom_tts,
    )

    assert components.stt is custom_stt
    assert components.llm is custom_llm
    assert components.tts is custom_tts
    assert components.vad.kwargs["speech_recognizer"] is custom_stt


def test_build_components_can_disable_namo_turn(
    monkeypatch,
    clean_builtin_environment,
    component_fakes,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        cli_components,
        "_namo_turn_end_gate_class",
        lambda: pytest.fail("Namo Turn must not be imported"),
    )

    components = cli_components.build_components(use_namo_turn=False)

    turn_end_gates = components.vad.kwargs["turn_end_gates"]
    assert len(turn_end_gates) == 1
    assert turn_end_gates[0].kwargs == {
        "fillers": cli_config.DEFAULT_JA_FILLERS + cli_config.DEFAULT_EN_FILLERS,
        "timeout": 3.0,
    }
