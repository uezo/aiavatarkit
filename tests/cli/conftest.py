from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter

import aiavatar.cli.builtin as builtin
import aiavatar.cli.components as cli_components
import aiavatar.cli.tts as cli_tts


DEFAULT_APP_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AIAVATAR_STT_OPENAI_API_KEY",
    "AIAVATAR_STT_OPENAI_BASE_URL",
    "AIAVATAR_LLM_OPENAI_API_KEY",
    "AIAVATAR_LLM_OPENAI_BASE_URL",
    "AIAVATAR_TTS_OPENAI_API_KEY",
    "AIAVATAR_TTS_OPENAI_BASE_URL",
    "AIAVATAR_LLM_API",
    "AIAVATAR_JA_TTS",
    "AIAVATAR_JA_TTS_CONFIG",
    "AIAVATAR_MULTI_TTS",
    "AIAVATAR_MULTI_TTS_CONFIG",
    "AIAVATAR_LLM_EXTRA_BODY",
    "AIAVATAR_DEBUG",
    "AIAVATAR_STT_MODEL",
    "AIAVATAR_STT_LANGUAGE",
    "AIAVATAR_NEAR_FIELD_INITIAL_AMBIENT_DB",
    "AIAVATAR_FILLER_PHRASES",
    "FILLER_TURN_GATE_TIMEOUT",
    "AIAVATAR_NAMO_LANGUAGE",
    "NAMO_TURN_THRESHOLD",
    "NAMO_TURN_GATE_TIMEOUT",
    "AIAVATAR_NAMO_FORCE_END_PHRASES",
    "AIAVATAR_VAD_SILENCE_DURATION_THRESHOLD",
    "AIAVATAR_VAD_SEGMENT_SILENCE_THRESHOLD",
    "AIAVATAR_VAD_USE_ITERATOR",
    "AIAVATAR_LLM_MODEL",
    "AIAVATAR_LLM_SYSTEM_PROMPT",
    "AIAVATAR_LLM_REASONING_EFFORT",
    "AIAVATAR_LLM_VOICE_TEXT_TAGS",
    "AIAVATAR_VOICEVOX_URL",
    "AIAVATAR_VOICEVOX_SPEAKER",
    "AIAVATAR_VOICEVOX_CACHE_DIR",
    "AIAVATAR_OPENAI_TTS_MODEL",
    "AIAVATAR_OPENAI_TTS_SPEAKER",
    "AIAVATAR_OPENAI_TTS_INSTRUCTIONS",
    "AIAVATAR_OPENAI_TTS_CACHE_DIR",
    "AIAVATAR_TIMESTAMP_INTERVAL_SECONDS",
    "AIAVATAR_TIMESTAMP_TIMEZONE",
    "AIAVATAR_MERGE_REQUEST_THRESHOLD",
    "AIAVATAR_USE_INVOKE_QUEUE",
    "AIAVATAR_MUTE_ON_BARGE_IN",
)


@pytest.fixture
def clean_builtin_environment(monkeypatch):
    for name in DEFAULT_APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeComponent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePreprocessor(FakeComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.http_client = SimpleNamespace(close=AsyncMock())


class FakeVad(FakeComponent):
    def on_speech_detecting(self, handler):
        self.partial_handler = handler
        return handler


class FakeRouter(FakeComponent):
    def route(self, handler):
        self.route_handler = handler
        return handler


class FakePipeline:
    def __init__(self, **components):
        vars(self).update(components)
        self.responses = []
        self.shutdown_calls = 0

    async def handle_response(self, response):
        self.responses.append(response)

    async def shutdown(self):
        self.shutdown_calls += 1


class FakeAdapter:
    def __init__(self, sts, **kwargs):
        self.sts = sts
        self.kwargs = kwargs
        self.responses = []

    async def handle_response(self, response):
        self.responses.append(response)

    def get_websocket_router(self):
        return APIRouter()


@pytest.fixture
def component_fakes(monkeypatch):
    monkeypatch.setattr(cli_components, "OpenAISpeechRecognizer", FakeComponent)
    monkeypatch.setattr(
        cli_components,
        "OpenAIResponsesWebSocketService",
        FakeComponent,
    )
    monkeypatch.setattr(cli_components, "NearFieldAudioGate", FakeComponent)
    monkeypatch.setattr(cli_components, "FillerOnlyTurnEndGate", FakeComponent)
    monkeypatch.setattr(cli_components, "NamoTurnEndGate", FakeComponent)
    monkeypatch.setattr(cli_components, "SileroStreamSpeechDetector", FakeVad)
    monkeypatch.setattr(cli_tts, "AlphabetToKanaPreprocessor", FakePreprocessor)
    monkeypatch.setattr(cli_tts, "VoicevoxSpeechSynthesizer", FakeComponent)
    monkeypatch.setattr(cli_tts, "OpenAISpeechSynthesizer", FakeComponent)
    monkeypatch.setattr(cli_tts, "SpeechSynthesizerRouter", FakeRouter)
    return SimpleNamespace(component=FakeComponent)


@pytest.fixture
def builtin_fakes(monkeypatch):
    monkeypatch.setattr(builtin, "STSPipeline", FakePipeline)
    monkeypatch.setattr(builtin, "AIAvatarWebSocketServer", FakeAdapter)
    monkeypatch.setattr(builtin, "setup_admin_panel", lambda *args, **kwargs: None)
