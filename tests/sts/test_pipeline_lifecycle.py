import pytest

from aiavatar.sts.llm.base import LLMServiceDummy
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.stt.base import SpeechRecognizerDummy
from aiavatar.sts.tts.base import SpeechSynthesizerDummy
from aiavatar.sts.vad.base import SpeechDetectorDummy


class FakePerformanceRecorder:
    def __init__(self, events, **kwargs):
        self.events = events

    def close(self):
        self.events.append("performance:close")


class FakeVoiceRecorder:
    def __init__(self, events, **kwargs):
        self.events = events

    async def stop(self):
        self.events.append("voice:stop")


class FakeLLM:
    def __init__(self, events, **kwargs):
        self.events = events
        self.kwargs = kwargs

    async def close(self):
        self.events.append("llm:close")


def create_pipeline(*, performance_recorder=None, voice_recorder=None):
    return STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=LLMServiceDummy(),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=performance_recorder,
        voice_recorder=voice_recorder,
    )


@pytest.mark.asyncio
async def test_shutdown_closes_only_internally_created_resources(monkeypatch):
    events = []

    monkeypatch.setattr(
        "aiavatar.sts.pipeline.SQLitePerformanceRecorder",
        lambda **kwargs: FakePerformanceRecorder(events, **kwargs),
    )
    monkeypatch.setattr(
        "aiavatar.sts.pipeline.FileVoiceRecorder",
        lambda **kwargs: FakeVoiceRecorder(events, **kwargs),
    )

    pipeline = create_pipeline()

    await pipeline.shutdown()
    await pipeline.shutdown()

    assert events == ["voice:stop", "performance:close"]


@pytest.mark.asyncio
async def test_shutdown_leaves_injected_resources_open():
    events = []
    performance_recorder = FakePerformanceRecorder(events)
    voice_recorder = FakeVoiceRecorder(events)
    pipeline = create_pipeline(
        performance_recorder=performance_recorder,
        voice_recorder=voice_recorder,
    )

    await pipeline.shutdown()

    assert events == []


@pytest.mark.asyncio
async def test_shutdown_closes_internally_created_llm_once(monkeypatch):
    events = []
    monkeypatch.setattr(
        "aiavatar.sts.pipeline.ChatGPTService",
        lambda **kwargs: FakeLLM(events, **kwargs),
    )
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=FakePerformanceRecorder(events),
        voice_recorder=FakeVoiceRecorder(events),
    )

    await pipeline.shutdown()
    await pipeline.shutdown()

    assert events == ["llm:close"]


def test_default_llm_uses_default_model_and_unset_generation_params(
    monkeypatch,
    tmp_path,
):
    events = []
    created = []

    def create_llm(**kwargs):
        llm = FakeLLM(events, **kwargs)
        created.append(llm)
        return llm

    monkeypatch.setattr("aiavatar.sts.pipeline.ChatGPTService", create_llm)
    STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=FakePerformanceRecorder(events),
        voice_recorder=FakeVoiceRecorder(events),
        db_connection_str=str(tmp_path / "default-llm.db"),
    )

    assert created[0].kwargs["model"] == "gpt-5.6-terra"
    assert created[0].kwargs["temperature"] is None
    assert created[0].kwargs["reasoning_effort"] is None


def test_default_llm_forwards_generation_params(monkeypatch, tmp_path):
    events = []
    created = []

    def create_llm(**kwargs):
        llm = FakeLLM(events, **kwargs)
        created.append(llm)
        return llm

    monkeypatch.setattr("aiavatar.sts.pipeline.ChatGPTService", create_llm)
    STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        tts=SpeechSynthesizerDummy(),
        llm_temperature=0.0,
        llm_reasoning_effort="none",
        performance_recorder=FakePerformanceRecorder(events),
        voice_recorder=FakeVoiceRecorder(events),
        db_connection_str=str(tmp_path / "configured-llm.db"),
    )

    assert created[0].kwargs["temperature"] == 0.0
    assert created[0].kwargs["reasoning_effort"] == "none"
