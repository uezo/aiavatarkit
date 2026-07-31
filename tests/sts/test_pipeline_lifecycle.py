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
