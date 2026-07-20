from datetime import datetime, timezone

import pytest

from aiavatar.sts.llm.base import LLMServiceDummy
from aiavatar.sts.models import STSRequest
from aiavatar.sts.performance_recorder import PerformanceRecord, PerformanceRecorder
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.stt.base import SpeechRecognizerDummy
from aiavatar.sts.tts.base import SpeechSynthesizerDummy
from aiavatar.sts.vad.base import SpeechDetectorDummy


class CapturePerformanceRecorder(PerformanceRecorder):
    def __init__(self):
        self.records = []

    def record(self, record: PerformanceRecord):
        self.records.append(record)

    def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected_threshold"),
    [
        (
            {
                "vad_performance": {
                    "speech_end_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
                    "silence_threshold_time": 0.5,
                    "stt_after_threshold_time": 0.1,
                    "turn_end_gate_time": 0.3,
                    "turn_end_gate_held": True,
                }
            },
            0.5,
        ),
        (None, None),
    ],
)
async def test_pipeline_copies_optional_vad_performance(
    tmp_path,
    metadata,
    expected_threshold,
):
    recorder = CapturePerformanceRecorder()
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=LLMServiceDummy(response_text="ok", db_connection_str=str(tmp_path / "llm.db")),
        tts=SpeechSynthesizerDummy(synthesized_bytes=b"audio"),
        performance_recorder=recorder,
        voice_recorder_enabled=False,
        db_connection_str=str(tmp_path / "pipeline.db"),
    )

    responses = [
        response
        async for response in pipeline.invoke(STSRequest(
            session_id="session",
            text="hello",
            metadata=metadata,
        ))
    ]

    assert responses[-1].type == "final"
    record = recorder.records[0]
    assert record.session_id == "session"
    assert record.silence_threshold_time == expected_threshold
    if metadata is None:
        assert record.speech_end_at is None
        assert record.stt_after_threshold_time is None
        assert record.turn_end_gate_time is None
        assert record.turn_end_gate_held is None
    else:
        assert record.speech_end_at == datetime(2026, 1, 2, tzinfo=timezone.utc)
        assert record.stt_after_threshold_time == 0.1
        assert record.turn_end_gate_time == 0.3
        assert record.turn_end_gate_held is True
