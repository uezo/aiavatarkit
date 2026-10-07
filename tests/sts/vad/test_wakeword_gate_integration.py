"""Wakeword admission through the real stream callback, without external services."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_taking_gates import (
    TurnTakingDecision, TurnTakingGate, TurnTakingGateManager,
)
from aiavatar.sts.vad.turn_taking_gates.wakeword import WakewordGate


class ApplicationGate(TurnTakingGate):
    def __init__(self):
        super().__init__()
        self.result = True
        self.texts = []

    async def should_take_turn(
        self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs,
    ):
        self.texts.append(user_text)
        return TurnTakingDecision(self.result, None, "application_policy")


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [0.3, 5.0], ids=["short", "long_without_bypass"])
async def test_stream_wakeword_controls_invocation_and_preserves_complete_request(monkeypatch, duration):
    # Neither constructing the detector nor dispatching finalized text may load
    # a model, perform recognition, or construct a real pipeline/provider/DB.
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", lambda *args: None)
    monkeypatch.setattr(SileroSpeechDetector, "_create_vad_iterator", lambda *args: None)

    async def unexpected_recognition(*args):
        raise AssertionError("Finalized text must not be recognized again")

    last_conversation = {}

    async def get_last_conversation_at(session_id):
        return last_conversation.get(session_id)

    downstream = ApplicationGate()
    manager = TurnTakingGateManager([
        WakewordGate(["こんにちは"], get_last_conversation_at=get_last_conversation_at),
        downstream,
    ], bypass_enabled=False)
    vad = SileroStreamSpeechDetector(
        speech_recognizer=SimpleNamespace(recognize=unexpected_recognition),
        turn_taking_gate=manager,
    )
    vad.get_session("connection")
    vad.set_session_data("connection", "context_id", "conversation", create_session=True)
    vad.set_session_data("connection", "user_id", "user")
    requests = []

    async def memory_invoke(request):
        requests.append(request)
        for response in ():
            yield response

    pipeline = object.__new__(STSPipeline)
    pipeline.vad = vad
    pipeline.response_handlers = [SimpleNamespace(can_handle=lambda session_id: True)]
    pipeline.invoke = memory_invoke
    vad.on_speech_detected(pipeline.on_speech_detected)
    metadata = {"recording_id": "recording"}

    async def dispatch(text):
        await vad.execute_on_speech_detected(b"audio", text, metadata, duration, "connection")

    try:
        await dispatch("今日の天気は？")
        assert requests == []
        assert downstream.texts == []

        text = "こんにちは、今日の天気は？"
        await dispatch(text)
        assert downstream.texts == [text]
        assert len(requests) == 1
        request = requests[0]
        assert request.text == text
        assert request.audio_data == b"audio"
        assert request.audio_duration == duration
        assert request.metadata is metadata
        assert request.session_id == "connection"
        assert request.context_id == "conversation"
        assert request.user_id == "user"

        # Only the supplied activity source determines conversation continuity.
        await dispatch("明日は？")
        assert len(requests) == 1
        last_conversation["connection"] = datetime.now(timezone.utc)
        await dispatch("明日は？")
        assert [request.text for request in requests] == [text, "明日は？"]

        downstream.result = False
        await dispatch("こんにちは、質問です")
        assert len(requests) == 2
        assert downstream.texts[-1] == "こんにちは、質問です"

        last_conversation["connection"] = datetime.now(timezone.utc) - timedelta(seconds=120)
        calls = len(downstream.texts)
        await dispatch("もう一つ質問です")
        assert len(downstream.texts) == calls
        assert len(requests) == 2
    finally:
        await vad.finalize_session("connection")
