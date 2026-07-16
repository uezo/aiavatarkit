from types import SimpleNamespace

import pytest

from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.vad.stream import (
    RecordingSession as StreamRecordingSession,
    SileroStreamSpeechDetector,
)


class TrackingSpeechRecognizer:
    def __init__(self, text):
        self.text = text
        self.calls = []

    async def recognize(self, session_id, data):
        self.calls.append((session_id, data))
        return SimpleNamespace(text=self.text)


class CaptureAwareSpeechRecognizer(TrackingSpeechRecognizer):
    def __init__(self, text, playback_timeline):
        super().__init__(text)
        self.playback_timeline = playback_timeline
        self.capture_calls = []

    async def recognize(self, session_id, data):
        raise AssertionError("recognize_with_capture must be used")

    async def recognize_with_capture(
        self,
        session_id,
        data,
        *,
        capture_ended_at,
    ):
        self.capture_calls.append((session_id, data, capture_ended_at))
        return SimpleNamespace(text=self.text)


@pytest.mark.parametrize(
    "session_class",
    [
        StreamRecordingSession,
    ],
)
def test_stream_session_speech_recognizer_override_survives_reset(session_class):
    session = session_class("session")
    speech_recognizer = object()

    assert session.get_speech_recognizer() is None

    session.set_speech_recognizer(speech_recognizer)
    session.reset()

    assert session.get_speech_recognizer() is speech_recognizer

    session.clear_speech_recognizer()
    assert session.get_speech_recognizer() is None


@pytest.mark.asyncio
async def test_stream_uses_session_override_and_falls_back_to_default():
    default = TrackingSpeechRecognizer("default")
    override = TrackingSpeechRecognizer("override")
    detector = object.__new__(SileroStreamSpeechDetector)
    detector.speech_recognizer = default
    default_session = StreamRecordingSession("default-session")
    override_session = StreamRecordingSession("override-session")
    override_session.set_speech_recognizer(override)

    default_result = await detector._recognize_audio(default_session, b"default")
    override_result = await detector._recognize_audio(
        override_session,
        b"override",
    )

    assert default_result.text == "default"
    assert override_result.text == "override"
    assert default.calls == [("default-session", b"default")]
    assert override.calls == [("override-session", b"override")]

    override_session.clear_speech_recognizer()
    fallback_result = await detector._recognize_audio(
        override_session,
        b"fallback",
    )

    assert fallback_result.text == "default"
    assert default.calls[-1] == ("override-session", b"fallback")


def test_pipeline_batch_stt_uses_pipeline_owned_override():
    default = TrackingSpeechRecognizer("default")
    override = TrackingSpeechRecognizer("override")
    pipeline = object.__new__(STSPipeline)
    pipeline.stt = default
    pipeline._speech_recognizer_overrides = {}

    assert pipeline.get_speech_recognizer("session") is default

    pipeline.set_speech_recognizer("session", override)

    assert pipeline.get_speech_recognizer("session") is override
    assert pipeline.get_speech_recognizer("other-session") is default

    pipeline.clear_speech_recognizer("session")
    assert pipeline.get_speech_recognizer("session") is default


def test_stream_vad_manages_its_session_override_directly():
    default = TrackingSpeechRecognizer("default")
    override = TrackingSpeechRecognizer("override")
    detector = object.__new__(SileroStreamSpeechDetector)
    detector.speech_recognizer = default
    session = StreamRecordingSession("session")
    detector.recording_sessions = {"session": session}

    detector.set_speech_recognizer("session", override)

    assert session.get_speech_recognizer() is override
    assert detector.get_speech_recognizer("session") is override

    detector.clear_speech_recognizer("session")

    assert session.get_speech_recognizer() is None
    assert detector.get_speech_recognizer("session") is default


@pytest.mark.asyncio
async def test_pipeline_finalize_keeps_batch_override_until_user_clears_it():
    class FinalizableVAD:
        def __init__(self):
            self.finalized = []

        async def finalize_session(self, session_id):
            self.finalized.append(session_id)

    default = TrackingSpeechRecognizer("default")
    override = TrackingSpeechRecognizer("override")
    pipeline = object.__new__(STSPipeline)
    pipeline.stt = default
    pipeline.vad = FinalizableVAD()
    pipeline._speech_recognizer_overrides = {"session": override}

    await pipeline.finalize("session")

    assert pipeline.vad.finalized == ["session"]
    assert pipeline.get_speech_recognizer("session") is override

    pipeline.clear_speech_recognizer("session")
    assert pipeline.get_speech_recognizer("session") is default
