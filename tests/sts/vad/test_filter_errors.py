import asyncio
from unittest.mock import Mock

import pytest

from aiavatar.sts.vad.filters import AudioFilter, RecoverableAudioFilterError
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector


class TemporaryFailureFilter(AudioFilter):
    def __init__(self, error):
        self.error = error
        self.calls = []
        self.reset_session = Mock()

    def process(self, samples, session_id):
        self.calls.append((samples, session_id))
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return samples


class UnusedRecognizer:
    async def recognize(self, session_id, audio):
        raise AssertionError("These tests must not invoke speech recognition")


@pytest.fixture(params=[SileroSpeechDetector, SileroStreamSpeechDetector], ids=["batch", "stream"])
def make_detector(monkeypatch, request):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", lambda *args: None)
    monkeypatch.setattr(SileroSpeechDetector, "_create_vad_iterator", lambda *args: Mock())

    def create(filters):
        kwargs = {"audio_filters": filters}
        if request.param is SileroStreamSpeechDetector:
            kwargs["speech_recognizer"] = UnusedRecognizer()
        detector = request.param(**kwargs)
        detector._detect_speech_silero = Mock(return_value=False)
        return detector

    return create


@pytest.mark.asyncio
async def test_recoverable_filter_error_drops_chunk_without_reset(make_detector):
    failing = TemporaryFailureFilter(RecoverableAudioFilterError("temporary overload"))
    downstream = Mock(spec=AudioFilter)
    downstream.process.side_effect = lambda samples, session_id: samples
    detector = make_detector([failing, downstream])
    session = detector.get_session("speaker")
    session.data["context_id"] = "context-1"
    session.is_recording = True
    session.buffer.extend(b"\x10\x00")
    session.preroll_buffer.append(b"\x20\x00")
    session.vad_buffer.extend(b"\x30\x00")
    session.record_duration = 0.25
    session.reset = Mock(wraps=session.reset)
    rejected = b"\x01\x00" * 512
    accepted = b"\x02\x00" * 512

    assert await detector.process_samples(rejected, "speaker") is True

    assert detector.get_session("speaker") is session
    assert session.buffer == b"\x10\x00"
    assert list(session.preroll_buffer) == [b"\x20\x00"]
    assert session.vad_buffer == b"\x30\x00"
    assert session.record_duration == 0.25
    assert session.data["context_id"] == "context-1"
    downstream.process.assert_not_called()
    detector._detect_speech_silero.assert_not_called()

    assert await detector.process_samples(accepted, "speaker") is True

    assert failing.calls == [(rejected, "speaker"), (accepted, "speaker")]
    downstream.process.assert_called_once_with(accepted, "speaker")
    assert session.buffer == b"\x10\x00" + accepted
    assert session.data["context_id"] == "context-1"
    detector._detect_speech_silero.assert_called_once()
    session.reset.assert_not_called()
    session.vad_iterator.reset_states.assert_not_called()
    failing.reset_session.assert_not_called()
    downstream.reset_session.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [BufferError, RuntimeError, asyncio.CancelledError])
async def test_other_filter_errors_propagate(make_detector, error_type):
    error = error_type("not recoverable")
    failing = TemporaryFailureFilter(error)
    downstream = Mock(spec=AudioFilter)
    detector = make_detector([failing, downstream])

    with pytest.raises(error_type) as raised:
        await detector.process_samples(b"\x01\x00", "speaker")

    assert raised.value is error
    downstream.process.assert_not_called()
    failing.reset_session.assert_not_called()


@pytest.mark.asyncio
async def test_empty_filter_output_still_reaches_downstream_filter(make_detector):
    buffering = Mock(spec=AudioFilter)
    buffering.process.return_value = b""
    downstream = Mock(spec=AudioFilter)
    downstream.process.return_value = b""
    detector = make_detector([buffering, downstream])

    assert await detector.process_samples(b"\x01\x00", "speaker") is False

    downstream.process.assert_called_once_with(b"", "speaker")
