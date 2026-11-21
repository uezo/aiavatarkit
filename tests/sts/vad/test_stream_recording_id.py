"""No audio models, providers, files or credentials are used."""

import asyncio
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.models import RequestContext, STSResponse
from aiavatar.sts.pipeline import STSPipeline, ResponseHandler


class Recognizer:
    async def recognize(self, session_id, data):
        return SimpleNamespace(text="聞いた内容")


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setattr(SileroStreamSpeechDetector, "_init_silero_model", lambda *a: None)
    monkeypatch.setattr(SileroStreamSpeechDetector, "_create_vad_iterator", lambda *a: None)
    monkeypatch.setattr(SileroStreamSpeechDetector, "_detect_speech_silero", lambda self, data, session: any(data))
    return SileroStreamSpeechDetector(speech_recognizer=Recognizer(),
        segment_silence_threshold=0.1, silence_duration_threshold=0.4,
        min_duration=0.1, max_duration=10.0)


VOICE = b"\x01\x00" * 1600
SILENCE = b"\x00\x00" * 1600


@pytest.mark.asyncio
async def test_recording_id_survives_partial_and_is_snapshotted_before_reset(detector):
    session = detector.get_session("s")
    assert session.recording_id is None
    final, partial = [], []
    @detector.on_speech_detecting
    async def on_partial(text, recording):
        partial.append(recording.recording_id)
    @detector.on_speech_detected
    async def on_final(audio, text, metadata, duration, session_id):
        final.append(metadata)
    await detector.process_samples(VOICE, "s")
    identifier = session.recording_id
    assert identifier
    await detector.process_samples(VOICE, "s")
    await detector.process_samples(SILENCE, "s")
    await session.pending_recognition_task
    assert partial == [identifier]
    await detector.process_samples(VOICE, "s")
    assert session.recording_id == identifier
    for _ in range(4):
        await detector.process_samples(SILENCE, "s")
    assert session.recording_id is None
    await asyncio.sleep(0)
    assert final[0]["recording_id"] == identifier
    assert "vad_performance" in final[0]
    await detector.process_samples(VOICE, "s")
    assert session.recording_id and session.recording_id != identifier
    await detector.finalize_session("s")


@pytest.mark.asyncio
async def test_speech_started_callbacks_share_fresh_metadata_for_each_recording(detector):
    started, annotated = [], []
    finalized = asyncio.Queue()

    @detector.on_speech_started
    def mark_first_recording(session_id, metadata):
        started.append(metadata)
        if len(started) == 1:
            metadata["first_recording"] = session_id
        return {"ignored_return": True}

    @detector.on_speech_started
    def number_recording(session_id, metadata):
        annotated.append((session_id, metadata))
        metadata["recording_number"] = len(annotated)

    @detector.on_speech_detected
    async def on_final(audio, text, metadata, duration, session_id):
        finalized.put_nowait(metadata)

    try:
        for recording_number in (1, 2):
            await detector.process_samples(VOICE, "s")
            assert len(started) == len(annotated) == recording_number
            session_id, shared_metadata = annotated[-1]
            assert session_id == "s"
            assert shared_metadata is started[-1]

            await detector.process_samples(VOICE, "s")
            for _ in range(4):
                await detector.process_samples(SILENCE, "s")
            metadata = await asyncio.wait_for(finalized.get(), timeout=1)
            assert metadata["recording_number"] == recording_number
            assert metadata.get("first_recording") == ("s" if recording_number == 1 else None)
            assert "ignored_return" not in metadata

        assert started[0] is not started[1]
    finally:
        await detector.finalize_session("s")


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", ["reset_session", "reset_session_audio_state", "mute", "finalize"])
async def test_recording_reset_clears_id(detector, reset):
    await detector.process_samples(VOICE, "s")
    session = detector.get_session("s")
    assert session.recording_id
    if reset == "mute":
        detector.should_mute = lambda: True
        await detector.process_samples(VOICE, "s")
    elif reset == "finalize":
        await detector.finalize_session("s")
    else:
        getattr(detector, reset)("s")
    assert session.recording_id is None


@pytest.mark.asyncio
async def test_older_partial_result_does_not_replace_newer_result(detector):
    entered, release = asyncio.Event(), asyncio.Event()
    class DelayedRecognizer:
        async def recognize(self, session_id, data):
            if entered.is_set():
                return SimpleNamespace(text="新しい認識")
            entered.set()
            await release.wait()
            return SimpleNamespace(text="古い認識")
    detector.speech_recognizer = DelayedRecognizer()
    partial = []
    @detector.on_speech_detecting
    async def on_partial(text, session):
        partial.append(text)
    await detector.process_samples(VOICE, "s")
    await detector.process_samples(SILENCE, "s")
    session = detector.get_session("s")
    old = session.pending_recognition_task
    await entered.wait()
    try:
        await detector.process_samples(VOICE, "s")
        await detector.process_samples(SILENCE, "s")
        await session.pending_recognition_task
    finally:
        release.set()
        await old
    assert partial == ["新しい認識"]
    assert session.last_recognized_text == "新しい認識"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "captured_context", [None, RequestContext(track_interruption=True), {"track_interruption": True}],
    ids=["missing", "typed", "plain-dict"],
)
async def test_recording_metadata_reaches_existing_request_path(captured_context):
    pipeline = STSPipeline.__new__(STSPipeline)
    pipeline.vad = SimpleNamespace(get_session_data=lambda *args: None)
    received = []
    async def handle(response):
        pass
    async def invoke(request):
        received.append(request)
        yield STSResponse(type="final", session_id=request.session_id)
    pipeline.response_handlers = [ResponseHandler(can_handle=lambda _: True, handle_response=handle, stop_response=handle)]
    pipeline.invoke = invoke
    metadata = {"recording_id": "a"}
    if captured_context is not None:
        metadata["request_context"] = captured_context
    await pipeline.on_speech_detected(b"audio", "text", metadata, 1.0, "s")
    assert received[0].metadata["recording_id"] == "a"
    expected_context = captured_context if isinstance(captured_context, RequestContext) else None
    assert received[0].request_context is expected_context


@pytest.mark.asyncio
async def test_max_duration_final_keeps_recording_id(detector):
    detector.max_duration = 0.3
    final = []
    @detector.on_speech_detected
    async def on_final(audio, text, metadata, duration, session_id):
        final.append(metadata)
    await detector.process_samples(VOICE, "s")
    session = detector.get_session("s")
    identifier = session.recording_id
    await detector.process_samples(SILENCE, "s")
    await session.pending_recognition_task
    await detector.process_samples(VOICE, "s")
    await asyncio.sleep(0)
    assert session.recording_id is None
    assert final[0]["recording_id"] == identifier


@pytest.mark.asyncio
async def test_vad_internal_state_reset_does_not_end_recording(detector):
    await detector.process_samples(VOICE, "s")
    session = detector.get_session("s")
    identifier = session.recording_id
    detector.reset_vad_state("s")
    session.reset_turn_end_timing()
    detector.turn_end_gate_manager.reset_session("s", session=session)
    assert session.recording_id == identifier


@pytest.mark.asyncio
@pytest.mark.parametrize("next_recording_id", [None, "next"])
async def test_partial_callback_reset_still_calls_remaining_handlers(detector, next_recording_id):
    received = []
    trigger_checks = []
    @detector.on_recording_started
    async def on_recording_started(session_id):
        pass
    @detector.should_trigger_recording_started
    def should_trigger(text, session):
        trigger_checks.append(session.recording_id)
        return False
    @detector.on_speech_detecting
    async def first(text, session):
        detector.reset_session(session.session_id)
        session.recording_id = next_recording_id
    @detector.on_speech_detecting
    async def second(text, session):
        received.append((text, session.recording_id))
    await detector.process_samples(VOICE, "s")
    await detector.process_samples(SILENCE, "s")
    task = detector.get_session("s").pending_recognition_task
    await task
    assert received == [("聞いた内容", next_recording_id)]
    assert trigger_checks[-1] == next_recording_id
