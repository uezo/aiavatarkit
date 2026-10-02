import asyncio
import inspect

import pytest

from aiavatar.sts.vad import SpeechDetectorDummy


def test_dummy_session_data_requires_an_existing_session():
    vad = SpeechDetectorDummy()

    vad.set_session_data("session-1", "user_id", "user-1")

    assert vad.get_session_data("session-1", "user_id") is None


@pytest.mark.asyncio
async def test_dummy_session_data_is_created_and_finalized():
    vad = SpeechDetectorDummy()

    vad.set_session_data(
        "session-1",
        "user_id",
        "user-1",
        create_session=True,
    )
    vad.set_session_data("session-1", "context_id", "context-1")

    assert vad.get_session_data("session-1", "user_id") == "user-1"
    assert vad.get_session_data("session-1", "context_id") == "context-1"

    await vad.finalize_session("session-1")

    assert vad.get_session_data("session-1", "user_id") is None
    assert vad.get_session_data("session-1", "context_id") is None


@pytest.mark.asyncio
async def test_speech_detected_true_does_not_consume_the_event():
    vad = SpeechDetectorDummy()
    assert inspect.iscoroutinefunction(vad._execute_on_speech_detected)
    seen = []

    @vad.on_speech_detected
    async def consume(audio, text, metadata, duration, session_id):
        seen.append(("first", text))
        return text == "consume"

    @vad.on_speech_detected
    async def observe(audio, text, metadata, duration, session_id):
        seen.append(("second", text))

    await vad._execute_on_speech_detected(b"", "consume", {}, 1.0, "s")
    await asyncio.create_task(vad._execute_on_speech_detected(b"", "continue", {}, 1.0, "s"))

    assert seen == [
        ("first", "consume"), ("second", "consume"),
        ("first", "continue"), ("second", "continue"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, False, True, 0, 1, "handled", {"handled": True}])
async def test_speech_detected_results_keep_registration_order(result):
    vad = SpeechDetectorDummy()
    seen = []
    metadata = {"recording_id": "recording"}

    @vad.on_speech_detected
    async def first(*args):
        seen.append(("first", args))
        return result

    @vad.on_speech_detected
    async def second(*args):
        seen.append(("second", args))

    args = (b"audio", "text", metadata, 1.0, "s")
    await vad._execute_on_speech_detected(*args)

    assert seen == [("first", args), ("second", args)]
    assert seen[1][1][2] is metadata


@pytest.mark.asyncio
async def test_speech_detected_exception_still_reaches_later_handlers(caplog):
    vad = SpeechDetectorDummy()
    seen = []

    @vad.on_speech_detected
    async def first(*args):
        raise ValueError("test callback failure")

    @vad.on_speech_detected
    async def second(*args):
        seen.append("second")

    await vad._execute_on_speech_detected(b"", "text", {}, 1.0, "s")

    assert seen == ["second"]
    assert "Error in on_speech_detected callback" in caplog.text


@pytest.mark.asyncio
async def test_speech_detected_cancellation_propagates_without_later_handlers():
    vad = SpeechDetectorDummy()
    entered = asyncio.Event()
    seen = []

    @vad.on_speech_detected
    async def first(*args):
        entered.set()
        await asyncio.Event().wait()

    @vad.on_speech_detected
    async def second(*args):
        seen.append("second")

    task = asyncio.create_task(vad._execute_on_speech_detected(b"", "text", {}, 1.0, "s"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert seen == []
