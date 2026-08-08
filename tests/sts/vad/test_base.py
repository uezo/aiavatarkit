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
