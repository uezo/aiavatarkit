import pytest

from aiavatar.adapter.asterisk.registry import AsteriskCallRegistry
from aiavatar.adapter.asterisk.models import AsteriskSessionData


def _session(
    session_id: str,
    caller_id: str,
    *,
    media_id: str = "",
    destination_id: str = "",
) -> AsteriskSessionData:
    return AsteriskSessionData(
        session_id=session_id,
        ari_caller_channel_id=caller_id,
        media_channel_id=media_id,
        destination_channel_id=destination_id,
    )


def test_register_bind_rebind_and_remove_are_consistent():
    registry = AsteriskCallRegistry()
    session = _session("call-1", "caller-1", media_id="media-1")

    registry.register(session)
    registry.bind_media(session, "media-2")
    registry.bind_destination(session, "destination-1")

    assert registry.get("call-1") is session
    assert registry.by_caller("caller-1") == "call-1"
    assert registry.by_media("media-1") is None
    assert registry.by_media("media-2") == "call-1"
    assert registry.by_destination("destination-1") == "call-1"
    assert session.media_channel_id == "media-2"

    assert registry.remove("call-1") is session
    assert registry.get("call-1") is None
    assert registry.by_caller("caller-1") is None
    assert registry.by_media("media-2") is None
    assert registry.by_destination("destination-1") is None


def test_failed_rebind_keeps_existing_channel_index():
    registry = AsteriskCallRegistry()
    first = _session("call-1", "caller-1", media_id="media-1")
    second = _session("call-2", "caller-2", media_id="media-2")
    registry.register(first)
    registry.register(second)

    with pytest.raises(ValueError, match="belongs to another session"):
        registry.bind_media(first, "media-2")

    assert first.media_channel_id == "media-1"
    assert registry.by_media("media-1") == "call-1"
    assert registry.by_media("media-2") == "call-2"


def test_failed_registration_does_not_leave_partial_session():
    registry = AsteriskCallRegistry()
    registry.register(_session("call-1", "caller-1"))

    with pytest.raises(ValueError, match="belongs to another session"):
        registry.register(_session("call-2", "caller-1"))

    assert registry.get("call-2") is None
    assert registry.by_caller("caller-1") == "call-1"


def test_sessions_view_is_read_only():
    registry = AsteriskCallRegistry()
    registry.register(_session("call-1", "caller-1"))

    with pytest.raises(TypeError):
        registry.sessions["call-2"] = _session("call-2", "caller-2")
