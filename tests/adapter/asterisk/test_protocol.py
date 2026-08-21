import json

import pytest

from aiavatar.adapter.asterisk.protocol import (
    AsteriskProtocolError,
    media_command,
    parse_media_event,
)


def test_parse_media_start_json():
    event = parse_media_event(json.dumps({
        "event": "MEDIA_START",
        "connection_id": "connection-1",
        "channel": "WebSocket/media",
        "channel_id": "media-1",
        "format": "slin16",
        "optimal_frame_size": 640,
        "ptime": 20,
        "channel_variables": {"AIAVATAR_SESSION_ID": "call-1"},
    }))

    assert event.event == "MEDIA_START"
    assert event.optimal_frame_size == 640
    assert event.channel_variables["AIAVATAR_SESSION_ID"] == "call-1"


def test_media_command_uses_json_control_shape():
    assert json.loads(media_command(
        "MARK_MEDIA",
        correlation_id="mark-1",
    )) == {
        "command": "MARK_MEDIA",
        "correlation_id": "mark-1",
    }


def test_channel_variable_values_must_use_current_json_string_shape():
    source = json.dumps({
        "event": "MEDIA_START",
        "connection_id": "connection-1",
        "channel_id": "media-1",
        "format": "slin16",
        "optimal_frame_size": 640,
        "ptime": 20,
        "channel_variables": {
            "AIAVATAR_SESSION_ID": {"value": "call-1"},
        },
    })

    with pytest.raises(AsteriskProtocolError, match="values must be strings"):
        parse_media_event(source)


@pytest.mark.parametrize("source", [
    "MEDIA_START connection_id:1",
    "[]",
    '{"event":"media_start"}',
    '{"event":"MEDIA_START","connection_id":"c"}',
    '{"event":"DTMF_END","digit":"12"}',
])
def test_invalid_control_event_is_rejected(source):
    with pytest.raises(AsteriskProtocolError):
        parse_media_event(source)


def test_unknown_command_is_rejected():
    with pytest.raises(AsteriskProtocolError):
        media_command("flush_media")
