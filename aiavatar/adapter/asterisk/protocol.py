import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


MEDIA_SUBPROTOCOL = "media"
MAX_WEBSOCKET_MESSAGE_SIZE = 65_500

COMMANDS = frozenset({
    "ANSWER",
    "HANGUP",
    "START_MEDIA_BUFFERING",
    "STOP_MEDIA_BUFFERING",
    "FLUSH_MEDIA",
    "PAUSE_MEDIA",
    "CONTINUE_MEDIA",
    "MARK_MEDIA",
    "GET_STATUS",
    "REPORT_QUEUE_DRAINED",
    "SET_MEDIA_DIRECTION",
})

EVENTS = frozenset({
    "MEDIA_START",
    "DTMF_END",
    "MEDIA_XOFF",
    "MEDIA_XON",
    "STATUS",
    "MEDIA_BUFFERING_COMPLETED",
    "MEDIA_MARK_PROCESSED",
    "QUEUE_DRAINED",
    "HANGUP",
    "ERROR",
})


class AsteriskProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AsteriskMediaEvent:
    event: str
    channel_id: str = ""
    connection_id: str = ""
    channel: str = ""
    format: str = ""
    optimal_frame_size: int = 0
    ptime: int = 0
    correlation_id: str = ""
    digit: str = ""
    channel_variables: Dict[str, str] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


def parse_media_event(source: str) -> AsteriskMediaEvent:
    """Parse one supported Asterisk JSON control event from a TEXT frame."""

    try:
        payload = json.loads(source)
    except (TypeError, json.JSONDecodeError) as ex:
        raise AsteriskProtocolError("Control frame is not valid JSON") from ex

    if not isinstance(payload, dict):
        raise AsteriskProtocolError("Control frame must contain one JSON object")

    event = payload.get("event")
    if not isinstance(event, str) or event not in EVENTS:
        raise AsteriskProtocolError(f"Unsupported Asterisk media event: {event!r}")

    channel_variables = payload.get("channel_variables", {})
    if not isinstance(channel_variables, dict):
        raise AsteriskProtocolError("channel_variables must be an object")
    if not all(isinstance(value, str) for value in channel_variables.values()):
        raise AsteriskProtocolError("channel_variables values must be strings")

    if event == "MEDIA_START":
        required_strings = ("connection_id", "channel_id", "format")
        for name in required_strings:
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise AsteriskProtocolError(f"MEDIA_START.{name} is required")
        for name in ("optimal_frame_size", "ptime"):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise AsteriskProtocolError(f"MEDIA_START.{name} must be a positive integer")

    if event == "DTMF_END":
        digit = payload.get("digit")
        if not isinstance(digit, str) or len(digit) != 1 or digit not in "0123456789*#ABCD":
            raise AsteriskProtocolError("DTMF_END.digit is invalid")

    return AsteriskMediaEvent(
        event=event,
        channel_id=_string_value(payload.get("channel_id")),
        connection_id=_string_value(payload.get("connection_id")),
        channel=_string_value(payload.get("channel")),
        format=_string_value(payload.get("format")),
        optimal_frame_size=_integer_value(payload.get("optimal_frame_size")),
        ptime=_integer_value(payload.get("ptime")),
        correlation_id=_string_value(payload.get("correlation_id")),
        digit=_string_value(payload.get("digit")),
        channel_variables=channel_variables,
        payload=payload,
    )


def media_command(
    command: str,
    *,
    correlation_id: Optional[str] = None,
    **parameters: Any,
) -> str:
    """Serialize one Asterisk JSON command for a TEXT frame."""

    if command not in COMMANDS:
        raise AsteriskProtocolError(f"Unsupported Asterisk media command: {command!r}")
    payload: Dict[str, Any] = {"command": command}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    payload.update(parameters)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _integer_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
