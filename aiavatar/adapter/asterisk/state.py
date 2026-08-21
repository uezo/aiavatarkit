from dataclasses import dataclass
from enum import Enum


class AsteriskCallState(str, Enum):
    """Lifecycle states owned by one :class:`AsteriskCallActor`."""

    ACTIVE = "active"
    REFER_PENDING = "refer_pending"
    REFER_FAILED = "refer_failed"
    REFER_UNKNOWN = "refer_unknown"
    REFER_COMPLETED = "refer_completed"
    BRIDGE_DIALING = "bridge_dialing"
    BRIDGE_COMPLETED = "bridge_completed"
    RESTORING_AI = "restoring_ai"
    CLEANING_UP = "cleaning_up"
    CLOSED = "closed"


CALL_STATE_TRANSITIONS = {
    AsteriskCallState.ACTIVE: frozenset({
        AsteriskCallState.REFER_PENDING,
        AsteriskCallState.BRIDGE_DIALING,
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.REFER_PENDING: frozenset({
        AsteriskCallState.REFER_FAILED,
        AsteriskCallState.REFER_UNKNOWN,
        AsteriskCallState.REFER_COMPLETED,
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.REFER_FAILED: frozenset({
        AsteriskCallState.REFER_PENDING,
        AsteriskCallState.BRIDGE_DIALING,
        AsteriskCallState.RESTORING_AI,
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.REFER_COMPLETED: frozenset({
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.REFER_UNKNOWN: frozenset({
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.BRIDGE_DIALING: frozenset({
        AsteriskCallState.BRIDGE_COMPLETED,
        AsteriskCallState.RESTORING_AI,
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.BRIDGE_COMPLETED: frozenset({
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.RESTORING_AI: frozenset({
        AsteriskCallState.ACTIVE,
        AsteriskCallState.CLEANING_UP,
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.CLEANING_UP: frozenset({
        AsteriskCallState.CLOSED,
    }),
    AsteriskCallState.CLOSED: frozenset(),
}


class AsteriskCallEvent:
    """Marker base class for typed per-call lifecycle events."""


@dataclass(frozen=True, slots=True)
class TransferRequested(AsteriskCallEvent):
    destination_alias: str


@dataclass(frozen=True, slots=True)
class HangupRequested(AsteriskCallEvent):
    hangup_caller: bool = True


@dataclass(frozen=True, slots=True)
class StasisEnded(AsteriskCallEvent):
    pass


@dataclass(frozen=True, slots=True)
class ReferFailed(AsteriskCallEvent):
    pass


@dataclass(frozen=True, slots=True)
class ReferCompleted(AsteriskCallEvent):
    pass


@dataclass(frozen=True, slots=True)
class ReferTimedOut(AsteriskCallEvent):
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class DestinationAnswered(AsteriskCallEvent):
    channel_id: str


@dataclass(frozen=True, slots=True)
class DestinationFailed(AsteriskCallEvent):
    reason: str


@dataclass(frozen=True, slots=True)
class DestinationDestroyed(AsteriskCallEvent):
    channel_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class MediaDestroyed(AsteriskCallEvent):
    channel_id: str


@dataclass(frozen=True, slots=True)
class MediaConnected(AsteriskCallEvent):
    channel_id: str


@dataclass(frozen=True, slots=True)
class MediaRestoreTimedOut(AsteriskCallEvent):
    channel_id: str


@dataclass(frozen=True, slots=True)
class CallerDestroyed(AsteriskCallEvent):
    reason: str = ""
