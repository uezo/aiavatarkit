import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class AsteriskOperation:
    """A validated operation requested by an AI response."""

    name: str
    destination: Optional[str] = None


@dataclass(slots=True)
class AsteriskTransferRequest:
    """Trusted transfer metadata that an application may enrich before ARI."""

    session_id: str
    user_id: str
    context_id: str
    destination_alias: str
    destination: str
    transfer_strategy: str
    variables: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AsteriskSessionData:
    """State shared by the Media WebSocket adapter and the call manager."""

    session_id: str
    ari_caller_channel_id: str = ""
    media_channel_id: str = ""
    media_connection_id: str = ""
    # One WebSocket is accepted per media channel. Manager-owned recovery
    # creates and pre-registers a new channel ID instead of reconnecting one.
    connected_media_channel_id: str = ""
    # Internal STS/VAD ownership key for the current media lifecycle. Public
    # callbacks and call control continue to use session_id.
    pipeline_session_id: str = ""
    bridge_id: str = ""
    caller_number: str = ""
    caller_name: str = ""
    user_id: str = ""
    context_id: str = ""
    caller_presentation: str = "allowed"
    called_number: str = ""
    trusted_pai: str = ""
    ucid: str = ""
    uui: str = ""
    websocket: Any = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    media_writable: asyncio.Event = field(default_factory=asyncio.Event)
    optimal_frame_size: int = 0
    ptime: int = 20
    buffering: bool = False
    audio_sent: bool = False
    flow_blocked: bool = False
    flow_timeout_expired: bool = False
    muted: bool = False
    last_mark: str = ""
    unmute_mark: str = ""
    pending_operation_mark: str = ""
    pending_operation: Optional[AsteriskOperation] = None
    channel_variables: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    # Connection ownership and outbound playback cancellation are independent.
    # Replacing a WebSocket advances connection_generation; interrupting audio
    # advances playback_generation without changing callback ownership.
    connection_generation: int = 0
    playback_generation: int = 0
    media_connected: bool = False
    media_cleanup_started: bool = False
    media_cleanup_task: Any = None
    media_cleanup_websocket: Any = None
    cleanup_started: bool = False
    active_transaction_id: str = ""

    # ARI transfer data. transfer_state is a read-only mirror of the call actor.
    transfer_alias: str = ""
    transfer_destination: str = ""
    transfer_method: str = ""
    transfer_state: str = "active"
    transfer_failure_reason: str = ""
    transfer_variables: Dict[str, str] = field(default_factory=dict)
    destination_channel_id: str = ""
    holding_bridge_id: str = ""

    def __post_init__(self) -> None:
        self.media_writable.set()
