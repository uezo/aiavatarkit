import asyncio
import base64
from dataclasses import replace
import logging
import math
from numbers import Real
import re
import secrets
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Set, TYPE_CHECKING
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...sts.models import STSRequest, STSResponse
from ...sts.pipeline import STSPipeline
from ..base import Adapter
from ..models import AIAvatarRequest, AIAvatarResponse
from .audio import audio_to_slin16, chunk_slin16
from .models import (
    AsteriskOperation,
    AsteriskSessionData,
    AsteriskTransferRequest,
)
from .protocol import (
    AsteriskMediaEvent,
    AsteriskProtocolError,
    MAX_WEBSOCKET_MESSAGE_SIZE,
    MEDIA_SUBPROTOCOL,
    media_command,
    parse_media_event,
)

if TYPE_CHECKING:
    from .manager import AsteriskCallManager


logger = logging.getLogger(__name__)

_MEDIA_SAMPLE_RATE = 16_000
_TRANSFER_VARIABLE_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_MAX_TRANSFER_VARIABLES = 32
_MAX_TRANSFER_VARIABLE_VALUE_LENGTH = 1024
_RESERVED_TRANSFER_VARIABLES = frozenset({
    "AIAVATAR_SESSION_ID",
    "AIAVATAR_TRANSFER_ALIAS",
    "AIAVATAR_TRANSFER_DESTINATION",
    "AIAVATAR_ORIGINAL_CALLER_NUMBER",
    "AIAVATAR_ORIGINAL_CALLER_NAME",
    "AIAVATAR_CALLER_PRESENTATION",
    "AIAVATAR_CALLER_NUMBER",
    "AIAVATAR_CALLER_NAME",
    "AIAVATAR_CALLED_NUMBER",
    "AIAVATAR_CALLER_CHANNEL_ID",
    "AIAVATAR_TRUSTED_PAI",
    "AIAVATAR_UCID",
    "AIAVATAR_UUI",
})


class AIAvatarAsteriskServer(Adapter):
    """AIAvatarKit adapter for Asterisk's Media WebSocket driver."""

    def __init__(
        self,
        *,
        sts: STSPipeline,
        tts_sample_rate: int = 24_000,
        mute_on_barge_in: bool = True,
        channel: str = "phone",
        api_username: Optional[str] = None,
        api_password: Optional[str] = None,
        media_chunk_duration_ms: int = 100,
        media_flow_timeout: float = 10.0,
        max_media_message_size: int = MAX_WEBSOCKET_MESSAGE_SIZE,
        debug: bool = False,
    ):
        if sts is None:
            raise ValueError("sts is required")
        if (api_username is None) != (api_password is None):
            raise ValueError("api_username and api_password must be configured together")
        self._validate_server_config(
            tts_sample_rate=tts_sample_rate,
            media_chunk_duration_ms=media_chunk_duration_ms,
            media_flow_timeout=media_flow_timeout,
            max_media_message_size=max_media_message_size,
            debug=debug,
        )

        super().__init__(sts)
        self.sessions: Dict[str, AsteriskSessionData] = {}
        # STS/VAD sessions are scoped to one External Media channel. This map
        # keeps delayed work from an old channel away from replacement media.
        self._pipeline_sessions: Dict[str, AsteriskSessionData] = {}
        self.tts_sample_rate = tts_sample_rate
        self.channel = channel
        self.api_username = api_username
        self.api_password = api_password
        self.media_chunk_duration_ms = media_chunk_duration_ms
        self.media_flow_timeout = media_flow_timeout
        self.max_media_message_size = max_media_message_size
        self.debug = debug
        self.last_response: Optional[AIAvatarResponse] = None
        self._call_manager: Optional["AsteriskCallManager"] = None
        # Media callbacks belong to the connection generation that created
        # them. A cleanup from an older WebSocket must never cancel callbacks
        # started by a replacement connection.
        self._session_tasks: Dict[str, Dict[int, Set[asyncio.Task]]] = {}

        self._on_connect: Optional[
            Callable[[AIAvatarRequest, AsteriskSessionData], Awaitable[None]]
        ] = None
        self._on_disconnect: Optional[
            Callable[[AsteriskSessionData], Awaitable[None]]
        ] = None
        self._on_dtmf: Optional[Callable[[str, str], Awaitable[None]]] = None
        self._on_transfer_prepare: Optional[
            Callable[
                [AsteriskTransferRequest, AsteriskSessionData],
                Awaitable[None],
            ]
        ] = None
        self._on_transfer_started: Optional[
            Callable[[str, str], Awaitable[None]]
        ] = None
        self._on_transfer_completed: Optional[
            Callable[[str, str, str], Awaitable[None]]
        ] = None
        self._on_transfer_failed: Optional[
            Callable[[str, str, str], Awaitable[None]]
        ] = None
        self._on_transfer_unknown: Optional[
            Callable[[str, str, str], Awaitable[None]]
        ] = None

        # The base parser does not register telephone operations by default.
        self.register_control_tag("operation")

        if mute_on_barge_in:
            @self.sts.vad.on_recording_started
            async def flush_on_barge_in(session_id: str):
                if self.can_handle(session_id):
                    await self.stop_response(session_id, "")

    def get_config(self) -> dict:
        return {
            "tts_sample_rate": self.tts_sample_rate,
            "media_chunk_duration_ms": self.media_chunk_duration_ms,
            "media_flow_timeout": self.media_flow_timeout,
            "debug": self.debug,
        }

    def set_config(self, config: dict) -> dict:
        """Apply runtime settings after validating the complete proposed state."""

        proposed = self.get_config()
        proposed.update({
            name: value
            for name, value in config.items()
            if name in proposed and value is not None
        })
        self._validate_server_config(
            tts_sample_rate=proposed["tts_sample_rate"],
            media_chunk_duration_ms=proposed["media_chunk_duration_ms"],
            media_flow_timeout=proposed["media_flow_timeout"],
            max_media_message_size=self.max_media_message_size,
            debug=proposed["debug"],
        )
        return super().set_config(config)

    def bind_call_manager(self, manager: "AsteriskCallManager") -> None:
        if self._call_manager is not None and self._call_manager is not manager:
            raise RuntimeError("This Asterisk adapter is already bound to a call manager")
        self._call_manager = manager

    def on_connect(
        self,
        func: Callable[[AIAvatarRequest, AsteriskSessionData], Awaitable[None]],
    ):
        self._on_connect = func
        return func

    def on_disconnect(
        self,
        func: Callable[[AsteriskSessionData], Awaitable[None]],
    ):
        self._on_disconnect = func
        return func

    def on_dtmf(self, func: Callable[[str, str], Awaitable[None]]):
        self._on_dtmf = func
        return func

    def on_transfer_prepare(
        self,
        func: Callable[
            [AsteriskTransferRequest, AsteriskSessionData],
            Awaitable[None],
        ],
    ):
        """Register a hook that adds safe Asterisk variables before transfer."""

        self._on_transfer_prepare = func
        return func

    def on_transfer_started(self, func: Callable[[str, str], Awaitable[None]]):
        self._on_transfer_started = func
        return func

    def on_transfer_completed(
        self,
        func: Callable[[str, str, str], Awaitable[None]],
    ):
        self._on_transfer_completed = func
        return func

    def on_transfer_failed(
        self,
        func: Callable[[str, str, str], Awaitable[None]],
    ):
        self._on_transfer_failed = func
        return func

    def on_transfer_unknown(
        self,
        func: Callable[[str, str, str], Awaitable[None]],
    ):
        """Register a notification for REFERs whose outcome cannot be proven."""

        self._on_transfer_unknown = func
        return func

    async def notify_transfer_started(self, session_id: str, destination: str) -> None:
        if self._on_transfer_started:
            try:
                await self._on_transfer_started(session_id, destination)
            except Exception:
                logger.exception(
                    "Asterisk transfer started callback failed: "
                    "session=%s destination=%s",
                    session_id,
                    destination,
                )

    async def notify_transfer_completed(
        self,
        session_id: str,
        destination: str,
        method: str,
    ) -> None:
        if self._on_transfer_completed:
            try:
                await self._on_transfer_completed(session_id, destination, method)
            except Exception:
                logger.exception(
                    "Asterisk transfer completed callback failed: "
                    "session=%s destination=%s method=%s",
                    session_id,
                    destination,
                    method,
                )

    async def notify_transfer_failed(
        self,
        session_id: str,
        destination: str,
        reason: str,
    ) -> None:
        if self._on_transfer_failed:
            try:
                await self._on_transfer_failed(session_id, destination, reason)
            except Exception:
                logger.exception(
                    "Asterisk transfer failed callback failed: "
                    "session=%s destination=%s reason=%s",
                    session_id,
                    destination,
                    reason,
                )

    async def notify_transfer_unknown(
        self,
        session_id: str,
        destination: str,
        reason: str,
    ) -> None:
        if self._on_transfer_unknown:
            try:
                await self._on_transfer_unknown(session_id, destination, reason)
            except Exception:
                logger.exception(
                    "Asterisk transfer outcome callback failed: "
                    "session=%s destination=%s reason=%s",
                    session_id,
                    destination,
                    reason,
                )

    async def prepare_transfer(
        self,
        session: AsteriskSessionData,
        *,
        destination_alias: str,
        destination: str,
        transfer_strategy: str,
    ) -> AsteriskTransferRequest:
        request = AsteriskTransferRequest(
            session_id=session.session_id,
            user_id=session.user_id or session.caller_number or session.session_id,
            context_id=session.context_id,
            destination_alias=destination_alias,
            destination=destination,
            transfer_strategy=transfer_strategy,
        )
        if self._on_transfer_prepare:
            await self._on_transfer_prepare(request, session)
        self._validate_transfer_variables(request.variables)
        return request

    def register_session(
        self,
        session_id: str,
        **values: Any,
    ) -> AsteriskSessionData:
        """Reserve or update a call session before MEDIA_START arrives."""

        if not session_id:
            raise ValueError("session_id is required")
        session = self.sessions.get(session_id)
        if session is None:
            session = AsteriskSessionData(session_id=session_id)
            self.sessions[session_id] = session
        for name, value in values.items():
            if not hasattr(session, name):
                raise ValueError(f"Unknown Asterisk session field: {name}")
            if value is not None:
                setattr(session, name, value)
        return session

    async def unregister_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is not None:
            session.cleanup_started = True
            try:
                await self._cleanup_media_session(session)
            finally:
                if self.sessions.get(session_id) is session:
                    self.sessions.pop(session_id, None)

    def can_handle(self, session_id: str) -> bool:
        session = self._pipeline_sessions.get(session_id)
        return bool(
            session
            and session.pipeline_session_id == session_id
            and session.media_connected
            and session.websocket
        )

    def _resolve_session(self, session_id: str) -> Optional[AsteriskSessionData]:
        """Resolve an internal media key or the stable public call key."""

        return (
            self._pipeline_sessions.get(session_id)
            or self.sessions.get(session_id)
        )

    async def handle_response(self, response: STSResponse):
        pipeline_session_id = response.session_id
        session = self._pipeline_sessions.get(pipeline_session_id)
        if (
            not session
            or session.pipeline_session_id != pipeline_session_id
            or not session.websocket
        ):
            logger.warning(
                "Session not found for response (Asterisk): %s",
                pipeline_session_id,
            )
            return

        # Keep the per-media pipeline key private. Application callbacks and
        # call control consistently observe the stable Asterisk call session ID.
        response = replace(response, session_id=session.session_id)

        response_websocket = session.websocket
        response_connection_generation = session.connection_generation

        if response.type == "accepted" and response.transaction_id:
            previous_transaction_id = session.active_transaction_id
            # Claim the transaction before the first await. Any old response
            # arriving while FLUSH_MEDIA is in flight is stale immediately.
            session.active_transaction_id = response.transaction_id
            if previous_transaction_id != response.transaction_id:
                await self.stop_response(response.session_id, response.context_id)

        if not self._is_current_response(
            session,
            response,
            websocket=response_websocket,
            connection_generation=response_connection_generation,
        ):
            if self.debug:
                logger.info(
                    "Skipped stale Asterisk response: type=%s transaction=%s active=%s",
                    response.type,
                    response.transaction_id,
                    session.active_transaction_id,
                )
            return

        if response.user_id:
            session.user_id = response.user_id
        if response.context_id:
            session.context_id = response.context_id

        metadata = dict(response.metadata or {})
        if response.tool_call is not None:
            metadata["tool_call"] = response.tool_call.to_dict()
        aiavatar_response = AIAvatarResponse(
            type=response.type,
            session_id=response.session_id,
            user_id=response.user_id,
            context_id=response.context_id,
            text=response.text,
            voice_text=response.voice_text,
            language=response.language,
            control_tags=(
                self.parse_control_tags(response.text) or None
                if response.type in ("chunk", "final")
                else None
            ),
            audio_data=response.audio_data,
            metadata=metadata,
            structured_content=response.structured_content,
        )
        for on_response in self._on_response_handlers:
            await on_response(aiavatar_response, response)
            if not self._is_current_response(
                session,
                response,
                websocket=response_websocket,
                connection_generation=response_connection_generation,
            ):
                return

        if response.type == "accepted":
            session.muted = bool(metadata.get("block_barge_in"))

        elif response.type == "start":
            session.last_mark = ""
            session.unmute_mark = ""
            session.pending_operation = None
            session.pending_operation_mark = ""

        elif response.type == "chunk":
            if metadata.get("is_guardrail_triggered"):
                await self.stop_response(response.session_id, response.context_id)
            elif response.audio_data:
                await self.send_voice(response.session_id, audio_data=response.audio_data)

        elif response.type == "final":
            operation = self._extract_operation(
                response.text,
                response.structured_content,
            )
            mark = await self._finish_utterance(session)
            if not self._is_current_response(
                session,
                response,
                websocket=response_websocket,
                connection_generation=response_connection_generation,
            ):
                return
            if session.muted:
                if mark:
                    session.unmute_mark = mark
                else:
                    session.muted = False
            if operation:
                session.pending_operation = operation
                session.pending_operation_mark = mark
                if not mark:
                    await self._execute_operation(session, operation)

        elif response.type in ("stop", "error", "canceled", "cancelled"):
            await self.stop_response(response.session_id, response.context_id)

        if self.debug:
            self.last_response = aiavatar_response

    async def send_voice(
        self,
        session_id: str,
        *,
        text: Optional[str] = None,
        audio_data: Optional[bytes] = None,
    ) -> None:
        session = self._resolve_session(session_id)
        if not session or not session.websocket:
            logger.warning("WebSocket not found for Asterisk session: %s", session_id)
            return
        if session.optimal_frame_size <= 0:
            raise RuntimeError("MEDIA_START has not supplied optimal_frame_size")

        websocket = session.websocket
        connection_generation = session.connection_generation
        playback_generation = session.playback_generation

        if audio_data is None:
            if not text:
                return
            audio_data = await self.sts.tts.synthesize(text)
            if not self._owns_media_playback(
                session,
                websocket=websocket,
                connection_generation=connection_generation,
                playback_generation=playback_generation,
            ):
                return

        samples = audio_to_slin16(
            audio_data,
            raw_sample_rate=self.tts_sample_rate,
            target_sample_rate=_MEDIA_SAMPLE_RATE,
        )
        if not samples:
            return

        if not session.buffering:
            started = await self._send_control(
                session,
                "START_MEDIA_BUFFERING",
                expected_playback_generation=playback_generation,
            )
            if (
                not started
                or not self._owns_media_playback(
                    session,
                    websocket=websocket,
                    connection_generation=connection_generation,
                    playback_generation=playback_generation,
                )
            ):
                return
            session.buffering = True

        for chunk in chunk_slin16(
            samples,
            optimal_frame_size=session.optimal_frame_size,
            sample_rate=_MEDIA_SAMPLE_RATE,
            target_duration_ms=self.media_chunk_duration_ms,
            max_message_size=self.max_media_message_size,
        ):
            if not await self._send_media_chunk(
                session,
                chunk,
                playback_generation,
            ):
                return
            session.audio_sent = True

    async def stop_response(self, session_id: str, context_id: str):
        session = self._resolve_session(session_id)
        if not session or not session.websocket:
            return

        # Revoke all local ownership before yielding to the WebSocket. MARK
        # events can arrive while FLUSH_MEDIA is in flight and must not execute
        # an operation from the interrupted response.
        session.playback_generation += 1
        session.media_writable.set()
        self._reset_response_state(session, clear_transaction=False)
        await self._send_control(session, "FLUSH_MEDIA")

    async def invoke(self, request: STSRequest):
        session = self._resolve_session(request.session_id)
        if not session or not session.pipeline_session_id:
            logger.warning(
                "Media session not found for Asterisk invoke: %s",
                request.session_id,
            )
            return
        pipeline_session_id = session.pipeline_session_id
        pipeline_request = replace(
            request,
            session_id=pipeline_session_id,
            channel=self.channel,
        )
        try:
            async for response in self.sts.invoke(pipeline_request):
                await self.sts.handle_response(response)
                if response.context_id:
                    self.sts.vad.set_session_data(
                        pipeline_session_id,
                        "context_id",
                        response.context_id,
                    )
        except Exception:
            logger.exception("Error invoking the Asterisk pipeline")

    async def process_control_frame(
        self,
        websocket: WebSocket,
        source: str,
        session: Optional[AsteriskSessionData] = None,
    ) -> Optional[AsteriskSessionData]:
        event = parse_media_event(source)
        if event.event == "MEDIA_START":
            return await self._start_media_session(websocket, event)
        if session is None:
            raise AsteriskProtocolError(
                f"{event.event} was received before MEDIA_START"
            )
        if session.websocket is not websocket or not session.media_connected:
            raise AsteriskProtocolError("Media frame arrived on a superseded connection")

        if event.channel_id and event.channel_id != session.media_channel_id:
            raise AsteriskProtocolError("Control event channel_id changed during a session")

        if event.event == "MEDIA_XOFF":
            session.flow_blocked = True
            session.flow_timeout_expired = False
            session.media_writable.clear()
        elif event.event == "MEDIA_XON":
            session.flow_blocked = False
            session.flow_timeout_expired = False
            session.media_writable.set()
        elif event.event == "DTMF_END":
            if self._on_dtmf:
                self._spawn_session_task(
                    session.session_id,
                    session.connection_generation,
                    self._on_dtmf(event.digit, session.session_id),
                    name=f"aiavatar-asterisk-dtmf-{session.session_id}",
                )
        elif event.event == "MEDIA_MARK_PROCESSED":
            await self._handle_mark_processed(session, event.correlation_id)
        elif event.event == "STATUS":
            session.data["media_status"] = event.payload
        elif event.event == "ERROR":
            logger.error("Asterisk Media WebSocket error: %s", event.payload)
        elif event.event == "HANGUP":
            await self._cleanup_media_session(
                session,
                expected_websocket=websocket,
            )
        return session

    async def process_binary_frame(
        self,
        websocket: WebSocket,
        session: Optional[AsteriskSessionData],
        samples: bytes,
    ) -> None:
        if session is None:
            raise AsteriskProtocolError("BINARY media was received before MEDIA_START")
        if not session.media_connected or session.websocket is not websocket:
            return
        if len(samples) > self.max_media_message_size:
            raise AsteriskProtocolError("BINARY media exceeds the configured message limit")
        if session.muted or not samples:
            return
        if len(samples) % 2:
            raise AsteriskProtocolError("slin16 input is not 16-bit aligned")
        await self.sts.vad.process_samples(samples, session.pipeline_session_id)

    def get_router(self, path: str = "/asterisk/media") -> APIRouter:
        if not path.startswith("/"):
            raise ValueError("WebSocket path must start with '/'")
        router = APIRouter()

        @router.websocket(path)
        async def media_websocket(websocket: WebSocket):
            try:
                self._authenticate_websocket(websocket)
            except AsteriskProtocolError as ex:
                logger.warning("Rejected Asterisk Media WebSocket: %s", ex)
                await websocket.close(code=1008)
                return

            await websocket.accept(subprotocol=MEDIA_SUBPROTOCOL)
            session: Optional[AsteriskSessionData] = None
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    if message.get("text") is not None:
                        session = await self.process_control_frame(
                            websocket,
                            message["text"],
                            session,
                        )
                        if (
                            session is not None
                            and (
                                not session.media_connected
                                or session.websocket is not websocket
                            )
                        ):
                            break
                    elif message.get("bytes") is not None:
                        await self.process_binary_frame(
                            websocket,
                            session,
                            message["bytes"],
                        )
            except WebSocketDisconnect:
                pass
            except AsteriskProtocolError as ex:
                logger.warning("Asterisk Media WebSocket protocol error: %s", ex)
                await websocket.close(code=1003)
            finally:
                if session is not None:
                    await self._cleanup_media_session(
                        session,
                        expected_websocket=websocket,
                    )

        return router

    async def _start_media_session(
        self,
        websocket: WebSocket,
        event: AsteriskMediaEvent,
    ) -> AsteriskSessionData:
        if event.format.lower() != "slin16":
            raise AsteriskProtocolError(
                f"Expected slin16 media but Asterisk selected {event.format!r}"
            )
        if event.optimal_frame_size > self.max_media_message_size:
            raise AsteriskProtocolError(
                "MEDIA_START.optimal_frame_size exceeds the configured message limit"
            )

        variables = event.channel_variables
        query_session_id = websocket.query_params.get("session_id")
        variable_session_id = self._channel_variable(
            variables,
            "AIAVATAR_SESSION_ID",
        )
        if (
            query_session_id
            and variable_session_id
            and query_session_id != variable_session_id
        ):
            raise AsteriskProtocolError(
                "Media session_id does not match the channel variables"
            )
        session_id = query_session_id or variable_session_id
        if not session_id:
            raise AsteriskProtocolError(
                "MEDIA_START requires the AIAVATAR_SESSION_ID channel variable "
                "or session_id URI parameter"
            )
        existing = self.sessions.get(session_id)
        if existing is None:
            raise AsteriskProtocolError(
                "Media connection did not match a pre-registered session"
            )
        if existing.cleanup_started:
            raise AsteriskProtocolError("The call session is already closing")
        if existing.media_connected:
            raise AsteriskProtocolError(
                "A duplicate MEDIA_START used the same session_id"
            )
        if existing.connected_media_channel_id == event.channel_id:
            raise AsteriskProtocolError(
                "Media WebSocket reconnection is not supported; "
                "a replacement requires a new manager-registered media channel"
            )
        if (
            not existing.media_channel_id
            or existing.media_channel_id != event.channel_id
        ):
            raise AsteriskProtocolError(
                "Media channel_id does not match the manager-registered session"
            )
        pipeline_owner = self._pipeline_sessions.get(event.channel_id)
        if pipeline_owner is not None and pipeline_owner is not existing:
            raise AsteriskProtocolError(
                "Media channel_id is already owned by another session"
            )

        session = self.register_session(
            session_id,
            media_channel_id=event.channel_id,
            media_connection_id=event.connection_id,
            websocket=websocket,
            optimal_frame_size=event.optimal_frame_size,
            ptime=event.ptime,
            channel_variables=dict(variables),
        )
        session.caller_number = session.caller_number or self._channel_variable(
            variables,
            "AIAVATAR_CALLER_NUMBER",
        )
        session.caller_name = session.caller_name or self._channel_variable(
            variables,
            "AIAVATAR_CALLER_NAME",
        )
        session.caller_presentation = self._channel_variable(
            variables,
            "AIAVATAR_CALLER_PRESENTATION",
        ) or session.caller_presentation
        session.called_number = session.called_number or self._channel_variable(
            variables,
            "AIAVATAR_CALLED_NUMBER",
        )
        session.ari_caller_channel_id = (
            session.ari_caller_channel_id
            or self._channel_variable(variables, "AIAVATAR_CALLER_CHANNEL_ID")
        )
        session.connection_generation += 1
        connection_generation = session.connection_generation
        session.playback_generation += 1
        session.connected_media_channel_id = event.channel_id
        session.pipeline_session_id = event.channel_id
        self._pipeline_sessions[event.channel_id] = session
        session.media_connected = True
        session.media_cleanup_started = False
        session.flow_blocked = False
        session.flow_timeout_expired = False
        self._reset_response_state(session, clear_transaction=True)
        session.media_writable.set()

        try:
            request = AIAvatarRequest(
                type="start",
                session_id=session.session_id,
                user_id=session.user_id or session.caller_number or session.session_id,
                context_id=session.context_id or None,
                metadata={
                    "caller_name": session.caller_name,
                    "caller_presentation": session.caller_presentation,
                    "called_number": session.called_number,
                },
            )
            for on_session_start in self._on_session_start_handlers:
                await on_session_start(request, session)
                self._require_media_ownership(
                    session,
                    websocket,
                    connection_generation,
                )

            session.user_id = request.user_id or session.caller_number or session.session_id
            session.context_id = request.context_id or session.context_id

            self.sts.vad.set_session_data(
                session.pipeline_session_id,
                "user_id",
                request.user_id,
                True,
            )
            if request.context_id:
                self.sts.vad.set_session_data(
                    session.pipeline_session_id,
                    "context_id",
                    request.context_id,
                    True,
                )
            self.sts.vad.set_session_data(
                session.pipeline_session_id,
                "channel",
                self.channel,
                True,
            )
            if request.system_prompt_params:
                self.sts.vad.set_session_data(
                    session.pipeline_session_id,
                    "system_prompt_params",
                    request.system_prompt_params,
                    True,
                )
            if self._on_connect:
                self._spawn_session_task(
                    session.session_id,
                    connection_generation,
                    self._on_connect(request, session),
                    name=f"aiavatar-asterisk-connect-{session.session_id}",
                )

            if self.debug:
                logger.info(
                    "Asterisk media connected: session=%s channel=%s frame=%s ptime=%s",
                    session.session_id,
                    session.media_channel_id,
                    session.optimal_frame_size,
                    session.ptime,
                )
            if self._call_manager is not None:
                self._require_media_ownership(
                    session,
                    websocket,
                    connection_generation,
                )
                await self._call_manager.media_connected(
                    session.session_id,
                    session.media_channel_id,
                )
                self._require_media_ownership(
                    session,
                    websocket,
                    connection_generation,
                )
        except BaseException:
            await self._cleanup_media_session(
                session,
                expected_websocket=websocket,
            )
            raise
        return session

    async def _finish_utterance(self, session: AsteriskSessionData) -> str:
        if not session.buffering:
            return ""

        generation = session.playback_generation
        buffering_id = str(uuid4())
        stopped = await self._send_control(
            session,
            "STOP_MEDIA_BUFFERING",
            correlation_id=buffering_id,
            expected_playback_generation=generation,
        )
        if not stopped or generation != session.playback_generation:
            return ""
        session.buffering = False
        if not session.audio_sent:
            return ""

        mark = f"{session.session_id}-{uuid4()}"
        marked = await self._send_control(
            session,
            "MARK_MEDIA",
            correlation_id=mark,
            expected_playback_generation=generation,
        )
        if not marked or generation != session.playback_generation:
            return ""
        session.audio_sent = False
        session.last_mark = mark
        return mark

    async def _send_control(
        self,
        session: AsteriskSessionData,
        command: str,
        *,
        correlation_id: Optional[str] = None,
        expected_playback_generation: Optional[int] = None,
        **parameters: Any,
    ) -> bool:
        websocket = session.websocket
        if websocket is None:
            return False
        source = media_command(
            command,
            correlation_id=correlation_id,
            **parameters,
        )
        async with session.send_lock:
            if (
                expected_playback_generation is not None
                and expected_playback_generation != session.playback_generation
            ):
                return False
            if session.websocket is None:
                return False
            if session.websocket is not websocket:
                return False
            await websocket.send_text(source)
        return True

    async def _send_media_chunk(
        self,
        session: AsteriskSessionData,
        chunk: bytes,
        playback_generation: int,
    ) -> bool:
        if session.flow_timeout_expired:
            return False
        while session.flow_blocked:
            session.media_writable.clear()
            try:
                await asyncio.wait_for(
                    session.media_writable.wait(),
                    timeout=self.media_flow_timeout,
                )
            except asyncio.TimeoutError:
                if (
                    playback_generation == session.playback_generation
                    and session.flow_blocked
                ):
                    session.flow_timeout_expired = True
                    logger.warning(
                        "Asterisk media remained XOFF beyond %.1fs: session=%s",
                        self.media_flow_timeout,
                        session.session_id,
                    )
                return False
            if playback_generation != session.playback_generation:
                return False

        if (
            playback_generation != session.playback_generation
            or session.websocket is None
        ):
            return False
        async with session.send_lock:
            if (
                playback_generation != session.playback_generation
                or session.flow_blocked
                or session.websocket is None
            ):
                return False
            await session.websocket.send_bytes(chunk)
        return True

    async def _handle_mark_processed(
        self,
        session: AsteriskSessionData,
        mark: str,
    ) -> None:
        if not mark:
            return
        if mark == session.last_mark:
            session.last_mark = ""
        if mark == session.unmute_mark:
            session.unmute_mark = ""
            session.muted = False
        if mark == session.pending_operation_mark and session.pending_operation:
            operation = session.pending_operation
            session.pending_operation = None
            session.pending_operation_mark = ""
            await self._execute_operation(session, operation)

    async def _execute_operation(
        self,
        session: AsteriskSessionData,
        operation: AsteriskOperation,
    ) -> None:
        if operation.name == "hangup":
            if self._call_manager:
                await self._call_manager.hangup(session.session_id)
            else:
                await self._send_control(session, "HANGUP")
            return

        if operation.name == "transfer":
            if not operation.destination:
                await self.notify_transfer_failed(
                    session.session_id,
                    "",
                    "missing_destination",
                )
            elif self._call_manager:
                await self._call_manager.transfer(
                    session.session_id,
                    operation.destination,
                )
            else:
                await self.notify_transfer_failed(
                    session.session_id,
                    operation.destination,
                    "call_manager_not_configured",
                )

    def _extract_operation(
        self,
        text: Optional[str],
        structured_content: Optional[Mapping[str, Any]],
    ) -> Optional[AsteriskOperation]:
        candidates = []
        if text:
            for tag in self.parse_control_tags(text):
                if tag.name == "operation":
                    candidates.append(tag.attributes)

        if isinstance(structured_content, Mapping):
            structured_operation = structured_content.get("operation")
            if isinstance(structured_operation, Mapping):
                candidates.append(structured_operation)

        for candidate in candidates:
            name = str(candidate.get("name", "")).strip().lower()
            if name == "hangup":
                return AsteriskOperation(name="hangup")
            if name == "transfer":
                destination = str(candidate.get("destination", "")).strip()
                return AsteriskOperation(
                    name="transfer",
                    destination=destination or None,
                )
        return None

    def _authenticate_websocket(self, websocket: WebSocket) -> None:
        protocols = {
            value.strip()
            for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if value.strip()
        }
        if MEDIA_SUBPROTOCOL not in protocols:
            raise AsteriskProtocolError("The 'media' WebSocket subprotocol is required")

        if self.api_username is None:
            return
        authorization = websocket.headers.get("authorization", "")
        if not authorization.lower().startswith("basic "):
            raise AsteriskProtocolError("Basic authentication is required")
        try:
            decoded = base64.b64decode(
                authorization.split(" ", 1)[1],
                validate=True,
            ).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as ex:
            raise AsteriskProtocolError("Invalid Basic authentication") from ex
        supplied_username, separator, supplied_password = decoded.partition(":")
        valid = bool(separator)
        valid &= secrets.compare_digest(
            supplied_username.encode("utf-8"),
            (self.api_username or "").encode("utf-8"),
        )
        valid &= secrets.compare_digest(
            supplied_password.encode("utf-8"),
            (self.api_password or "").encode("utf-8"),
        )
        if not valid:
            raise AsteriskProtocolError("Invalid Basic authentication")

    def _spawn_session_task(
        self,
        session_id: str,
        connection_generation: int,
        awaitable: Awaitable[Any],
        *,
        name: str,
    ) -> None:
        task = asyncio.create_task(awaitable, name=name)
        generations = self._session_tasks.setdefault(session_id, {})
        tasks = generations.setdefault(connection_generation, set())
        tasks.add(task)

        def task_done(done_task: asyncio.Task) -> None:
            session_generations = self._session_tasks.get(session_id)
            if session_generations is not None:
                generation_tasks = session_generations.get(connection_generation)
                if generation_tasks is not None:
                    generation_tasks.discard(done_task)
                    if not generation_tasks:
                        session_generations.pop(connection_generation, None)
                if not session_generations:
                    self._session_tasks.pop(session_id, None)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Asterisk session callback failed: session=%s task=%s",
                    session_id,
                    name,
                )

        task.add_done_callback(task_done)

    async def _cancel_session_tasks(
        self,
        session_id: str,
        connection_generation: int,
    ) -> None:
        generations = self._session_tasks.get(session_id)
        if generations is None:
            return
        tasks = generations.pop(connection_generation, set())
        if not generations:
            self._session_tasks.pop(session_id, None)
        current_task = asyncio.current_task()
        pending = [task for task in tasks if task is not current_task and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _reset_response_state(
        session: AsteriskSessionData,
        *,
        clear_transaction: bool,
    ) -> None:
        """Drop response state owned by the current playback generation."""

        session.buffering = False
        session.audio_sent = False
        session.muted = False
        session.last_mark = ""
        session.unmute_mark = ""
        session.pending_operation_mark = ""
        session.pending_operation = None
        if clear_transaction:
            session.active_transaction_id = ""

    def _require_media_ownership(
        self,
        session: AsteriskSessionData,
        websocket: WebSocket,
        connection_generation: int,
    ) -> None:
        """Reject startup work after its Media WebSocket lost ownership."""

        if not self._owns_media_connection(
            session,
            websocket=websocket,
            connection_generation=connection_generation,
        ):
            raise AsteriskProtocolError(
                "Media connection closed while session startup was in progress"
            )

    def _owns_media_connection(
        self,
        session: AsteriskSessionData,
        *,
        websocket: WebSocket,
        connection_generation: int,
    ) -> bool:
        return (
            self.sessions.get(session.session_id) is session
            and not session.cleanup_started
            and session.media_connected
            and session.websocket is websocket
            and session.connection_generation == connection_generation
        )

    def _owns_media_playback(
        self,
        session: AsteriskSessionData,
        *,
        websocket: WebSocket,
        connection_generation: int,
        playback_generation: int,
    ) -> bool:
        return (
            self._owns_media_connection(
                session,
                websocket=websocket,
                connection_generation=connection_generation,
            )
            and session.playback_generation == playback_generation
        )

    def _is_current_response(
        self,
        session: AsteriskSessionData,
        response: STSResponse,
        *,
        websocket: WebSocket,
        connection_generation: int,
    ) -> bool:
        return (
            self._owns_media_connection(
                session,
                websocket=websocket,
                connection_generation=connection_generation,
            )
            and (
                response.transaction_id == session.active_transaction_id
                if response.transaction_id
                else not session.active_transaction_id
            )
        )

    @staticmethod
    def _validate_server_config(
        *,
        tts_sample_rate: Any,
        media_chunk_duration_ms: Any,
        media_flow_timeout: Any,
        max_media_message_size: Any,
        debug: Any,
    ) -> None:
        if (
            isinstance(tts_sample_rate, bool)
            or not isinstance(tts_sample_rate, int)
            or tts_sample_rate <= 0
        ):
            raise ValueError("tts_sample_rate must be a positive integer")
        if (
            isinstance(media_chunk_duration_ms, bool)
            or not isinstance(media_chunk_duration_ms, int)
            or media_chunk_duration_ms <= 0
        ):
            raise ValueError("media_chunk_duration_ms must be a positive integer")
        if (
            isinstance(media_flow_timeout, bool)
            or not isinstance(media_flow_timeout, Real)
            or not math.isfinite(media_flow_timeout)
            or media_flow_timeout <= 0
        ):
            raise ValueError("media_flow_timeout must be a positive finite number")
        if (
            isinstance(max_media_message_size, bool)
            or not isinstance(max_media_message_size, int)
            or not 1 <= max_media_message_size <= MAX_WEBSOCKET_MESSAGE_SIZE
        ):
            raise ValueError(
                "max_media_message_size must be an integer between 1 and "
                f"{MAX_WEBSOCKET_MESSAGE_SIZE}"
            )
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")

    @staticmethod
    def _validate_transfer_variables(variables: Mapping[str, str]) -> None:
        if not isinstance(variables, Mapping):
            raise ValueError("transfer variables must be a mapping")
        if len(variables) > _MAX_TRANSFER_VARIABLES:
            raise ValueError(
                f"transfer variables cannot contain more than {_MAX_TRANSFER_VARIABLES} entries"
            )
        for name, value in variables.items():
            if not isinstance(name, str) or not _TRANSFER_VARIABLE_NAME_PATTERN.fullmatch(name):
                raise ValueError(
                    "transfer variable names must contain only uppercase letters, "
                    "digits, and underscores"
                )
            if name in _RESERVED_TRANSFER_VARIABLES:
                raise ValueError(f"transfer variable is reserved: {name}")
            if not isinstance(value, str):
                raise ValueError(f"transfer variable values must be strings: {name}")
            if len(value) > _MAX_TRANSFER_VARIABLE_VALUE_LENGTH:
                raise ValueError(
                    f"transfer variable value is too long: {name}"
                )
            if any(character in value for character in ("\r", "\n", "\x00")):
                raise ValueError(
                    f"transfer variable value contains a control character: {name}"
                )

    async def _cleanup_media_session(
        self,
        session: AsteriskSessionData,
        *,
        expected_websocket: Any = None,
    ) -> None:
        full_cleanup = expected_websocket is None
        cancellation: Optional[asyncio.CancelledError] = None

        while True:
            cleanup_task = session.media_cleanup_task
            cleanup_websocket = session.media_cleanup_websocket

            if cleanup_task is not None and cleanup_task.done():
                self._clear_media_cleanup_task(session, cleanup_task)
                cleanup_task = None
                cleanup_websocket = None

            if (
                expected_websocket is not None
                and session.websocket is not expected_websocket
            ):
                if (
                    cleanup_task is not None
                    and cleanup_websocket is expected_websocket
                ):
                    await self._await_cleanup_task(cleanup_task)
                    self._clear_media_cleanup_task(session, cleanup_task)
                break

            if cleanup_task is not None:
                same_websocket_cleanup = cleanup_websocket is expected_websocket
                try:
                    await self._await_cleanup_task(cleanup_task)
                except asyncio.CancelledError as ex:
                    if not full_cleanup and same_websocket_cleanup:
                        raise
                    cancellation = ex
                self._clear_media_cleanup_task(session, cleanup_task)
                if full_cleanup or not same_websocket_cleanup:
                    # A replacement WebSocket may have connected while the old
                    # generation's disconnect hook was running. Re-evaluate
                    # ownership instead of applying the old result to it.
                    continue
                break

            if session.media_cleanup_started:
                break

            cleanup_websocket = session.websocket
            connection_generation = session.connection_generation
            pipeline_session_id = session.pipeline_session_id
            if self._pipeline_sessions.get(pipeline_session_id) is session:
                self._pipeline_sessions.pop(pipeline_session_id, None)
            if session.pipeline_session_id == pipeline_session_id:
                session.pipeline_session_id = ""
            session.media_cleanup_started = True
            session.media_connected = False
            session.playback_generation += 1
            session.websocket = None
            session.flow_blocked = False
            session.flow_timeout_expired = False
            self._reset_response_state(session, clear_transaction=True)
            session.media_writable.set()

            cleanup_task = asyncio.create_task(
                self._run_media_cleanup(
                    session,
                    connection_generation,
                    pipeline_session_id=pipeline_session_id,
                    cleanup_websocket=cleanup_websocket,
                    close_websocket=full_cleanup,
                ),
                name=f"aiavatar-asterisk-media-cleanup-{session.session_id}",
            )
            session.media_cleanup_task = cleanup_task
            session.media_cleanup_websocket = cleanup_websocket
            try:
                await self._await_cleanup_task(cleanup_task)
            except asyncio.CancelledError as ex:
                if not full_cleanup:
                    raise
                cancellation = ex
            finally:
                self._clear_media_cleanup_task(session, cleanup_task)

            if full_cleanup:
                continue
            break

        if cancellation is not None:
            raise cancellation

    async def _run_media_cleanup(
        self,
        session: AsteriskSessionData,
        connection_generation: int,
        *,
        pipeline_session_id: str,
        cleanup_websocket: Any,
        close_websocket: bool,
    ) -> None:
        await self._cancel_session_tasks(
            session.session_id,
            connection_generation,
        )
        if close_websocket and cleanup_websocket is not None:
            try:
                await cleanup_websocket.close()
            except Exception:
                logger.exception(
                    "Failed to close Asterisk Media WebSocket: session=%s",
                    session.session_id,
                )
        try:
            if self._on_disconnect:
                await self._on_disconnect(session)
        except asyncio.CancelledError:
            logger.warning(
                "Asterisk on_disconnect callback was canceled: session=%s",
                session.session_id,
            )
        except Exception:
            logger.exception(
                "Asterisk on_disconnect callback failed: session=%s",
                session.session_id,
            )
        try:
            if pipeline_session_id:
                await self.sts.finalize(pipeline_session_id)
        except asyncio.CancelledError:
            logger.warning(
                "Asterisk pipeline finalization was canceled: session=%s media=%s",
                session.session_id,
                pipeline_session_id,
            )
        except Exception:
            logger.exception(
                "Asterisk pipeline finalization failed: session=%s media=%s",
                session.session_id,
                pipeline_session_id,
            )
        finally:
            if (
                self._call_manager is None
                and self.sessions.get(session.session_id) is session
                and session.connection_generation == connection_generation
                and session.websocket is None
            ):
                self.sessions.pop(session.session_id, None)

    @staticmethod
    def _clear_media_cleanup_task(
        session: AsteriskSessionData,
        cleanup_task: asyncio.Task,
    ) -> None:
        if session.media_cleanup_task is cleanup_task and cleanup_task.done():
            session.media_cleanup_task = None
            session.media_cleanup_websocket = None

    @staticmethod
    async def _await_cleanup_task(task: asyncio.Task) -> None:
        cancellation: Optional[asyncio.CancelledError] = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as ex:
                cancellation = ex
        task.result()
        if cancellation is not None:
            raise cancellation

    @staticmethod
    def _channel_variable(variables: Mapping[str, str], name: str) -> str:
        return variables.get(name, "")
