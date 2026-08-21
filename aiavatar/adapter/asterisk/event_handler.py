import asyncio
import logging
from typing import Any, Mapping, Protocol

from .ari_client import AsteriskARIClient
from .registry import AsteriskCallRegistry
from .service import AsteriskCallService
from .state import (
    AsteriskCallEvent,
    CallerDestroyed,
    DestinationAnswered,
    DestinationDestroyed,
    DestinationFailed,
    MediaDestroyed,
    ReferCompleted,
    ReferFailed,
    StasisEnded,
)


logger = logging.getLogger(__name__)

DIAL_FAILURE_STATUSES = frozenset({
    "BUSY",
    "CANCEL",
    "CHANUNAVAIL",
    "CONGESTION",
    "DONTCALL",
    "INVALIDARGS",
    "NOANSWER",
    "TORTURE",
})


class CallEventDispatcher(Protocol):
    async def __call__(
        self,
        session_id: str,
        event: AsteriskCallEvent,
        *,
        wait: bool,
    ) -> bool:
        ...


class AsteriskARIEventHandler:
    """Route raw ARI events and own transient inbound setup tasks."""

    def __init__(
        self,
        *,
        adapter: Any,
        ari: AsteriskARIClient,
        registry: AsteriskCallRegistry,
        call_service: AsteriskCallService,
        dispatch_call_event: CallEventDispatcher,
    ) -> None:
        self.adapter = adapter
        self.ari = ari
        self.registry = registry
        self.call_service = call_service
        self.dispatch_call_event = dispatch_call_event
        self._closing = False
        self._registration_lock = asyncio.Lock()
        self._inbound_setup_tasks: dict[str, asyncio.Task] = {}

    def open(self) -> None:
        self._closing = False

    async def close(self) -> None:
        self._closing = True
        tasks = list(self._inbound_setup_tasks.values())
        self._inbound_setup_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_event(self, event: Mapping[str, Any]) -> None:
        await self.process_event(event, wait=False)

    async def process_event(
        self,
        event: Mapping[str, Any],
        *,
        wait: bool = True,
    ) -> None:
        """Translate one raw ARI event into a serialized call event."""

        event_type = event.get("type")
        channel = event.get("channel") or {}
        channel_id = channel.get("id", "") if isinstance(channel, Mapping) else ""

        if event_type == "StasisStart":
            args = event.get("args") or []
            role = args[0] if args else ""
            if role == "inbound":
                if wait:
                    await self._handle_inbound_start(channel)
                else:
                    self._schedule_inbound_start(channel)
            elif role == "transfer-failed":
                if session_id := self.registry.by_caller(channel_id):
                    await self.dispatch_call_event(
                        session_id,
                        ReferFailed(),
                        wait=wait,
                    )
            elif role == "transfer-completed":
                if session_id := self.registry.by_caller(channel_id):
                    await self.dispatch_call_event(
                        session_id,
                        ReferCompleted(),
                        wait=wait,
                    )
            elif role == "transfer-destination" and len(args) > 1:
                await self.dispatch_call_event(
                    str(args[1]),
                    DestinationAnswered(channel_id=channel_id),
                    wait=wait,
                )

        elif event_type == "StasisEnd":
            await self._cancel_inbound_start(channel_id)
            if session_id := self.registry.by_caller(channel_id):
                await self.dispatch_call_event(
                    session_id,
                    StasisEnded(),
                    wait=wait,
                )

        elif event_type == "Dial":
            peer = event.get("peer") or {}
            peer_id = peer.get("id", "") if isinstance(peer, Mapping) else ""
            session_id = self.registry.by_destination(peer_id)
            dialstatus = str(event.get("dialstatus") or "").upper()
            if session_id and dialstatus in DIAL_FAILURE_STATUSES:
                await self.dispatch_call_event(
                    session_id,
                    DestinationFailed(reason=dialstatus.lower()),
                    wait=wait,
                )

        elif event_type == "ChannelDestroyed":
            destroyed = event.get("channel") or {}
            destroyed_id = (
                destroyed.get("id", "") if isinstance(destroyed, Mapping) else ""
            )
            await self._cancel_inbound_start(destroyed_id)
            if session_id := self.registry.by_destination(destroyed_id):
                await self.dispatch_call_event(
                    session_id,
                    DestinationDestroyed(
                        channel_id=destroyed_id,
                        reason=str(event.get("cause_txt") or "destination_hung_up"),
                    ),
                    wait=wait,
                )
            elif session_id := self.registry.by_media(destroyed_id):
                await self.dispatch_call_event(
                    session_id,
                    MediaDestroyed(channel_id=destroyed_id),
                    wait=wait,
                )
            elif session_id := self.registry.by_caller(destroyed_id):
                await self.dispatch_call_event(
                    session_id,
                    CallerDestroyed(
                        reason=str(event.get("cause_txt") or "caller_destroyed")
                    ),
                    wait=wait,
                )

    def _schedule_inbound_start(self, channel: Mapping[str, Any]) -> None:
        channel_id = str(channel.get("id") or "")
        if (
            not channel_id
            or self.registry.by_caller(channel_id) is not None
            or channel_id in self._inbound_setup_tasks
            or self._closing
        ):
            return

        async def run() -> None:
            try:
                await self._handle_inbound_start(dict(channel))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Asterisk inbound setup failed: channel=%s",
                    channel_id,
                )

        task = asyncio.create_task(
            run(),
            name=f"aiavatar-asterisk-inbound-{channel_id}",
        )
        self._inbound_setup_tasks[channel_id] = task

        def done(done_task: asyncio.Task) -> None:
            if self._inbound_setup_tasks.get(channel_id) is done_task:
                self._inbound_setup_tasks.pop(channel_id, None)

        task.add_done_callback(done)

    async def _cancel_inbound_start(self, channel_id: str) -> None:
        task = self._inbound_setup_tasks.pop(channel_id, None)
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _handle_inbound_start(self, channel: Mapping[str, Any]) -> None:
        caller_channel_id = str(channel.get("id") or "")
        if (
            not caller_channel_id
            or self.registry.by_caller(caller_channel_id) is not None
        ):
            return

        try:
            await self._establish_inbound_call(channel, caller_channel_id)
        except asyncio.CancelledError:
            try:
                await self._cleanup_inbound_start(
                    caller_channel_id,
                    # StasisEnd means Asterisk already owns the caller's
                    # departure. Shutdown has no such event, so release it.
                    hangup_caller=self._closing,
                )
            except Exception:
                logger.exception(
                    "Failed to clean up canceled Asterisk inbound setup: "
                    "channel=%s",
                    caller_channel_id,
                )
            raise
        except Exception:
            logger.exception(
                "Failed to establish Asterisk inbound call: channel=%s",
                caller_channel_id,
            )
            try:
                await self._cleanup_inbound_start(
                    caller_channel_id,
                    hangup_caller=True,
                )
            except Exception:
                logger.exception(
                    "Failed to clean up failed Asterisk inbound setup: "
                    "channel=%s",
                    caller_channel_id,
                )

    async def _establish_inbound_call(
        self,
        channel: Mapping[str, Any],
        caller_channel_id: str,
    ) -> None:
        session_id = (
            await self.ari.get_channel_variable(
                caller_channel_id,
                "AIAVATAR_SESSION_ID",
            )
            or caller_channel_id
        )
        if self.registry.get(session_id) is not None:
            logger.warning(
                "Rejected duplicate Asterisk session ID: session=%s channel=%s",
                session_id,
                caller_channel_id,
            )
            await self.ari.delete_channel(caller_channel_id)
            return

        caller = channel.get("caller") or {}
        dialplan = channel.get("dialplan") or {}
        caller_number = str(caller.get("number") or "")
        caller_name = str(caller.get("name") or "")
        called_number = (
            await self.ari.get_channel_variable(
                caller_channel_id,
                "AIAVATAR_CALLED_NUMBER",
            )
            or str(dialplan.get("exten") or "")
        )
        presentation = (
            await self.ari.get_channel_variable(caller_channel_id, "CALLERID(pres)")
            or "allowed"
        )
        trusted_pai = await self.ari.get_channel_variable(
            caller_channel_id,
            "AIAVATAR_TRUSTED_PAI",
        )
        ucid = await self.ari.get_channel_variable(
            caller_channel_id,
            "AIAVATAR_UCID",
        )
        uui = await self.ari.get_channel_variable(
            caller_channel_id,
            "AIAVATAR_UUI",
        )

        session = None
        async with self._registration_lock:
            duplicate = (
                self.registry.get(session_id) is not None
                or self.registry.by_caller(caller_channel_id) is not None
            )
            if not duplicate:
                session = self.adapter.register_session(
                    session_id,
                    ari_caller_channel_id=caller_channel_id,
                    caller_number=caller_number,
                    caller_name=caller_name,
                    caller_presentation=presentation,
                    called_number=called_number,
                    trusted_pai=trusted_pai,
                    ucid=ucid,
                    uui=uui,
                )
                self.registry.register(session)
        if duplicate:
            logger.warning(
                "Rejected concurrently duplicated Asterisk session: "
                "session=%s channel=%s",
                session_id,
                caller_channel_id,
            )
            await self.ari.delete_channel(caller_channel_id)
            return
        assert session is not None
        await self.call_service.create_conversation_media(
            session,
            answer_caller=True,
        )

    async def _cleanup_inbound_start(
        self,
        caller_channel_id: str,
        *,
        hangup_caller: bool,
    ) -> None:
        session_id = self.registry.by_caller(caller_channel_id)
        if session_id is not None:
            await self.call_service.cleanup_call(
                session_id,
                hangup_caller=hangup_caller,
            )
        elif hangup_caller:
            await self.ari.delete_channel(caller_channel_id)
