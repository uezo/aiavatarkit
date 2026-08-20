import logging
import re
from typing import Any, Dict, Mapping, Optional

from .ari_client import AsteriskARIClient
from .actor import AsteriskCallActor, AsteriskCallActorClosed
from .event_handler import AsteriskARIEventHandler
from .registry import AsteriskCallRegistry
from .service import AsteriskCallService, AsteriskReferOutcomeUnknown
from .state import (
    AsteriskCallEvent,
    AsteriskCallState,
    CallerDestroyed,
    DestinationAnswered,
    DestinationDestroyed,
    DestinationFailed,
    HangupRequested,
    MediaConnected,
    MediaDestroyed,
    MediaRestoreTimedOut,
    ReferCompleted,
    ReferFailed,
    ReferTimedOut,
    StasisEnded,
    TransferRequested,
)
from .models import AsteriskSessionData


logger = logging.getLogger(__name__)

TRANSFER_STRATEGIES = frozenset({"refer", "bridge", "refer_then_bridge"})
_EXTENSION_PATTERN = re.compile(r"^[0-9]+$")
_REFER_RECHECK_INTERVAL = 1.0


class AsteriskCallManager:
    """Application-facing facade that coordinates active Asterisk calls."""

    def __init__(
        self,
        *,
        adapter: Any,
        ari_client: AsteriskARIClient,
        transfer_destinations: Mapping[str, str],
        bridge_endpoint: Optional[str] = None,
        transfer_strategy: str = "refer_then_bridge",
        external_media_host: str = "aiavatarkit-media",
        transfer_context: str = "aiavatar-transfer",
        originate_timeout: int = 30,
        refer_timeout: float = 30.0,
        media_start_timeout: float = 10.0,
    ):
        if transfer_strategy not in TRANSFER_STRATEGIES:
            raise ValueError(
                f"transfer_strategy must be one of {sorted(TRANSFER_STRATEGIES)}"
            )
        if not external_media_host:
            raise ValueError("external_media_host is required")
        if (
            transfer_strategy in {"bridge", "refer_then_bridge"}
            and not bridge_endpoint
        ):
            raise ValueError(
                "bridge_endpoint is required when transfer_strategy uses bridge"
            )
        if originate_timeout <= 0:
            raise ValueError("originate_timeout must be positive")
        if refer_timeout <= 0:
            raise ValueError("refer_timeout must be positive")
        if media_start_timeout <= 0:
            raise ValueError("media_start_timeout must be positive")
        destinations: Dict[str, str] = {}
        for alias, extension in transfer_destinations.items():
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("Transfer destination aliases must be non-empty strings")
            if not isinstance(extension, str) or not _EXTENSION_PATTERN.fullmatch(extension):
                raise ValueError(
                    f"Transfer destination {alias!r} must map to a digits-only extension"
                )
            destinations[alias] = extension

        self.adapter = adapter
        self.ari = ari_client
        self.transfer_destinations = destinations
        self.transfer_strategy = transfer_strategy
        self.refer_timeout = refer_timeout
        self.media_start_timeout = media_start_timeout
        self.registry = AsteriskCallRegistry()
        self.call_service = AsteriskCallService(
            adapter=adapter,
            ari=ari_client,
            registry=self.registry,
            bridge_endpoint=bridge_endpoint or "",
            external_media_host=external_media_host,
            transfer_context=transfer_context,
            originate_timeout=originate_timeout,
        )
        self._actors: Dict[str, AsteriskCallActor] = {}
        self.event_handler = AsteriskARIEventHandler(
            adapter=adapter,
            ari=ari_client,
            registry=self.registry,
            call_service=self.call_service,
            dispatch_call_event=self._dispatch_call_event,
        )
        self._ari_event_handler = self.event_handler.handle_event

        self.adapter.bind_call_manager(self)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def start(self) -> None:
        self.event_handler.open()
        await self.ari.start(self._ari_event_handler)

    async def close(self) -> None:
        try:
            await self.ari.stop_events()
            await self.event_handler.close()

            # Stop state owners before touching their resources directly.
            for actor in list(self._actors.values()):
                if actor.state == AsteriskCallState.CLEANING_UP:
                    await actor.wait_closed()
                else:
                    await actor.cancel()
            for session_id in list(self.registry):
                try:
                    await self.call_service.cleanup_call(
                        session_id,
                        hangup_caller=True,
                    )
                except Exception:
                    logger.exception(
                        "Failed to close Asterisk call: session=%s",
                        session_id,
                    )
        finally:
            await self.ari.close()

    @property
    def event_connected(self) -> bool:
        return self.ari.event_connected

    @property
    def sessions(self) -> Mapping[str, AsteriskSessionData]:
        """Read-only view of currently owned call sessions."""

        return self.registry.sessions

    def resolve_destination(self, alias: str) -> str:
        """Resolve only configured aliases; arbitrary numbers and SIP URIs are rejected."""

        try:
            return self.transfer_destinations[alias]
        except (KeyError, TypeError) as ex:
            raise ValueError(f"Transfer destination is not allowed: {alias!r}") from ex

    async def transfer(self, session_id: str, destination_alias: str) -> None:
        if self.registry.get(session_id) is None:
            await self.adapter.notify_transfer_failed(
                session_id,
                destination_alias,
                "session_not_found",
            )
            return
        await self._dispatch_call_event(
            session_id,
            TransferRequested(destination_alias=destination_alias),
            wait=True,
        )

    async def _transfer_now(
        self,
        actor: AsteriskCallActor,
        event: TransferRequested,
        destination_alias: str,
    ) -> None:
        session_id = actor.session_id
        session = self.registry.get(session_id)
        if session is None:
            await self.adapter.notify_transfer_failed(
                session_id,
                destination_alias,
                "session_not_found",
            )
            return
        try:
            destination = self.resolve_destination(destination_alias)
        except ValueError:
            await self.adapter.notify_transfer_failed(
                session_id,
                destination_alias,
                "destination_not_allowed",
            )
            return

        if actor.state not in (
            AsteriskCallState.ACTIVE,
            AsteriskCallState.REFER_FAILED,
        ):
            await self.adapter.notify_transfer_failed(
                session_id,
                destination_alias,
                "transfer_already_in_progress",
            )
            return

        session.transfer_alias = destination_alias
        session.transfer_destination = destination
        try:
            transfer_request = await self.adapter.prepare_transfer(
                session,
                destination_alias=destination_alias,
                destination=destination,
                transfer_strategy=self.transfer_strategy,
            )
            session.transfer_variables = dict(transfer_request.variables)
        except Exception:
            logger.exception(
                "Asterisk transfer preparation failed: session=%s destination=%s",
                session_id,
                destination_alias,
            )
            await self.adapter.notify_transfer_failed(
                session_id,
                destination_alias,
                "transfer_prepare_failed",
            )
            return
        await self.adapter.notify_transfer_started(session_id, destination_alias)

        if self.transfer_strategy in ("refer", "refer_then_bridge"):
            actor.transition(AsteriskCallState.REFER_PENDING, event)
            try:
                await self.call_service.begin_refer(session)
                self._arm_refer_timeout(actor)
                return
            except AsteriskReferOutcomeUnknown:
                logger.warning(
                    "SIP REFER start outcome is unknown; scheduling reconciliation: "
                    "session=%s",
                    session_id,
                    exc_info=True,
                )
                self._arm_refer_timeout(actor)
                return
            except Exception as ex:
                logger.exception("SIP REFER setup failed for session %s", session_id)
                actor.transition(AsteriskCallState.REFER_FAILED, event)
                if self.transfer_strategy == "refer":
                    await self._restore_from_actor(
                        actor,
                        event,
                        session,
                        f"refer_setup_failed:{ex}",
                    )
                    return

        actor.transition(AsteriskCallState.BRIDGE_DIALING, event)
        try:
            await self.call_service.begin_bridge_transfer(session)
        except Exception as ex:
            logger.exception(
                "Asterisk bridge transfer setup failed for session %s",
                session_id,
            )
            await self._restore_from_actor(
                actor,
                event,
                session,
                f"bridge_setup_failed:{ex}",
            )

    async def hangup(self, session_id: str) -> None:
        if self.registry.get(session_id) is None:
            return
        await self._dispatch_call_event(
            session_id,
            HangupRequested(hangup_caller=True),
            wait=True,
        )

    async def handle_ari_event(
        self,
        event: Mapping[str, Any],
        *,
        wait: bool = True,
    ) -> None:
        """Handle one ARI event. Public for deterministic embedding and tests."""

        await self.event_handler.process_event(event, wait=wait)

    def _get_call_actor(self, session_id: str) -> Optional[AsteriskCallActor]:
        actor = self._actors.get(session_id)
        if actor is not None and not actor.closed:
            return actor
        session = self.registry.get(session_id)
        if session is None:
            return None
        actor = AsteriskCallActor(
            session_id=session_id,
            handler=self._handle_call_event,
            on_transition=self._on_call_transition,
            on_stopped=self._on_call_actor_stopped,
        )
        self._actors[session_id] = actor
        return actor

    async def _dispatch_call_event(
        self,
        session_id: str,
        event: AsteriskCallEvent,
        *,
        wait: bool,
    ) -> bool:
        actor = self._get_call_actor(session_id)
        if actor is None:
            return False
        try:
            await actor.dispatch(event, wait=wait)
        except AsteriskCallActorClosed:
            return False
        if wait and actor.closed:
            await actor.wait_closed()
        return True

    def _on_call_transition(
        self,
        actor: AsteriskCallActor,
        previous: AsteriskCallState,
        current: AsteriskCallState,
        event: AsteriskCallEvent,
    ) -> None:
        session = self.registry.get(actor.session_id)
        if session is not None:
            session.transfer_state = current.value
        logger.info(
            "Asterisk call state transition: session=%s %s --%s--> %s",
            actor.session_id,
            previous.value,
            type(event).__name__,
            current.value,
        )

    def _on_call_actor_stopped(self, actor: AsteriskCallActor) -> None:
        if self._actors.get(actor.session_id) is actor:
            self._actors.pop(actor.session_id, None)

    async def _cleanup_from_actor(
        self,
        actor: AsteriskCallActor,
        event: AsteriskCallEvent,
        *,
        hangup_caller: bool,
    ) -> None:
        if actor.state in (AsteriskCallState.CLEANING_UP, AsteriskCallState.CLOSED):
            return
        actor.transition(AsteriskCallState.CLEANING_UP, event)
        await self.call_service.cleanup_call(
            actor.session_id,
            hangup_caller=hangup_caller,
        )
        actor.transition(AsteriskCallState.CLOSED, event)

    async def _restore_from_actor(
        self,
        actor: AsteriskCallActor,
        event: AsteriskCallEvent,
        session: AsteriskSessionData,
        reason: str,
    ) -> None:
        session.transfer_failure_reason = reason
        actor.transition(AsteriskCallState.RESTORING_AI, event)
        try:
            await self.call_service.restore_ai(session)
        except Exception:
            logger.exception(
                "Failed to restore AI media: session=%s",
                actor.session_id,
            )
            await self._cleanup_from_actor(
                actor,
                event,
                hangup_caller=True,
            )
            return
        actor.arm_timeout(
            MediaRestoreTimedOut(channel_id=session.media_channel_id),
            delay=self.media_start_timeout,
        )

    async def _complete_restore_from_actor(
        self,
        actor: AsteriskCallActor,
        event: AsteriskCallEvent,
        session: AsteriskSessionData,
    ) -> None:
        reason = session.transfer_failure_reason
        session.transfer_failure_reason = ""
        actor.transition(AsteriskCallState.ACTIVE, event)
        try:
            await self.adapter.notify_transfer_failed(
                session.session_id,
                session.transfer_alias,
                reason,
            )
        except Exception:
            logger.exception(
                "Transfer failure callback failed: session=%s",
                actor.session_id,
            )

    def _arm_refer_timeout(
        self,
        actor: AsteriskCallActor,
        *,
        attempt: int = 1,
    ) -> None:
        delay = (
            self.refer_timeout
            if attempt == 1
            else min(self.refer_timeout, _REFER_RECHECK_INTERVAL)
        )
        actor.arm_timeout(
            ReferTimedOut(attempt=attempt),
            delay=delay,
        )

    async def _handle_refer_failure_from_actor(
        self,
        actor: AsteriskCallActor,
        event: AsteriskCallEvent,
        session: AsteriskSessionData,
        reason: str,
    ) -> None:
        actor.transition(AsteriskCallState.REFER_FAILED, event)
        if self.transfer_strategy == "refer_then_bridge":
            actor.transition(AsteriskCallState.BRIDGE_DIALING, event)
            try:
                await self.call_service.begin_bridge_transfer(session)
            except Exception as ex:
                logger.exception(
                    "Bridge fallback setup failed for session %s",
                    actor.session_id,
                )
                await self._restore_from_actor(
                    actor,
                    event,
                    session,
                    f"bridge_setup_failed:{ex}",
                )
        else:
            await self._restore_from_actor(
                actor,
                event,
                session,
                reason,
            )

    async def _handle_refer_timeout(
        self,
        actor: AsteriskCallActor,
        event: ReferTimedOut,
        session: AsteriskSessionData,
    ) -> None:
        if event.attempt == 1:
            logger.warning(
                "Asterisk REFER result timed out: session=%s timeout=%.1fs",
                actor.session_id,
                self.refer_timeout,
            )

        try:
            channel = await self.ari.get_channel(session.ari_caller_channel_id)
        except Exception:
            logger.exception(
                "Failed to reconcile timed-out REFER: session=%s",
                actor.session_id,
            )
            self._arm_refer_timeout(actor, attempt=event.attempt + 1)
            return

        if channel is None:
            await self._complete_refer_unknown_from_actor(
                actor,
                event,
                session,
                "caller_channel_missing",
            )
            return

        dialplan = channel.get("dialplan") or {}
        if not isinstance(dialplan, Mapping):
            dialplan = {}
        app_name = str(dialplan.get("app_name") or "").lower()
        app_data = str(dialplan.get("app_data") or "").lower()
        if app_name == "stasis":
            if "transfer-completed" in app_data:
                await self._complete_refer_from_actor(actor, event, session)
                return
            if "transfer-failed" not in app_data:
                try:
                    transfer_status = (
                        await self.ari.get_channel_variable(
                            session.ari_caller_channel_id,
                            "TRANSFERSTATUS",
                        )
                    ).upper()
                except Exception:
                    logger.exception(
                        "Failed to read timed-out REFER status: session=%s",
                        actor.session_id,
                    )
                    self._arm_refer_timeout(
                        actor,
                        attempt=event.attempt + 1,
                    )
                    return
                if transfer_status == "SUCCESS":
                    await self._complete_refer_from_actor(actor, event, session)
                    return
                if transfer_status not in {"FAILURE", "UNSUPPORTED"}:
                    logger.error(
                        "Unknown Asterisk TRANSFERSTATUS; closing call without "
                        "bridge fallback: session=%s status=%r",
                        actor.session_id,
                        transfer_status,
                    )
                    await self._complete_refer_unknown_from_actor(
                        actor,
                        event,
                        session,
                        "unknown_transfer_status",
                        hangup_caller=True,
                    )
                    return
            await self._handle_refer_failure_from_actor(
                actor,
                event,
                session,
                "refer_timeout",
            )
            return

        # The channel is still executing Transfer() outside Stasis, where ARI
        # cannot safely restore media or hang it up. Recheck until Asterisk
        # returns the channel or the successful handoff removes it.
        logger.debug(
            "Asterisk REFER still outside Stasis: session=%s app=%s attempt=%d",
            actor.session_id,
            app_name or "unknown",
            event.attempt,
        )
        self._arm_refer_timeout(actor, attempt=event.attempt + 1)

    async def _handle_call_event(
        self,
        actor: AsteriskCallActor,
        event: AsteriskCallEvent,
    ) -> None:
        session = self.registry.get(actor.session_id)
        if session is None:
            actor.transition(AsteriskCallState.CLOSED, event)
            return
        if actor.state in (AsteriskCallState.CLEANING_UP, AsteriskCallState.CLOSED):
            return

        if isinstance(event, TransferRequested):
            await self._transfer_now(
                actor,
                event,
                event.destination_alias,
            )
            return

        if isinstance(event, HangupRequested):
            await self._cleanup_from_actor(
                actor,
                event,
                hangup_caller=event.hangup_caller,
            )
            return

        if isinstance(event, StasisEnded):
            if actor.state != AsteriskCallState.REFER_PENDING:
                await self._cleanup_from_actor(
                    actor,
                    event,
                    hangup_caller=False,
                )
            return

        if isinstance(event, ReferFailed):
            if actor.state == AsteriskCallState.REFER_PENDING:
                await self._handle_refer_failure_from_actor(
                    actor,
                    event,
                    session,
                    "refer_failed",
                )
            return

        if isinstance(event, ReferCompleted):
            if actor.state == AsteriskCallState.REFER_PENDING:
                await self._complete_refer_from_actor(actor, event, session)
            return

        if isinstance(event, ReferTimedOut):
            if actor.state == AsteriskCallState.REFER_PENDING:
                await self._handle_refer_timeout(actor, event, session)
            return

        if isinstance(event, DestinationAnswered):
            if actor.state == AsteriskCallState.BRIDGE_DIALING:
                try:
                    completed = await self.call_service.connect_destination(
                        session,
                        event.channel_id,
                    )
                except Exception as ex:
                    logger.exception(
                        "Failed to complete bridge transfer for session %s",
                        actor.session_id,
                    )
                    await self._restore_from_actor(
                        actor,
                        event,
                        session,
                        f"bridge_connect_failed:{ex}",
                    )
                else:
                    if completed:
                        actor.transition(AsteriskCallState.BRIDGE_COMPLETED, event)
                        await self.adapter.notify_transfer_completed(
                            session.session_id,
                            session.transfer_alias,
                            "bridge",
                        )
            return

        if isinstance(event, DestinationFailed):
            if actor.state == AsteriskCallState.BRIDGE_DIALING:
                await self._restore_from_actor(
                    actor,
                    event,
                    session,
                    event.reason,
                )
            return

        if isinstance(event, DestinationDestroyed):
            if event.channel_id != session.destination_channel_id:
                return
            if actor.state == AsteriskCallState.BRIDGE_DIALING:
                await self._restore_from_actor(
                    actor,
                    event,
                    session,
                    event.reason,
                )
            elif actor.state == AsteriskCallState.BRIDGE_COMPLETED:
                await self._cleanup_from_actor(
                    actor,
                    event,
                    hangup_caller=True,
                )
            return

        if isinstance(event, MediaDestroyed):
            if event.channel_id != session.media_channel_id:
                self.registry.unbind_media(session, event.channel_id)
                return
            if actor.state == AsteriskCallState.ACTIVE and not session.cleanup_started:
                try:
                    await self.call_service.restore_lost_media(
                        session,
                        event.channel_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to restore Asterisk media: session=%s",
                        actor.session_id,
                    )
                    await self._cleanup_from_actor(
                        actor,
                        event,
                        hangup_caller=True,
                    )
            return

        if isinstance(event, MediaConnected):
            if (
                actor.state == AsteriskCallState.RESTORING_AI
                and event.channel_id == session.media_channel_id
            ):
                await self._complete_restore_from_actor(
                    actor,
                    event,
                    session,
                )
            return

        if isinstance(event, MediaRestoreTimedOut):
            if (
                actor.state == AsteriskCallState.RESTORING_AI
                and event.channel_id == session.media_channel_id
            ):
                logger.error(
                    "Asterisk AI media restore timed out: session=%s channel=%s "
                    "timeout=%.1fs",
                    actor.session_id,
                    event.channel_id,
                    self.media_start_timeout,
                )
                await self._cleanup_from_actor(
                    actor,
                    event,
                    hangup_caller=True,
                )
            return

        if isinstance(event, CallerDestroyed):
            if actor.state == AsteriskCallState.REFER_PENDING:
                await self._complete_refer_unknown_from_actor(
                    actor,
                    event,
                    session,
                    event.reason or "caller_hangup",
                )
            else:
                await self._cleanup_from_actor(
                    actor,
                    event,
                    hangup_caller=False,
                )

    async def _complete_refer_from_actor(
        self,
        actor: AsteriskCallActor,
        event: AsteriskCallEvent,
        session: AsteriskSessionData,
    ) -> None:
        actor.transition(AsteriskCallState.REFER_COMPLETED, event)
        try:
            await self.adapter.notify_transfer_completed(
                session.session_id,
                session.transfer_alias,
                "refer",
            )
        finally:
            await self._cleanup_from_actor(
                actor,
                event,
                hangup_caller=False,
            )

    async def _complete_refer_unknown_from_actor(
        self,
        actor: AsteriskCallActor,
        event: AsteriskCallEvent,
        session: AsteriskSessionData,
        reason: str,
        *,
        hangup_caller: bool = False,
    ) -> None:
        actor.transition(AsteriskCallState.REFER_UNKNOWN, event)
        try:
            await self.adapter.notify_transfer_unknown(
                session.session_id,
                session.transfer_alias,
                reason,
            )
        finally:
            await self._cleanup_from_actor(
                actor,
                event,
                hangup_caller=hangup_caller,
            )

    async def media_connected(self, session_id: str, channel_id: str) -> None:
        """Notify the per-call actor that replacement AI media is ready."""

        actor = self._actors.get(session_id)
        if actor is None or actor.state != AsteriskCallState.RESTORING_AI:
            return
        await self._dispatch_call_event(
            session_id,
            MediaConnected(channel_id=channel_id),
            wait=False,
        )
