import asyncio
import logging
from typing import Any, Awaitable, Dict, Mapping
from urllib.parse import quote
from uuid import uuid4

from .ari_client import (
    AsteriskARIClient,
    AsteriskARIError,
    AsteriskARITransportError,
)
from .registry import AsteriskCallRegistry
from .models import AsteriskSessionData


logger = logging.getLogger(__name__)

_MEDIA_STASIS_RETRY_DELAY = 0.05
_MEDIA_STASIS_RETRY_LIMIT = 40


class AsteriskReferOutcomeUnknown(RuntimeError):
    """ARI lost the response after a REFER may already have started."""


class AsteriskCallService:
    """Mutate the Asterisk channel/bridge topology for one call at a time.

    Lifecycle decisions remain in ``AsteriskCallManager``. This component
    performs only the requested topology transition and synchronously updates
    the shared session and channel registry around each ARI operation.
    """

    def __init__(
        self,
        *,
        adapter: Any,
        ari: AsteriskARIClient,
        registry: AsteriskCallRegistry,
        bridge_endpoint: str,
        external_media_host: str,
        transfer_context: str,
        originate_timeout: int,
    ) -> None:
        self.adapter = adapter
        self.ari = ari
        self.registry = registry
        self.bridge_endpoint = bridge_endpoint
        self.external_media_host = external_media_host
        self.transfer_context = transfer_context
        self.originate_timeout = originate_timeout
        self._cleanup_tasks: Dict[str, asyncio.Task] = {}

    async def create_conversation_media(
        self,
        session: AsteriskSessionData,
        *,
        answer_caller: bool,
    ) -> None:
        session.cleanup_started = False
        session.media_cleanup_started = False
        registered = self.adapter.register_session(
            session.session_id,
            ari_caller_channel_id=session.ari_caller_channel_id,
            caller_number=session.caller_number,
            caller_name=session.caller_name,
            caller_presentation=session.caller_presentation,
            called_number=session.called_number,
            trusted_pai=session.trusted_pai,
            ucid=session.ucid,
            uui=session.uui,
        )
        if registered is not session:
            raise RuntimeError(
                "Asterisk adapter replaced the manager-owned session object"
            )

        if answer_caller:
            await self.ari.answer_channel(session.ari_caller_channel_id)

        bridge_id = f"aiavatar-{uuid4()}"
        # Keep the client-generated identity before yielding. If ARI times out
        # after creating the bridge, later recovery still knows what to delete.
        session.bridge_id = bridge_id
        await self.ari.create_bridge(
            bridge_id,
            "mixing,dtmf_events,proxy_media",
        )
        await self.ari.add_channel(bridge_id, session.ari_caller_channel_id)

        media_channel_id = f"aiavatar-media-{uuid4()}"
        # Reserve the expected ID before ARI creates the channel. Asterisk may
        # connect Media WebSocket before the HTTP response is returned.
        self.registry.bind_media(session, media_channel_id)
        transport_data = (
            "f(json)d(both)v(session_id="
            + quote(session.session_id, safe="")
            + ")"
        )
        media = await self.ari.create_external_media(
            channel_id=media_channel_id,
            external_host=self.external_media_host,
            transport_data=transport_data,
            variables=self._media_variables(session),
        )

        if isinstance(media, Mapping) and media.get("id"):
            actual_id = str(media["id"])
            if actual_id != media_channel_id:
                self.registry.bind_media(session, actual_id)
                media_channel_id = actual_id
        await self._add_media_channel(bridge_id, media_channel_id)

    async def begin_refer(self, session: AsteriskSessionData) -> None:
        session.transfer_method = "refer"
        variables = dict(session.transfer_variables)
        variables.update({
            "AIAVATAR_SESSION_ID": session.session_id,
            "AIAVATAR_TRANSFER_ALIAS": session.transfer_alias,
            "AIAVATAR_TRANSFER_DESTINATION": session.transfer_destination,
        })
        for name, value in variables.items():
            await self.ari.set_channel_variable(
                session.ari_caller_channel_id,
                name,
                value,
            )
        await self.detach_conversation_media(session)
        try:
            await self.ari.continue_channel(
                session.ari_caller_channel_id,
                context=self.transfer_context,
                extension=session.transfer_destination,
            )
        except AsteriskARIError as ex:
            if ex.status_code == 408 or 500 <= ex.status_code < 600:
                raise AsteriskReferOutcomeUnknown(
                    "ARI returned an indeterminate status after the REFER "
                    "dialplan continuation may have started"
                ) from ex
            raise
        except AsteriskARITransportError as ex:
            raise AsteriskReferOutcomeUnknown(
                "ARI did not confirm whether the REFER dialplan continuation started"
            ) from ex

    async def begin_bridge_transfer(self, session: AsteriskSessionData) -> None:
        session.transfer_method = "bridge"
        await self.detach_conversation_media(session)

        holding_bridge_id = f"aiavatar-hold-{uuid4()}"
        session.holding_bridge_id = holding_bridge_id
        await self.ari.create_bridge(holding_bridge_id, "holding")
        await self.ari.add_channel(
            holding_bridge_id,
            session.ari_caller_channel_id,
        )
        await self.ari.start_moh(holding_bridge_id)

        destination_channel_id = f"aiavatar-transfer-{uuid4()}"
        # As with media, reserve before originate because StasisStart can race
        # with the originate HTTP response.
        self.registry.bind_destination(session, destination_channel_id)
        originated = await self.ari.originate(
            endpoint=(
                f"PJSIP/{session.transfer_destination}@{self.bridge_endpoint}"
            ),
            app_args=f"transfer-destination,{session.session_id}",
            caller_id=self._outbound_caller_id(session),
            timeout=self.originate_timeout,
            channel_id=destination_channel_id,
            originator=session.ari_caller_channel_id,
            variables=self._outbound_identity_variables(session),
        )

        if isinstance(originated, Mapping) and originated.get("id"):
            actual_id = str(originated["id"])
            if actual_id != destination_channel_id:
                self.registry.bind_destination(session, actual_id)

    async def connect_destination(
        self,
        session: AsteriskSessionData,
        destination_channel_id: str,
    ) -> bool:
        if destination_channel_id != session.destination_channel_id:
            return False

        mixing_bridge_id = f"aiavatar-transfer-bridge-{uuid4()}"
        session.bridge_id = mixing_bridge_id
        try:
            await self.ari.create_bridge(
                mixing_bridge_id,
                "mixing,dtmf_events,proxy_media",
            )
            await self.ari.remove_channel(
                session.holding_bridge_id,
                session.ari_caller_channel_id,
            )
            await self.ari.add_channel(
                mixing_bridge_id,
                session.ari_caller_channel_id,
            )
            await self.ari.add_channel(
                mixing_bridge_id,
                destination_channel_id,
            )
            await self.ari.destroy_bridge(session.holding_bridge_id)
            session.holding_bridge_id = ""
        except Exception:
            try:
                await self.ari.destroy_bridge(mixing_bridge_id)
            except Exception:
                logger.exception(
                    "Failed to roll back transfer bridge for session %s",
                    session.session_id,
                )
            else:
                if session.bridge_id == mixing_bridge_id:
                    session.bridge_id = ""
            raise
        return True

    async def restore_ai(self, session: AsteriskSessionData) -> None:
        destination_id = session.destination_channel_id
        if destination_id:
            await self.ari.delete_channel(destination_id)
            self.registry.unbind_destination(session, destination_id)
        if session.holding_bridge_id:
            holding_bridge_id = session.holding_bridge_id
            await self.ari.destroy_bridge(holding_bridge_id)
            session.holding_bridge_id = ""

        # Recovery may begin after only part of a transfer preparation ran.
        # Remove whatever conversation topology is still owned before replacing
        # it so session IDs never overwrite live remote resources.
        await self.detach_conversation_media(session)
        await self.create_conversation_media(session, answer_caller=False)

    async def restore_lost_media(
        self,
        session: AsteriskSessionData,
        media_channel_id: str,
    ) -> None:
        if media_channel_id != session.media_channel_id:
            self.registry.unbind_media(session, media_channel_id)
            return
        self.registry.unbind_media(session, media_channel_id)
        if session.bridge_id:
            await self.ari.destroy_bridge(session.bridge_id)
            session.bridge_id = ""
        await self.create_conversation_media(session, answer_caller=False)

    async def detach_conversation_media(
        self,
        session: AsteriskSessionData,
    ) -> None:
        media_channel_id = session.media_channel_id
        if media_channel_id:
            await self.ari.delete_channel(media_channel_id)
            self.registry.unbind_media(session, media_channel_id)
        if session.bridge_id:
            bridge_id = session.bridge_id
            await self.ari.destroy_bridge(bridge_id)
            session.bridge_id = ""

    async def cleanup_call(
        self,
        session_id: str,
        *,
        hangup_caller: bool,
    ) -> None:
        cleanup_task = self._cleanup_tasks.get(session_id)
        if cleanup_task is None:
            session = self.registry.get(session_id)
            if session is None:
                return
            cleanup_task = asyncio.create_task(
                self._run_cleanup_call(
                    session,
                    hangup_caller=hangup_caller,
                ),
                name=f"aiavatar-asterisk-call-cleanup-{session_id}",
            )
            self._cleanup_tasks[session_id] = cleanup_task
        try:
            await self._await_cleanup_task(cleanup_task)
        finally:
            if (
                self._cleanup_tasks.get(session_id) is cleanup_task
                and cleanup_task.done()
            ):
                self._cleanup_tasks.pop(session_id, None)

    async def _run_cleanup_call(
        self,
        session: AsteriskSessionData,
        *,
        hangup_caller: bool,
    ) -> None:
        session_id = session.session_id
        session.cleanup_started = True

        media_id = session.media_channel_id
        destination_id = session.destination_channel_id
        caller_id = session.ari_caller_channel_id
        bridge_ids = {session.bridge_id, session.holding_bridge_id} - {""}
        # Stop routing events before any awaited remote cleanup operation.
        self.registry.remove(session_id)
        failures = []

        async def best_effort(label: str, operation: Awaitable[Any]) -> None:
            try:
                await operation
            except Exception as ex:
                failures.append((label, ex))
                logger.exception(
                    "Asterisk call cleanup operation failed: "
                    "session=%s resource=%s",
                    session_id,
                    label,
                )

        if media_id:
            await best_effort(
                f"media_channel:{media_id}",
                self.ari.delete_channel(media_id),
            )
        if destination_id:
            await best_effort(
                f"destination_channel:{destination_id}",
                self.ari.delete_channel(destination_id),
            )
        for bridge_id in bridge_ids:
            await best_effort(
                f"bridge:{bridge_id}",
                self.ari.destroy_bridge(bridge_id),
            )
        if hangup_caller and caller_id:
            await best_effort(
                f"caller_channel:{caller_id}",
                self.ari.delete_channel(caller_id),
            )
        await best_effort(
            "adapter_session",
            self.adapter.unregister_session(session_id),
        )

        if failures:
            logger.warning(
                "Asterisk call cleanup completed with %d failure(s): session=%s",
                len(failures),
                session_id,
            )

    @staticmethod
    async def _await_cleanup_task(task: asyncio.Task) -> None:
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as ex:
                cancellation = ex
        task.result()
        if cancellation is not None:
            raise cancellation

    async def _add_media_channel(self, bridge_id: str, channel_id: str) -> None:
        """Wait for a newly-created external media channel to enter Stasis."""

        for attempt in range(_MEDIA_STASIS_RETRY_LIMIT):
            try:
                await self.ari.add_channel(bridge_id, channel_id)
                return
            except AsteriskARIError as ex:
                not_ready = (
                    ex.status_code == 422
                    and "not in Stasis application" in ex.detail
                )
                if not not_ready or attempt + 1 == _MEDIA_STASIS_RETRY_LIMIT:
                    raise
                await asyncio.sleep(_MEDIA_STASIS_RETRY_DELAY)

    @staticmethod
    def _media_variables(session: AsteriskSessionData) -> Dict[str, str]:
        return {
            "AIAVATAR_SESSION_ID": session.session_id,
            "AIAVATAR_CALLER_CHANNEL_ID": session.ari_caller_channel_id,
            "AIAVATAR_CALLER_NUMBER": session.caller_number,
            "AIAVATAR_CALLER_NAME": session.caller_name,
            "AIAVATAR_CALLER_PRESENTATION": session.caller_presentation,
            "AIAVATAR_CALLED_NUMBER": session.called_number,
        }

    @staticmethod
    def _outbound_identity_variables(
        session: AsteriskSessionData,
    ) -> Dict[str, str]:
        variables = dict(session.transfer_variables)
        variables.update({
            "AIAVATAR_SESSION_ID": session.session_id,
            "AIAVATAR_ORIGINAL_CALLER_NUMBER": session.caller_number,
            "AIAVATAR_ORIGINAL_CALLER_NAME": session.caller_name,
            "AIAVATAR_CALLER_PRESENTATION": session.caller_presentation,
            "CALLERID(num)": session.caller_number,
            "CALLERID(name)": session.caller_name,
            "CALLERID(num-pres)": session.caller_presentation,
        })
        if session.ucid:
            variables["AIAVATAR_UCID"] = session.ucid
        if session.uui:
            variables["AIAVATAR_UUI"] = session.uui
        if session.trusted_pai:
            variables["AIAVATAR_TRUSTED_PAI"] = session.trusted_pai
        return variables

    @staticmethod
    def _outbound_caller_id(session: AsteriskSessionData) -> str:
        presentation = session.caller_presentation.lower()
        if "prohib" in presentation or "unavailable" in presentation:
            return "Anonymous <anonymous>"
        name = session.caller_name.replace("\r", " ").replace("\n", " ")
        number = session.caller_number.replace("\r", "").replace("\n", "")
        return f"{name} <{number}>" if name else number
