import asyncio
import base64
from contextlib import suppress
import json
import logging
import ssl
from typing import Any, Awaitable, Callable, Mapping, Optional, Set
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import connect as websocket_connect


logger = logging.getLogger(__name__)

ARIEventHandler = Callable[[Mapping[str, Any]], Awaitable[None]]


class AsteriskARIError(RuntimeError):
    def __init__(self, method: str, path: str, status_code: int, detail: str):
        super().__init__(f"ARI {method} {path} failed ({status_code}): {detail}")
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail


class AsteriskARITransportError(RuntimeError):
    """ARI request outcome is unknown because its transport failed."""

    def __init__(self, method: str, path: str):
        super().__init__(f"ARI {method} {path} transport failed")
        self.method = method
        self.path = path


class AsteriskARIClient:
    """Own the ARI HTTP and event WebSocket transports.

    This client deliberately knows nothing about calls, sessions, or transfer
    state. It validates ARI responses and forwards decoded events to one
    application callback.
    """

    def __init__(
        self,
        *,
        base_url: str,
        username: str,
        password: str,
        application: str = "aiavatar",
        tls_verify: bool = True,
        reconnect_delay: float = 1.0,
        startup_timeout: float = 10.0,
        http_client: Optional[Any] = None,
        event_connector: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not base_url or not username or not password:
            raise ValueError("ARI URL, username, and password are required")
        if not application:
            raise ValueError("application is required")
        if reconnect_delay <= 0:
            raise ValueError("reconnect_delay must be positive")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")

        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.application = application
        self.reconnect_delay = reconnect_delay
        self.startup_timeout = startup_timeout
        self._event_ssl: Optional[ssl.SSLContext] = None
        if urlsplit(self.base_url).scheme == "https" and not tls_verify:
            self._event_ssl = ssl.create_default_context()
            self._event_ssl.check_hostname = False
            self._event_ssl.verify_mode = ssl.CERT_NONE

        self._owns_http_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self.base_url + "/",
            auth=(username, password),
            verify=tls_verify,
            timeout=30.0,
        )
        self._event_connector = event_connector or websocket_connect
        self._event_handler: Optional[ARIEventHandler] = None
        self._event_task: Optional[asyncio.Task] = None
        self._closing = asyncio.Event()
        self._event_connected = asyncio.Event()

    @property
    def event_connected(self) -> bool:
        return self._event_connected.is_set()

    async def start(self, event_handler: ARIEventHandler) -> None:
        if self._event_task and not self._event_task.done():
            if self._event_handler != event_handler:
                raise RuntimeError("ARI event client is already bound to a handler")
            if self._event_connected.is_set():
                return
        else:
            self._event_handler = event_handler
            self._closing.clear()
            self._event_connected.clear()
            self._event_task = asyncio.create_task(
                self._run_event_loop(),
                name="aiavatar-asterisk-ari-events",
            )

        try:
            await asyncio.wait_for(
                self._event_connected.wait(),
                timeout=self.startup_timeout,
            )
        except asyncio.TimeoutError as ex:
            await self._stop_event_task()
            raise RuntimeError(
                "Timed out connecting to the Asterisk ARI event WebSocket"
            ) from ex

    async def close(self) -> None:
        await self.stop_events()
        if self._owns_http_client:
            await self._http.aclose()

    async def stop_events(self) -> None:
        """Stop event delivery while leaving ARI HTTP available for cleanup."""

        self._closing.set()
        self._event_connected.clear()
        await self._stop_event_task()

    async def answer_channel(self, channel_id: str) -> None:
        await self.request(
            "POST",
            f"channels/{self._id(channel_id)}/answer",
            tolerate={409, 412},
        )

    async def create_bridge(self, bridge_id: str, bridge_type: str) -> None:
        await self.request(
            "POST",
            f"bridges/{self._id(bridge_id)}",
            params={"type": bridge_type},
        )

    async def add_channel(self, bridge_id: str, channel_id: str) -> None:
        await self.request(
            "POST",
            f"bridges/{self._id(bridge_id)}/addChannel",
            params={"channel": channel_id},
        )

    async def remove_channel(self, bridge_id: str, channel_id: str) -> None:
        if not bridge_id or not channel_id:
            return
        await self.request(
            "POST",
            f"bridges/{self._id(bridge_id)}/removeChannel",
            params={"channel": channel_id},
            tolerate={404, 409, 422},
        )

    async def destroy_bridge(self, bridge_id: str) -> None:
        if bridge_id:
            await self.request(
                "DELETE",
                f"bridges/{self._id(bridge_id)}",
                tolerate={404},
            )

    async def delete_channel(self, channel_id: str) -> None:
        if channel_id:
            await self.request(
                "DELETE",
                f"channels/{self._id(channel_id)}",
                tolerate={404},
            )

    async def get_channel_variable(self, channel_id: str, variable: str) -> str:
        response = await self.request(
            "GET",
            f"channels/{self._id(channel_id)}/variable",
            params={"variable": variable},
            tolerate={400, 404, 409},
        )
        if isinstance(response, Mapping):
            value = response.get("value")
            return value if isinstance(value, str) else ""
        return ""

    async def get_channel(self, channel_id: str) -> Optional[Mapping[str, Any]]:
        response = await self.request(
            "GET",
            f"channels/{self._id(channel_id)}",
            tolerate={404},
        )
        return response if isinstance(response, Mapping) else None

    async def set_channel_variable(
        self,
        channel_id: str,
        variable: str,
        value: str,
    ) -> None:
        await self.request(
            "POST",
            f"channels/{self._id(channel_id)}/variable",
            params={"variable": variable, "value": value},
        )

    async def create_external_media(
        self,
        *,
        channel_id: str,
        external_host: str,
        transport_data: str,
        variables: Mapping[str, str],
    ) -> Any:
        return await self.request(
            "POST",
            "channels/externalMedia",
            params={
                "channelId": channel_id,
                "app": self.application,
                "external_host": external_host,
                "encapsulation": "none",
                "transport": "websocket",
                "connection_type": "client",
                "format": "slin16",
                "direction": "both",
                "transport_data": transport_data,
            },
            json_body={"variables": dict(variables)},
        )

    async def continue_channel(
        self,
        channel_id: str,
        *,
        context: str,
        extension: str,
    ) -> None:
        await self.request(
            "POST",
            f"channels/{self._id(channel_id)}/continue",
            params={"context": context, "extension": extension, "priority": 1},
        )

    async def start_moh(self, bridge_id: str) -> None:
        await self.request("POST", f"bridges/{self._id(bridge_id)}/moh")

    async def originate(
        self,
        *,
        endpoint: str,
        app_args: str,
        caller_id: str,
        timeout: int,
        channel_id: str,
        originator: str,
        variables: Mapping[str, str],
    ) -> Any:
        return await self.request(
            "POST",
            "channels",
            params={
                "endpoint": endpoint,
                "app": self.application,
                "appArgs": app_args,
                "callerId": caller_id,
                "timeout": timeout,
                "channelId": channel_id,
                "originator": originator,
            },
            json_body={"variables": dict(variables)},
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
        tolerate: Optional[Set[int]] = None,
    ) -> Any:
        try:
            response = await self._http.request(
                method,
                path,
                params=params,
                json=json_body,
            )
        except (httpx.RequestError, TimeoutError) as ex:
            raise AsteriskARITransportError(method, path) from ex
        if response.status_code in (200, 201, 202, 204):
            if response.status_code == 204 or not getattr(response, "content", b""):
                return None
            try:
                return response.json()
            except (TypeError, ValueError):
                return None
        if tolerate and response.status_code in tolerate:
            return None
        detail = getattr(response, "text", "")
        raise AsteriskARIError(method, path, response.status_code, detail)

    async def _stop_event_task(self) -> None:
        if self._event_task:
            self._event_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._event_task
            self._event_task = None

    async def _run_event_loop(self) -> None:
        handler = self._event_handler
        if handler is None:
            raise RuntimeError("ARI event handler is not configured")

        headers = {
            "Authorization": "Basic " + base64.b64encode(
                f"{self.username}:{self.password}".encode("utf-8")
            ).decode("ascii")
        }
        uri = self._events_url()
        connection_options: dict[str, Any] = {
            "additional_headers": headers,
            "max_size": 2**20,
        }
        if self._event_ssl is not None:
            connection_options["ssl"] = self._event_ssl
        while not self._closing.is_set():
            try:
                async with self._event_connector(
                    uri,
                    **connection_options,
                ) as websocket:
                    self._event_connected.set()
                    try:
                        async for source in websocket:
                            event = self._decode_event(source)
                            if event is None:
                                continue
                            try:
                                await handler(event)
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                channel = event.get("channel") or {}
                                channel_id = (
                                    channel.get("id", "")
                                    if isinstance(channel, Mapping)
                                    else ""
                                )
                                logger.exception(
                                    "Asterisk ARI event handling failed: "
                                    "type=%s channel=%s",
                                    event.get("type"),
                                    channel_id,
                                )
                    finally:
                        self._event_connected.clear()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._event_connected.clear()
                if not self._closing.is_set():
                    logger.exception("Asterisk ARI event WebSocket disconnected")
            if not self._closing.is_set():
                try:
                    await asyncio.wait_for(
                        self._closing.wait(),
                        timeout=self.reconnect_delay,
                    )
                except asyncio.TimeoutError:
                    pass

    @staticmethod
    def _decode_event(source: Any) -> Optional[Mapping[str, Any]]:
        try:
            event = json.loads(source)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Ignored malformed Asterisk ARI event")
            return None
        if not isinstance(event, dict):
            logger.warning("Ignored non-object Asterisk ARI event")
            return None
        return event

    def _events_url(self) -> str:
        parts = urlsplit(self.base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        query_string = urlencode({"app": self.application, "subscribeAll": "false"})
        return urlunsplit((scheme, parts.netloc, parts.path + "/events", query_string, ""))

    @staticmethod
    def _id(value: str) -> str:
        return quote(value, safe="")
