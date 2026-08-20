import asyncio
import ssl

import httpx
import pytest

from aiavatar.adapter.asterisk.ari_client import (
    AsteriskARIClient,
    AsteriskARITransportError,
)

from .conftest import FakeARIClient


def _client(connector, **values):
    return AsteriskARIClient(
        base_url="https://asterisk.invalid:8089/ari",
        username="ari-user",
        password="ari-secret",
        http_client=FakeARIClient(),
        event_connector=connector,
        **values,
    )


@pytest.mark.asyncio
async def test_start_waits_for_event_connection():
    release = asyncio.Event()

    class BlockingEventWebSocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await release.wait()
            raise StopAsyncIteration

    class EventConnection:
        async def __aenter__(self):
            return BlockingEventWebSocket()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    client = _client(lambda *args, **kwargs: EventConnection())

    async def handle_event(event):
        pass

    await client.start(handle_event)
    assert client.event_connected is True
    await client.close()
    assert client.event_connected is False


@pytest.mark.asyncio
async def test_start_fails_when_event_connection_never_succeeds():
    def connector(*args, **kwargs):
        raise OSError("connection refused")

    client = _client(
        connector,
        reconnect_delay=0.01,
        startup_timeout=0.03,
    )

    async def handle_event(event):
        pass

    with pytest.raises(RuntimeError, match="Timed out connecting"):
        await client.start(handle_event)
    assert client.event_connected is False
    await client.close()


@pytest.mark.asyncio
async def test_event_handler_failure_does_not_drop_following_event():
    second_event_seen = asyncio.Event()
    keep_open = asyncio.Event()

    class EventWebSocket:
        def __init__(self):
            self.events = iter([
                '{"type":"First"}',
                '{"type":"Second"}',
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration:
                await keep_open.wait()
                raise StopAsyncIteration

    class EventConnection:
        async def __aenter__(self):
            return EventWebSocket()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    client = _client(
        lambda *args, **kwargs: EventConnection(),
        startup_timeout=0.5,
    )

    async def handle_event(event):
        if event["type"] == "First":
            raise RuntimeError("one bad event")
        second_event_seen.set()

    await client.start(handle_event)
    await asyncio.wait_for(second_event_seen.wait(), timeout=0.5)
    assert client.event_connected is True
    await client.close()


@pytest.mark.asyncio
async def test_tls_verify_false_applies_to_event_websocket():
    release = asyncio.Event()
    connection_options = {}

    class EventWebSocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await release.wait()
            raise StopAsyncIteration

    class EventConnection:
        async def __aenter__(self):
            return EventWebSocket()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    def connector(*args, **kwargs):
        connection_options.update(kwargs)
        return EventConnection()

    client = _client(connector, tls_verify=False)
    await client.start(lambda event: None)

    context = connection_options["ssl"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is False
    assert context.verify_mode == ssl.CERT_NONE
    await client.close()


@pytest.mark.asyncio
async def test_http_transport_failure_is_normalized():
    class FailingTransport(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            raise httpx.ConnectError("connection lost")

    client = AsteriskARIClient(
        base_url="https://asterisk.invalid:8089/ari",
        username="ari-user",
        password="ari-secret",
        http_client=FailingTransport(),
    )

    with pytest.raises(AsteriskARITransportError) as error:
        await client.get_channel("caller-1")

    assert error.value.method == "GET"
    assert error.value.path == "channels/caller-1"


@pytest.mark.asyncio
async def test_http_response_decoding_failure_is_normalized():
    class FailingTransport(FakeARIClient):
        async def request(self, method, path, params=None, json=None):
            raise httpx.DecodingError(
                "response body could not be decoded",
                request=httpx.Request(method, f"https://asterisk.invalid/{path}"),
            )

    client = AsteriskARIClient(
        base_url="https://asterisk.invalid:8089/ari",
        username="ari-user",
        password="ari-secret",
        http_client=FailingTransport(),
    )

    with pytest.raises(AsteriskARITransportError) as error:
        await client.continue_channel(
            "caller-1",
            context="aiavatar-transfer",
            extension="1234",
        )

    assert error.value.method == "POST"
    assert error.value.path == "channels/caller-1/continue"
