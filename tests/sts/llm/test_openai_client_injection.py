from unittest.mock import AsyncMock

import pytest

from aiavatar.sts.llm import chatgpt as chatgpt_module
from aiavatar.sts.llm import openai_responses as responses_module
from aiavatar.sts.llm import (
    openai_responses_websocket as responses_websocket_module,
)
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.llm.openai_responses import OpenAIResponsesService
from aiavatar.sts.llm.openai_responses_websocket import (
    OpenAIResponsesWebSocketService,
)


class FakeAsyncClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.close = AsyncMock()


@pytest.mark.asyncio
@pytest.mark.parametrize("service_class", [ChatGPTService, OpenAIResponsesService])
async def test_http_services_use_injected_client_without_closing_it(
    service_class,
    tmp_path,
):
    client = FakeAsyncClient()
    service = service_class(
        openai_client=client,
        model="azure-deployment-name",
        db_connection_str=str(tmp_path / f"{service_class.__name__}.db"),
    )

    assert service.openai_client is client

    await service.close()
    await service.close()

    client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_chatgpt_accepts_official_async_azure_openai_client(tmp_path):
    client = chatgpt_module.openai_module.AsyncAzureOpenAI(
        api_key="test-key",
        azure_endpoint="https://resource.openai.azure.com",
        api_version="2024-10-21",
    )
    try:
        service = ChatGPTService(
            openai_client=client,
            model="chat-deployment",
            db_connection_str=str(tmp_path / "official-azure-client.db"),
        )

        assert service.openai_client is client
        await service.close()
        assert not client.is_closed()
    finally:
        await client.close()

    assert client.is_closed()


@pytest.mark.asyncio
async def test_responses_accepts_official_client_configured_for_azure_v1(
    tmp_path,
):
    client = chatgpt_module.openai_module.AsyncOpenAI(
        api_key="test-key",
        base_url="https://resource.openai.azure.com/openai/v1/",
    )
    try:
        service = OpenAIResponsesService(
            openai_client=client,
            model="responses-deployment",
            db_connection_str=str(tmp_path / "azure-v1-client.db"),
        )

        assert service.openai_client is client
        await service.close()
        assert not client.is_closed()
    finally:
        await client.close()

    assert client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "service_class"),
    [
        (chatgpt_module, ChatGPTService),
        (responses_module, OpenAIResponsesService),
    ],
)
async def test_http_services_close_internally_created_client_once(
    monkeypatch,
    module,
    service_class,
    tmp_path,
):
    created = []

    def create_client(**kwargs):
        client = FakeAsyncClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(module.openai_module, "AsyncClient", create_client)
    service = service_class(
        openai_api_key="test-key",
        base_url="https://example.test/v1",
        db_connection_str=str(tmp_path / f"{service_class.__name__}.db"),
    )

    await service.close()
    await service.close()

    assert created[0].kwargs == {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
    }
    created[0].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_chatgpt_injected_client_takes_precedence_over_constructor_options(
    tmp_path,
):
    client = FakeAsyncClient()
    service = ChatGPTService(
        openai_client=client,
        openai_api_key="ignored-key",
        base_url="https://ignored.example/v1",
        custom_openai_module=object(),
        model="azure-deployment-name",
        db_connection_str=str(tmp_path / "chatgpt.db"),
    )

    assert service.openai_client is client
    await service.close()
    client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_responses_injected_client_takes_precedence_over_constructor_options(
    tmp_path,
):
    client = FakeAsyncClient()
    service = OpenAIResponsesService(
        openai_client=client,
        openai_api_key="ignored-key",
        base_url="https://ignored.example/v1",
        db_connection_str=str(tmp_path / "responses.db"),
    )

    assert service.openai_client is client
    await service.close()
    client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_azure_model_still_builds_azure_client_with_warning(
    monkeypatch,
    tmp_path,
):
    created = []

    def create_azure_client(**kwargs):
        client = FakeAsyncClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(
        chatgpt_module.openai_module,
        "AsyncAzureOpenAI",
        create_azure_client,
    )

    with pytest.warns(DeprecationWarning, match="Selecting Azure"):
        service = ChatGPTService(
            openai_api_key="azure-key",
            base_url=(
                "https://resource.openai.azure.com/openai/deployments/chat"
                "?api-version=2024-10-21"
            ),
            model="azure",
            db_connection_str=str(tmp_path / "azure.db"),
        )

    assert created[0].kwargs == {
        "api_key": "azure-key",
        "api_version": "2024-10-21",
        "base_url": (
            "https://resource.openai.azure.com/openai/deployments/chat"
            "?api-version=2024-10-21"
        ),
    }

    await service.close()
    created[0].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_openai_module_remains_available_with_warning(tmp_path):
    class ClientWithoutClose:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModule:
        AsyncClient = ClientWithoutClose
        AsyncAzureOpenAI = ClientWithoutClose

    with pytest.warns(DeprecationWarning, match="custom_openai_module"):
        service = ChatGPTService(
            openai_api_key="test-key",
            custom_openai_module=FakeModule,
            db_connection_str=str(tmp_path / "custom-module.db"),
        )

    assert isinstance(service.openai_client, ClientWithoutClose)
    await service.close()


@pytest.mark.asyncio
async def test_websocket_service_closes_internally_created_pool_once(
    monkeypatch,
    tmp_path,
):
    created = []

    class FakePool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.close = AsyncMock()
            created.append(self)

    monkeypatch.setattr(responses_websocket_module, "WebSocketPool", FakePool)
    service = OpenAIResponsesWebSocketService(
        openai_api_key="test-key",
        ws_url="wss://example.test/openai",
        db_connection_str=str(tmp_path / "owned-websocket.db"),
    )

    await service.close()
    await service.close()

    assert created[0].kwargs == {
        "url": "wss://example.test/openai/v1/responses",
        "api_key": "test-key",
        "max_size": 100,
        "max_age": 3300,
    }
    created[0].close.assert_awaited_once()
