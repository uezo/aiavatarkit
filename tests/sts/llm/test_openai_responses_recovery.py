import configparser
import os
from pathlib import Path
from uuid import uuid4

import openai
import pytest

from aiavatar.sts.llm.openai_responses import OpenAIResponsesService
from aiavatar.sts.llm.openai_responses_websocket import (
    OpenAIResponsesWebSocketService,
)


def get_pytest_env(name: str) -> str:
    if value := os.getenv(name):
        return value
    config = configparser.ConfigParser()
    config.read(Path(__file__).parents[3] / "pytest.ini")
    for line in config.get("pytest", "env", fallback="").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip()
    return None


OPENAI_API_KEY = get_pytest_env("OPENAI_API_KEY")


async def collect_response(service, context_id: str, text: str) -> str:
    chunks = []
    async for response in service.chat_stream(context_id, "test_user", text):
        if response.error_info:
            pytest.fail(f"Unexpected API error: {response.error_info}")
        if response.text:
            chunks.append(response.text)
    return "".join(chunks)


@pytest.mark.asyncio
async def test_http_service_recovers_deleted_response_with_local_history(tmp_path):
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="短く回答してください。",
        model="gpt-4.1",
        temperature=0.0,
        db_connection_str=str(tmp_path / "http_recovery.db"),
    )
    context_id = f"test_http_recovery_{uuid4()}"

    try:
        await collect_response(
            service,
            context_id,
            "合言葉はアボカドです。覚えてください。",
        )
        deleted_response_id = await service.response_id_store.get(context_id)
        assert deleted_response_id

        # Make the stored previous_response_id genuinely unavailable on the API.
        await service.openai_client.responses.delete(deleted_response_id)

        recovered_text = await collect_response(
            service,
            context_id,
            "合言葉は何ですか？合言葉だけ答えてください。",
        )
        recovered_response_id = await service.response_id_store.get(context_id)

        assert "アボカド" in recovered_text
        assert recovered_response_id
        assert recovered_response_id != deleted_response_id
    finally:
        await service.openai_client.close()


@pytest.mark.asyncio
async def test_websocket_service_recovers_deleted_response_with_local_history(tmp_path):
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="短く回答してください。",
        model="gpt-5.4-mini",
        reasoning_effort="none",
        db_connection_str=str(tmp_path / "websocket_recovery.db"),
    )
    rest_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    context_id = f"test_websocket_recovery_{uuid4()}"

    try:
        await collect_response(
            service,
            context_id,
            "合言葉はアボカドです。覚えてください。",
        )
        deleted_response_id = await service.response_id_store.get(context_id)
        assert deleted_response_id

        # WebSocket responses can be deleted through the REST Responses API.
        await rest_client.responses.delete(deleted_response_id)

        recovered_text = await collect_response(
            service,
            context_id,
            "合言葉は何ですか？合言葉だけ答えてください。",
        )
        recovered_response_id = await service.response_id_store.get(context_id)

        assert "アボカド" in recovered_text
        assert recovered_response_id
        assert recovered_response_id != deleted_response_id
    finally:
        await service._ws_pool.close()
        await rest_client.close()
