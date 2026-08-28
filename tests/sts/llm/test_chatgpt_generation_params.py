from types import SimpleNamespace

import pytest

from aiavatar.sts.llm.chatgpt import ChatGPTService


class EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return EmptyStream()
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="[tools:NOT_FOUND]"),
            )],
        )


class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


async def make_dynamic_and_stream_requests(service):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    await service.get_dynamic_tools_default(messages)
    async for _ in service.get_llm_stream_response(
        "context-id",
        "user-id",
        messages,
    ):
        pass


@pytest.mark.asyncio
async def test_unset_generation_params_are_omitted(tmp_path):
    client = FakeOpenAIClient()
    service = ChatGPTService(
        openai_client=client,
        db_connection_str=str(tmp_path / "unset.db"),
    )

    await make_dynamic_and_stream_requests(service)

    for params in client.chat.completions.calls:
        assert "temperature" not in params
        assert "reasoning_effort" not in params


@pytest.mark.asyncio
async def test_explicit_generation_params_are_forwarded(tmp_path):
    client = FakeOpenAIClient()
    service = ChatGPTService(
        openai_client=client,
        temperature=0.0,
        reasoning_effort="none",
        db_connection_str=str(tmp_path / "explicit.db"),
    )

    await make_dynamic_and_stream_requests(service)

    for params in client.chat.completions.calls:
        assert params["temperature"] == 0.0
        assert params["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_other_reasoning_effort_is_forwarded_without_temperature(tmp_path):
    client = FakeOpenAIClient()
    service = ChatGPTService(
        openai_client=client,
        reasoning_effort="low",
        db_connection_str=str(tmp_path / "low.db"),
    )

    await make_dynamic_and_stream_requests(service)

    for params in client.chat.completions.calls:
        assert params["reasoning_effort"] == "low"
        assert "temperature" not in params
