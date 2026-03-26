import asyncio
import json
import os
import pytest
from typing import Any, Dict
from uuid import uuid4
from aiavatar.sts.llm.openai_responses_websocket import OpenAIResponsesWebSocketService

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMAGE_URL = os.getenv("IMAGE_URL")
MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """
## 基本設定

あなたはユーザーの妹として、感情表現豊かに振る舞ってください。

## 表情について

あなたは以下のようなタグで表情を表現することができます。

[face:Angry]はあ？何言ってるのか全然わからないんですけど。

表情のバリエーションは以下の通りです。

- Joy
- Angry
"""


@pytest.mark.asyncio
async def test_openai_responses_ws_service_simple():
    """
    Test OpenAIResponsesWebSocketService with a basic prompt to check if it can stream responses.
    This test actually calls OpenAI Responses API via WebSocket, so it may cost tokens.
    """
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        reasoning_effort="none",
    )
    context_id = f"test_ws_context_{uuid4()}"

    user_message = "君が大切にしていたプリンは、私が勝手に食べておいた。"

    collected_text = []
    collected_voice = []

    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.error_info:
            pytest.fail(f"Error from WebSocket service: {resp.error_info}")
        if resp.text:
            collected_text.append(resp.text)
        if resp.voice_text:
            collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)
    full_voice = "".join(collected_voice)
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "[face:Angry]" in full_text, "Control tag doesn't appear in text."
    assert "[face:Angry]" not in full_voice, "Control tag was not removed from voice_text."

    # Check server-side context management (response_id should be stored)
    assert context_id in service.response_ids, "response_id not stored for context."

    # Check context marker was saved
    histories = await service.context_manager.get_histories(context_id)
    assert len(histories) > 0, "No context marker was saved."

    await service._ws_pool.close()


@pytest.mark.asyncio
async def test_openai_responses_ws_service_context_continuity():
    """
    Test that the WebSocket service maintains conversation context across multiple calls
    via previous_response_id (server-side history).
    This test actually calls OpenAI Responses API via WebSocket, so it may cost tokens.
    """
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="あなたは親切なアシスタントです。短く回答してください。",
        model=MODEL,
        reasoning_effort="none",
    )
    context_id = f"test_ws_continuity_{uuid4()}"

    # First call: tell the model something specific
    async for resp in service.chat_stream(context_id, "test_user", "私の好きな果物はマンゴーです。覚えておいてください。"):
        if resp.error_info:
            pytest.fail(f"Error in first call: {resp.error_info}")

    first_response_id = service.response_ids.get(context_id)
    assert first_response_id is not None, "response_id should be set after first call."

    # Second call: ask about what was said before
    collected_text = []
    async for resp in service.chat_stream(context_id, "test_user", "私の好きな果物は何ですか？"):
        if resp.error_info:
            pytest.fail(f"Error in second call: {resp.error_info}")
        if resp.text:
            collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert "マンゴー" in full_text, f"Context was not maintained. Response: {full_text}"

    # response_id should have been updated
    second_response_id = service.response_ids.get(context_id)
    assert second_response_id != first_response_id, "response_id should be updated after second call."

    await service._ws_pool.close()


@pytest.mark.asyncio
async def test_openai_responses_ws_service_tool_calls():
    """
    Test OpenAIResponsesWebSocketService with a registered tool.
    This test actually calls OpenAI Responses API via WebSocket, so it may cost tokens.
    """
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="You MUST always use the solve_math tool to solve any math problem. Never calculate manually.",
        model=MODEL,
        reasoning_effort="none",
    )
    context_id = f"test_ws_tool_context_{uuid4()}"

    # Register tool using Chat Completions format (should be auto-converted)
    tool_spec = {
        "type": "function",
        "function": {
            "name": "solve_math",
            "description": "Solve simple math problems",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"}
                },
                "required": ["problem"]
            }
        }
    }

    @service.tool(tool_spec)
    async def solve_math(problem: str) -> Dict[str, Any]:
        if problem.strip() == "1+1":
            return {"answer": 2}
        else:
            return {"answer": "unknown"}

    user_message = "次の問題を解いて: 1+1"
    collected_text = []
    tool_called = False

    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.error_info:
            pytest.fail(f"Error from WebSocket service: {resp.error_info}")
        if resp.text:
            collected_text.append(resp.text)
        if resp.tool_call:
            tool_called = True

    full_text = "".join(collected_text)
    assert tool_called, "Tool was not called."
    assert "2" in full_text, f"Answer '2' not found in response: {full_text}"

    await service._ws_pool.close()


@pytest.mark.asyncio
async def test_openai_responses_ws_service_tool_calls_response_formatter():
    """
    Test OpenAIResponsesWebSocketService with a registered tool that has response_formatter.
    The tool result should be formatted directly without a 2nd LLM call.
    This test actually calls OpenAI Responses API via WebSocket, so it may cost tokens.
    """
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="You MUST always use the solve_math tool to solve any math problem. Never calculate manually.",
        model=MODEL,
        reasoning_effort="none",
    )
    context_id = f"test_ws_tool_rf_context_{uuid4()}"

    tool_spec = {
        "type": "function",
        "function": {
            "name": "solve_math",
            "description": "Solve simple math problems",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"}
                },
                "required": ["problem"]
            }
        }
    }

    @service.tool(tool_spec)
    async def solve_math(problem: str) -> Dict[str, Any]:
        if problem.strip() == "1+1":
            return {"answer": 2}
        else:
            return {"answer": "unknown"}

    @service.tools["solve_math"].response_formatter
    def format_result(result, arguments):
        return f"Formatter: {arguments['problem']}の計算結果は{result['answer']}です。"

    user_message = "次の問題を解いて: 1+1"
    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.error_info:
            pytest.fail(f"Error from WebSocket service: {resp.error_info}")
        if resp.text:
            collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert "Formatter: 1+1の計算結果は2です。" in full_text, f"Direct response not found in text: {full_text}"

    await service._ws_pool.close()


@pytest.mark.asyncio
async def test_openai_responses_ws_service_context_isolation():
    """
    Test that conversations with different context_ids are properly isolated
    when called concurrently over WebSocket. This verifies that one user's
    conversation does not leak into another's — the bug that was found during development.
    This test actually calls OpenAI Responses API via WebSocket, so it may cost tokens.
    """
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="あなたは親切なアシスタントです。短く回答してください。",
        model=MODEL,
        reasoning_effort="none",
    )

    context_a = f"test_ws_isolation_a_{uuid4()}"
    context_b = f"test_ws_isolation_b_{uuid4()}"

    errors = []

    # First round: give each context different information
    async def setup_context(ctx_id, message):
        async for resp in service.chat_stream(ctx_id, "test_user", message):
            if resp.error_info:
                errors.append(f"{ctx_id}: {resp.error_info}")

    await asyncio.gather(
        setup_context(context_a, "私の名前は太郎です。覚えておいてください。"),
        setup_context(context_b, "私の名前は花子です。覚えておいてください。"),
    )

    assert not errors, f"Errors during setup: {errors}"

    # Verify each context has its own response_id
    assert context_a in service.response_ids, f"response_id not set for context_a. response_ids={service.response_ids}"
    assert context_b in service.response_ids, f"response_id not set for context_b. response_ids={service.response_ids}"
    assert service.response_ids[context_a] != service.response_ids[context_b], \
        "Different contexts should have different response_ids."

    # Second round: ask each context about the stored information concurrently
    results = {}

    async def ask_name(ctx_id):
        texts = []
        async for resp in service.chat_stream(ctx_id, "test_user", "私の名前は何ですか？"):
            if resp.error_info:
                errors.append(f"{ctx_id}: {resp.error_info}")
            if resp.text:
                texts.append(resp.text)
        results[ctx_id] = "".join(texts)

    await asyncio.gather(
        ask_name(context_a),
        ask_name(context_b),
    )

    assert not errors, f"Errors during ask: {errors}"

    assert "太郎" in results[context_a], f"Context A should remember '太郎'. Got: {results[context_a]}"
    assert "花子" in results[context_b], f"Context B should remember '花子'. Got: {results[context_b]}"

    # Cross-check: ensure no leakage
    assert "花子" not in results[context_a], f"Context A should NOT contain '花子'. Got: {results[context_a]}"
    assert "太郎" not in results[context_b], f"Context B should NOT contain '太郎'. Got: {results[context_b]}"

    await service._ws_pool.close()


@pytest.mark.asyncio
async def test_openai_responses_ws_service_image():
    """
    Test OpenAIResponsesWebSocketService with image input.
    This test actually calls OpenAI Responses API via WebSocket, so it may cost tokens.
    """
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        reasoning_effort="none",
    )
    context_id = f"test_ws_image_context_{uuid4()}"

    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", "これは何ですか？漢字で答えてください。", files=[{"type": "image", "url": IMAGE_URL}]):
        if resp.error_info:
            pytest.fail(f"Error from WebSocket service: {resp.error_info}")
        if resp.text:
            collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert len(full_text) > 0, "No text was returned from the LLM."
    assert "寿司" in full_text, f"寿司 is not in text: {full_text}"

    await service._ws_pool.close()


@pytest.mark.asyncio
async def test_openai_responses_ws_service_connection_pool_reuse():
    """
    Test that the WebSocket connection pool properly reuses connections.
    After one request, the connection should be returned to the pool and
    reused for the next request.
    This test actually calls OpenAI Responses API via WebSocket, so it may cost tokens.
    """
    service = OpenAIResponsesWebSocketService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="あなたは親切なアシスタントです。短く回答してください。",
        model=MODEL,
        reasoning_effort="none",
    )

    context_id = f"test_ws_pool_reuse_{uuid4()}"

    # First call
    async for resp in service.chat_stream(context_id, "test_user", "こんにちは"):
        if resp.error_info:
            pytest.fail(f"Error in first call: {resp.error_info}")

    # After first call, pool should have one idle connection
    assert not service._ws_pool._idle.empty(), "Pool should have an idle connection after first call."

    # Second call should reuse the pooled connection
    async for resp in service.chat_stream(context_id, "test_user", "元気ですか？"):
        if resp.error_info:
            pytest.fail(f"Error in second call: {resp.error_info}")

    # Pool should still have an idle connection
    assert not service._ws_pool._idle.empty(), "Pool should still have an idle connection after second call."

    await service._ws_pool.close()

    # After close, pool should be empty
    assert service._ws_pool._idle.empty(), "Pool should be empty after close."
