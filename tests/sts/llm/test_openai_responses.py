import asyncio
import json
import os
import pytest
from typing import Any, Dict
from uuid import uuid4
from aiavatar.sts.llm.openai_responses import OpenAIResponsesService

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMAGE_URL = os.getenv("IMAGE_URL")
MODEL = "gpt-4.1"

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
async def test_openai_responses_service_simple():
    """
    Test OpenAIResponsesService with a basic prompt to check if it can stream responses.
    This test actually calls OpenAI Responses API, so it may cost tokens.
    """
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_context_{uuid4()}"

    user_message = "君が大切にしていたプリンは、私が勝手に食べておいた。"

    collected_text = []
    collected_voice = []

    async for resp in service.chat_stream(context_id, "test_user", user_message):
        collected_text.append(resp.text)
        collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)
    full_voice = "".join(filter(None, collected_voice))
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "[face:Angry]" in full_text, "Control tag doesn't appear in text."
    assert "[face:Angry]" not in full_voice, "Control tag was not removed from voice_text."

    # Check server-side context management (response_id should be stored)
    assert await service.response_id_store.get(context_id) is not None, "response_id not stored for context."

    # Check context was saved locally
    histories = await service.context_manager.get_histories(context_id)
    assert len(histories) > 0, "No local context was saved."

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_openai_responses_service_context_continuity():
    """
    Test that the service maintains conversation context across multiple calls
    via previous_response_id (server-side history).
    This test actually calls OpenAI Responses API, so it may cost tokens.
    """
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="あなたは親切なアシスタントです。短く回答してください。",
        model=MODEL,
        temperature=0.0
    )
    context_id = f"test_continuity_{uuid4()}"

    # First call: tell the model something specific
    async for resp in service.chat_stream(context_id, "test_user", "私の好きな果物はマンゴーです。覚えておいてください。"):
        pass

    first_response_id = await service.response_id_store.get(context_id)
    assert first_response_id is not None, "response_id should be set after first call."

    # Second call: ask about what was said before
    collected_text = []
    async for resp in service.chat_stream(context_id, "test_user", "私の好きな果物は何ですか？"):
        if resp.text:
            collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert "マンゴー" in full_text, f"Context was not maintained. Response: {full_text}"

    # response_id should have been updated
    second_response_id = await service.response_id_store.get(context_id)
    assert second_response_id != first_response_id, "response_id should be updated after second call."

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_openai_responses_service_tool_calls():
    """
    Test OpenAIResponsesService with a registered tool.
    This test actually calls OpenAI Responses API, so it may cost tokens.
    """
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_tool_context_{uuid4()}"

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
        if resp.text:
            collected_text.append(resp.text)
        if resp.tool_call:
            tool_called = True

    full_text = "".join(collected_text)
    assert tool_called, "Tool was not called."
    assert "2" in full_text, f"Answer '2' not found in response: {full_text}"

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_openai_responses_service_tool_calls_response_formatter():
    """
    Test OpenAIResponsesService with a registered tool that has response_formatter.
    The tool result should be formatted directly without a 2nd LLM call.
    This test actually calls OpenAI Responses API, so it may cost tokens.
    """
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_tool_rf_context_{uuid4()}"

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
        if resp.text:
            collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert "Formatter: 1+1の計算結果は2です。" in full_text, f"Direct response not found in text: {full_text}"

    # Verify response_ids is valid after direct response (tool output must be sent to API)
    assert await service.response_id_store.get(context_id) is not None, "response_id should be set after response_formatter call."

    # Verify conversation can continue without "No tool output found" error
    collected_text2 = []
    async for resp in service.chat_stream(context_id, "test_user", "ありがとう"):
        if resp.error_info:
            pytest.fail(f"Error after response_formatter call (response_ids corrupted): {resp.error_info}")
        if resp.text:
            collected_text2.append(resp.text)

    assert len("".join(collected_text2)) > 0, "No response after response_formatter call."

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_openai_responses_service_chained_tool_calls_mixed():
    """
    Test chained tool calls where a direct-response tool (response_formatter) is called first,
    and its result triggers the LLM to call a second tool (normal LLM response).
    Verifies that suppress_output resets correctly so the chained tool's LLM response is yielded.
    This test actually calls OpenAI Responses API, so it may cost tokens.
    """
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=(
            "You have two tools: get_balance and get_campaign_info.\n"
            "When the user asks about campaigns:\n"
            "1. FIRST call get_balance\n"
            "2. If balance >= 1000000, call get_campaign_info\n"
            "3. Tell the user about the campaign based on get_campaign_info result\n"
            "Always follow this exact order. Never skip steps."
        ),
        model=MODEL,
        temperature=0.0
    )
    context_id = f"test_chained_mixed_{uuid4()}"

    # Tool 1: get_balance (with response_formatter = direct response)
    balance_spec = {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Get the user's account balance. Always call this first.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }

    @service.tool(balance_spec)
    async def get_balance() -> Dict[str, Any]:
        return {"balance": 1500000, "currency": "JPY"}

    @service.tools["get_balance"].response_formatter(continue_chain=True)
    def format_balance(result, arguments):
        return f"残高: {result['balance']:,}{result['currency']}\n"

    # Tool 2: get_campaign_info (normal LLM response, no response_formatter)
    campaign_spec = {
        "type": "function",
        "function": {
            "name": "get_campaign_info",
            "description": "Get campaign information for eligible users with balance >= 1,000,000 JPY. Call this after get_balance.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }

    @service.tool(campaign_spec)
    async def get_campaign_info() -> Dict[str, Any]:
        return {"campaign_name": "Premium Gold Campaign", "bonus_rate": "5%"}

    collected_text = []
    tools_called = []

    async for resp in service.chat_stream(context_id, "test_user", "キャンペーンはありますか？"):
        if resp.error_info:
            pytest.fail(f"Error during chained tool call: {resp.error_info}")
        if resp.text:
            collected_text.append(resp.text)
        if resp.tool_call and resp.tool_call.name:
            tools_called.append(resp.tool_call.name)

    full_text = "".join(collected_text)

    # Verify get_balance was called (direct response tool)
    assert "get_balance" in tools_called, f"get_balance was not called. Called: {tools_called}"

    # Verify the direct response text from response_formatter appeared
    assert "残高" in full_text, f"Direct response from get_balance not found in text: {full_text}"

    # Verify get_campaign_info was called (chained tool, normal LLM response)
    assert "get_campaign_info" in tools_called, f"get_campaign_info was not called (chain broken). Called: {tools_called}"

    # Verify the LLM generated a response using the campaign info (not suppressed)
    assert "Premium Gold Campaign" in full_text or "5%" in full_text, \
        f"LLM response about campaign not found (suppressed?): {full_text}"

    # Verify conversation can continue
    collected_text2 = []
    async for resp in service.chat_stream(context_id, "test_user", "ありがとう"):
        if resp.error_info:
            pytest.fail(f"Error after chained tool calls: {resp.error_info}")
        if resp.text:
            collected_text2.append(resp.text)

    assert len("".join(collected_text2)) > 0, "No response after chained tool calls."

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_openai_responses_service_context_isolation():
    """
    Test that conversations with different context_ids are properly isolated
    when called concurrently. This verifies that one user's conversation
    does not leak into another's.
    This test actually calls OpenAI Responses API, so it may cost tokens.
    """
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="あなたは親切なアシスタントです。短く回答してください。",
        model=MODEL,
        temperature=0.0
    )

    context_a = f"test_isolation_a_{uuid4()}"
    context_b = f"test_isolation_b_{uuid4()}"

    # First round: give each context different information
    async def setup_context(ctx_id, message):
        async for _ in service.chat_stream(ctx_id, "test_user", message):
            pass

    await asyncio.gather(
        setup_context(context_a, "私の名前は太郎です。覚えておいてください。"),
        setup_context(context_b, "私の名前は花子です。覚えておいてください。"),
    )

    # Verify each context has its own response_id
    response_id_a = await service.response_id_store.get(context_a)
    response_id_b = await service.response_id_store.get(context_b)
    assert response_id_a is not None
    assert response_id_b is not None
    assert response_id_a != response_id_b, \
        "Different contexts should have different response_ids."

    # Second round: ask each context about the stored information concurrently
    results = {}

    async def ask_name(ctx_id):
        texts = []
        async for resp in service.chat_stream(ctx_id, "test_user", "私の名前は何ですか？"):
            if resp.text:
                texts.append(resp.text)
        results[ctx_id] = "".join(texts)

    await asyncio.gather(
        ask_name(context_a),
        ask_name(context_b),
    )

    assert "太郎" in results[context_a], f"Context A should remember '太郎'. Got: {results[context_a]}"
    assert "花子" in results[context_b], f"Context B should remember '花子'. Got: {results[context_b]}"

    # Cross-check: ensure no leakage
    assert "花子" not in results[context_a], f"Context A should NOT contain '花子'. Got: {results[context_a]}"
    assert "太郎" not in results[context_b], f"Context B should NOT contain '太郎'. Got: {results[context_b]}"

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_openai_responses_service_image():
    """
    Test OpenAIResponsesService with image input.
    This test actually calls OpenAI Responses API, so it may cost tokens.
    """
    service = OpenAIResponsesService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_image_context_{uuid4()}"

    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", "これは何ですか？漢字で答えてください。", files=[{"type": "image", "url": IMAGE_URL}]):
        collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert len(full_text) > 0, "No text was returned from the LLM."
    assert "寿司" in full_text, f"寿司 is not in text: {full_text}"

    await service.openai_client.close()
