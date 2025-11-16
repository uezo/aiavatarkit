import asyncio
import json
import os
from typing import Any, Dict, AsyncGenerator, Tuple
from uuid import uuid4
import pytest
from aiavatar.sts.llm import Guardrail, GuardrailRespose
from aiavatar.sts.llm.claude import ClaudeService, ToolCall

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
MODEL = "claude-haiku-4-5"
IMAGE_URL = os.getenv("IMAGE_URL")

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

SYSTEM_PROMPT_COT = SYSTEM_PROMPT + """

## 思考について

応答する前に内容をよく考えてください。これまでの文脈を踏まえて適切な内容か、または兄が言い淀んだだけなので頷くだけにするか、など。
まずは考えた内容を<thinking>〜</thinking>に出力してください。
そのあと、発話すべき内容を<answer>〜</answer>に出力してください。
その2つのタグ以外に文言を含むことは禁止です。
"""


@pytest.mark.asyncio
async def test_claude_service_simple():
    """
    Test ClaudeService with a basic prompt to check if it can stream responses.
    This test actually calls Anthropic API, so it may cost tokens.
    """
    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
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

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "assistant" for m in messages), "Assistant message not found in context."



@pytest.mark.asyncio
async def test_claude_service_system_prompt_params():
    """
    Test ClaudeService with a basic prompt and its dynamic params.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
        system_prompt="あなたは{animal_name}です。語尾をそれらしくしてください。カタカナで表現します。",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_system_prompt_params_context_{uuid4()}"

    user_message = "こんにちは"

    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", user_message, system_prompt_params={"animal_name": "猫"}):
        collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "ニャ" in full_text, "ニャ doesn't appear in text."

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "assistant" for m in messages), "Assistant message not found in context."

    await service.anthropic_client.close()


@pytest.mark.asyncio
async def test_claude_service_image():
    """
    Test ClaudeService with a basic prompt to check if it can handle image and stream responses.
    This test actually calls Anthropic API, so it may cost tokens.
    """
    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_context_{uuid4()}"

    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", "これは何ですか？漢字で答えてください。", files=[{"type": "image", "url": IMAGE_URL}]):
        collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "寿司" in full_text, "寿司 is not in text."

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "assistant" for m in messages), "Assistant message not found in context."


@pytest.mark.asyncio
async def test_claude_service_cot():
    """
    Test ClaudeService with a prompt to check Chain-of-Thought.
    This test actually calls Anthropic API, so it may cost tokens.
    """
    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
        system_prompt=SYSTEM_PROMPT_COT,
        model=MODEL,
        temperature=0.5,
        voice_text_tag="answer"
    )
    context_id = f"test_context_cot_{uuid4()}"

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

    # Check the response content (CoT)
    assert "<answer>" in full_text, "Answer tag doesn't appear in text."
    assert "</answer>" in full_text, "Answer tag closing doesn't appear in text."
    assert "<answer>" not in full_voice, "Answer tag was not removed from voice_text."
    assert "</answer>" not in full_voice, "Answer tag closing was not removed from voice_text."

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "assistant" for m in messages), "Assistant message not found in context."


@pytest.mark.asyncio
async def test_claude_service_with_initial_messages():
    """
    Test ClaudeService with initial messages (few-shot examples).
    This test actually calls Anthropic API, so it may cost tokens.
    """
    initial_messages = [
        {"role": "user", "content": "今日のランチ、寿司かラーメンかで悩んでる。"},
        {"role": "assistant", "content": "どちらも美味しいですね。"},
    ]

    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
        system_prompt="グルメアドバイザーとして振る舞ってください。",
        model=MODEL,
        temperature=0.5,
        initial_messages=initial_messages
    )
    context_id = f"test_initial_context_{uuid4()}"

    # Ask the recommendation for lunch
    user_message = "おすすめは？"

    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", user_message):
        collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check if the response contains sushi or ramen
    assert "寿司" in full_text or "ラーメン" in full_text, "Food name from initial messages not found in response."

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert len(messages) == 2, "Expected 2 messages (1 user + 1 assistant, without system and initial 2 messages)"

    # Verify messages
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == user_message
    assert messages[1]["role"] == "assistant"
    assert "寿司" in messages[1]["content"][0]["text"] or "ラーメン" in messages[1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_claude_service_tool_calls():
    """
    Test ClaudeService with a registered tool.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_context_tool_{uuid4()}"

    # Register tool
    tool_spec = {
        "name": "solve_math",
        "description": "Solve simple math problems",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"}
            },
            "required": ["problem"]
        }
    }
    @service.tool(tool_spec)
    async def solve_math(problem: str) -> Dict[str, Any]:
        """
        Tool function example: parse the problem and return a result.
        """
        if problem.strip() == "1+1":
            return {"answer": 2}
        else:
            return {"answer": "unknown"}

    @service.on_before_tool_calls
    async def on_before_tool_calls(tool_calls: list[ToolCall]):
        assert len(tool_calls) > 0

    user_message = "次の問題を解いて。必ずsolve_mathを使用すること: 1+1"
    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", user_message):
        collected_text.append(resp.text)

    # Check context
    messages = await service.context_manager.get_histories(context_id)
    assert len(messages) == 4

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": user_message}]

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[1]["content"][0]["name"] == "solve_math"
    tool_use_id = messages[1]["content"][0]["id"]

    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == tool_use_id
    assert messages[2]["content"][0]["content"] == json.dumps({"answer": 2})

    assert messages[3]["role"] == "assistant"
    assert "2" in messages[3]["content"][0]["text"]


@pytest.mark.asyncio
async def test_claude_service_tool_calls_iter():
    """
    Test ClaudeService with a registered tool that returns iteration.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_context_tool_iter_{uuid4()}"

    # Register tool
    tool_spec = {
        "name": "solve_math",
        "description": "Solve simple math problems",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"}
            },
            "required": ["problem"]
        }
    }
    @service.tool(tool_spec)
    async def solve_math(problem: str) -> AsyncGenerator[Tuple[Dict[str, Any], bool], None]:
        """
        Tool function example: parse the problem and return a result.
        """
        yield {"message": "It takes long time..."}, False
        await asyncio.sleep(3)
        yield {"message": "Solved! Preparing answer..."}, False
        if problem.strip() == "1+1":
            yield {"answer": 2}, True
        else:
            yield {"answer": "unknown"}, True

    @service.on_before_tool_calls
    async def on_before_tool_calls(tool_calls: list[ToolCall]):
        assert len(tool_calls) > 0

    user_message = "次の問題を解いて。必ずsolve_mathを使用すること: 1+1"
    collected_text = []

    progress = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.tool_call:
            if resp.tool_call.result and not resp.tool_call.result.is_final:
                progress.append(resp.tool_call.result.data)
        collected_text.append(resp.text)

    # Check progress
    assert len(progress) == 2
    assert progress[0]["message"] == "It takes long time..."
    assert progress[1]["message"] == "Solved! Preparing answer..."

    # Check context
    messages = await service.context_manager.get_histories(context_id)
    assert len(messages) == 4

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": user_message}]

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[1]["content"][0]["name"] == "solve_math"
    tool_use_id = messages[1]["content"][0]["id"]

    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == tool_use_id
    assert messages[2]["content"][0]["content"] == json.dumps({"answer": 2})

    assert messages[3]["role"] == "assistant"
    assert "2" in messages[3]["content"][0]["text"]


@pytest.mark.asyncio
async def test_claude_service_guardrails():
    """
    Test ClaudeService with guardrails.
    """
    service = ClaudeService(
        anthropic_api_key=CLAUDE_API_KEY,
        system_prompt="ユーザーの指示に従い、入力内容を復唱してください。",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_context_guardrails_{uuid4()}"

    # Define guardrails
    class RequestGuardrail(Guardrail):
        async def apply(self, context_id, user_id, text, files = None, system_prompt_params = None):
            if text == "問題のある入力":
                return GuardrailRespose(
                    guardrail_name=self.name,
                    is_triggered=True,
                    action="block",
                    text="問題のある入力を遮断しました"
                )
            elif text == "hello":
                return GuardrailRespose(
                    guardrail_name=self.name,
                    is_triggered=True,
                    action="replace",
                    text="こんにちは"
                )
            else:
                return GuardrailRespose(
                    guardrail_name=self.name,
                    is_triggered=False
                )

    class ResponseGuardrail(Guardrail):
        async def apply(self, context_id, user_id, text, files = None, system_prompt_params = None):
            if "ラーメン" in text:
                return GuardrailRespose(
                    guardrail_name=self.name,
                    is_triggered=True,
                    action="replace",
                    text="問題のある出力を遮断しました"
                )
            else:
                return GuardrailRespose(
                    guardrail_name=self.name,
                    is_triggered=False
                )

    service.guardrails.append(RequestGuardrail(applies_to="request"))
    service.guardrails.append(ResponseGuardrail(applies_to="response"))

    # Doesn't trigger guardrails
    response_text = ""
    async for resp in service.chat_stream(context_id, "test_user", "こんにちは"):
        response_text += resp.text
    assert "こんにちは" in response_text

    # Trigger RequestGuardrail:block
    response_text = ""
    async for resp in service.chat_stream(context_id, "test_user", "問題のある入力"):
        response_text += resp.text
    assert response_text == "問題のある入力を遮断しました"

    # Trigger RequestGuardrail:replace
    response_text = ""
    async for resp in service.chat_stream(context_id, "test_user", "hello"):
        response_text += resp.text
    assert "こんにちは" in response_text

    # Trigger ResponseGuardrail:replace
    response_text = ""
    async for resp in service.chat_stream(context_id, "test_user", "ラーメン"):
        response_text += resp.text
    assert "ラーメン" in response_text
    assert "問題のある出力を遮断しました" in response_text
