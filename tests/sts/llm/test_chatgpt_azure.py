import asyncio
import json
import os
import pytest
from typing import Any, Dict, AsyncGenerator, Tuple
from uuid import uuid4
from aiavatar.sts.llm.chatgpt import ChatGPTService, ToolCall, Tool

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
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
async def test_chatgpt_azure_service_simple():
    """
    Test ChatGPTService with a basic prompt to check if it can stream responses.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        system_prompt=SYSTEM_PROMPT,
        model="azure",
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

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_azure_service_system_prompt_params():
    """
    Test ChatGPTService with a basic prompt and its dynamic params.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        system_prompt="あなたは{animal_name}です。語尾をそれらしくしてください。カタカナで表現します。",
        model="azure",
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

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_azure_service_image():
    """
    Test ChatGPTService with a basic prompt to check if it can handle image stream responses.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        system_prompt=SYSTEM_PROMPT,
        model="azure",
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

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_azure_service_cot():
    """
    Test ChatGPTService with a prompt to check Chain-of-Thought.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        system_prompt=SYSTEM_PROMPT_COT,
        model="azure",
        temperature=0.5,
        voice_text_tag="answer"
    )
    context_id = f"test_cot_context_{uuid4()}"

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

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_azure_service_tool_calls():
    """
    Test ChatGPTService with a registered tool.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ChatGPTService(
        openai_api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model="azure",
        temperature=0.5
    )
    context_id = f"test_tool_context_{uuid4()}"

    # Register tool
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

    user_message = "次の問題を解いて: 1+1"
    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", user_message):
        collected_text.append(resp.text)

    # Check context
    messages = await service.context_manager.get_histories(context_id)
    assert len(messages) == 4

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == user_message

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"] is not None
    assert messages[1]["tool_calls"][0]["function"]["name"] == "solve_math"
    tool_call_id = messages[1]["tool_calls"][0]["id"]

    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == tool_call_id
    assert messages[2]["content"] == json.dumps({"answer": 2})

    assert messages[3]["role"] == "assistant"
    assert "2" in messages[3]["content"]

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_azure_service_tool_calls_iter():
    """
    Test ChatGPTService with a registered tool that returns iteration.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ChatGPTService(
        openai_api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model="azure",
        temperature=0.5
    )
    context_id = f"test_tool_context_iter_{uuid4()}"

    # Register tool
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

    user_message = "次の問題を解いて: 1+1"
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
    assert messages[0]["content"] == user_message

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"] is not None
    assert messages[1]["tool_calls"][0]["function"]["name"] == "solve_math"
    tool_call_id = messages[1]["tool_calls"][0]["id"]

    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == tool_call_id
    assert messages[2]["content"] == json.dumps({"answer": 2})

    assert messages[3]["role"] == "assistant"
    assert "2" in messages[3]["content"]

    await service.openai_client.close()
