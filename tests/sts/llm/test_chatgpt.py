import asyncio
import json
import os
import pytest
from typing import Any, Dict, AsyncGenerator, Tuple
from uuid import uuid4
from aiavatar.sts.llm import Guardrail, GuardrailRespose
from aiavatar.sts.llm.chatgpt import ChatGPTService, ToolCall

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMAGE_URL = os.getenv("IMAGE_URL")
MODEL = "gpt-4.1"

XAI_MODEL = "grok-4-1-fast-non-reasoning"
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL")


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
async def test_chatgpt_service_simple():
    """
    Test ChatGPTService with a basic prompt to check if it can stream responses.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
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

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "assistant" for m in messages), "Assistant message not found in context."

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_system_prompt_params():
    """
    Test ChatGPTService with a basic prompt and its dynamic params.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
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

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_image():
    """
    Test ChatGPTService with a basic prompt to check if it can handle image and stream responses.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
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

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_cot():
    """
    Test ChatGPTService with a prompt to check Chain-of-Thought.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT_COT,
        model=MODEL,
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
async def test_chatgpt_service_with_initial_messages():
    """
    Test ChatGPTService with initial messages (few-shot examples).
    This test actually calls OpenAI API, so it may cost tokens.
    """
    initial_messages = [
        {"role": "user", "content": "今日のランチ、寿司かラーメンかで悩んでる。"},
        {"role": "assistant", "content": "どちらも美味しいですね。"},
    ]

    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
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
    assert messages[0]["content"] == user_message
    assert messages[1]["role"] == "assistant"
    assert "寿司" in messages[1]["content"] or "ラーメン" in messages[1]["content"]

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_tool_calls():
    """
    Test ChatGPTService with a registered tool.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
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
async def test_chatgpt_service_tool_calls_iter():
    """
    Test ChatGPTService with a registered tool that returns iteration.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_tool_iter_context_{uuid4()}"

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

@pytest.mark.asyncio
async def test_chatgpt_guardrails():
    """
    Test ChatGPTService with guardrails.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
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

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_compat_simple():
    """
    Test ChatGPTService with a basic prompt to check if it can stream responses.
    This test actually calls xAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=XAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=XAI_MODEL,
        base_url=XAI_BASE_URL,
        temperature=0.5
    )
    context_id = f"test_compat_context_{uuid4()}"

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
async def test_chatgpt_service_compat_tool_calls():
    """
    Test ChatGPTService with a registered tool.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ChatGPTService(
        openai_api_key=XAI_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=XAI_MODEL,
        base_url=XAI_BASE_URL,
        temperature=0.5
    )
    context_id = f"test_compat_tool_context_{uuid4()}"

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

    user_message = "次の問題を解いて: 1+1\n必ずsolve_mathを使用すること。"
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
async def test_chatgpt_service_split_by_period():
    """
    Test that responses are split by period (。).
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="ユーザーの指示に従い、入力内容を一字一句変えずに復唱してください。",
        model=MODEL,
        temperature=0.0
    )
    context_id = f"test_split_period_{uuid4()}"

    # Two sentences separated by period
    user_message = "以下を一字一句変えずに復唱して：今日は晴れです。明日は雨です。"

    collected_responses = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.voice_text:
            collected_responses.append(resp.voice_text)

    # Should be split into at least 2 segments
    assert len(collected_responses) >= 2, f"Expected at least 2 segments, got {len(collected_responses)}: {collected_responses}"

    # Verify content: first segment should end with 。 and contain first sentence
    assert "今日は晴れです。" in collected_responses[0], f"First segment should contain '今日は晴れです。', got: {collected_responses[0]}"
    # Second segment should contain second sentence
    assert "明日は雨です" in collected_responses[1], f"Second segment should contain '明日は雨です', got: {collected_responses[1]}"

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_split_by_comma_when_long():
    """
    Test that long responses are split by comma (、) when exceeding option_split_threshold.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="ユーザーの指示に従い、入力内容を一字一句変えずに復唱してください。",
        model=MODEL,
        temperature=0.0,
        option_split_threshold=20  # Set low threshold to trigger comma split
    )
    context_id = f"test_split_comma_{uuid4()}"

    # Long sentence with comma, no period - should split at comma due to length
    user_message = "以下を一字一句変えずに復唱して：これはとても長い文章で読点を含んでいます、そしてまだまだ続きます"

    collected_responses = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.voice_text:
            collected_responses.append(resp.voice_text)

    # Should be split at comma because it exceeds threshold
    assert len(collected_responses) >= 2, f"Expected at least 2 segments due to comma split, got {len(collected_responses)}: {collected_responses}"

    # Verify content: first segment should end with 、 (split at comma)
    assert collected_responses[0].endswith("、"), f"First segment should end with '、', got: {collected_responses[0]}"
    # Second segment should contain the rest
    assert "そしてまだまだ続きます" in collected_responses[-1], f"Last segment should contain 'そしてまだまだ続きます', got: {collected_responses[-1]}"

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_no_comma_split_when_tag_makes_it_long():
    """
    Test that control tags are excluded from length calculation,
    so comma split is not triggered when only the tag makes the buffer long.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="""ユーザーの指示に従い、入力内容を一字一句変えずに復唱してください。
ただし、復唱する前に必ず[face:happy][mood:excellent][tone:cheerful]というタグをつけてください。""",
        model=MODEL,
        temperature=0.0,
        option_split_threshold=20  # Threshold: tag included=52 chars (exceeds), tag excluded=6 chars (under)
    )
    context_id = f"test_split_tag_{uuid4()}"

    # Short sentence with comma - multiple tags make buffer long but actual voice text is short
    # "[face:happy][mood:excellent][tone:cheerful]短い文、続き" = 46 chars total (exceeds threshold of 20)
    # "短い文、続き" = 6 chars (under threshold of 20)
    user_message = "以下を一字一句変えずに復唱して：短い文、続き"

    collected_text = []
    collected_voice = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.text:
            collected_text.append(resp.text)
        if resp.voice_text:
            collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)

    # Verify that text contains control tags
    assert "[face:happy]" in full_text, f"Text should contain '[face:happy]' tag, got: {full_text}"
    assert "[mood:excellent]" in full_text, f"Text should contain '[mood:excellent]' tag, got: {full_text}"
    assert "[tone:cheerful]" in full_text, f"Text should contain '[tone:cheerful]' tag, got: {full_text}"

    # Should NOT be split at comma because voice text is short (tag is excluded from length)
    assert len(collected_voice) == 1, f"Expected 1 segment (no comma split for short voice text), got {len(collected_voice)}: {collected_voice}"

    # Verify voice_text does not contain control tags
    assert "[face:happy]" not in collected_voice[0], f"Voice text should not contain tags, got: {collected_voice[0]}"

    # Verify content: the single segment should contain the full text with comma intact
    assert "短い文、続き" in collected_voice[0], f"Segment should contain '短い文、続き' (not split at comma), got: {collected_voice[0]}"

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_split_on_control_tags():
    """
    Test that responses are split before control tags [xxx:yyy] when split_on_control_tags=True.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="""ユーザーの指示に従い、入力内容を一字一句変えずに復唱してください。""",
        model=MODEL,
        temperature=0.0,
        split_on_control_tags=True  # Default, but explicit for clarity
    )
    context_id = f"test_split_control_tags_{uuid4()}"

    # Text with control tags - should split before each tag
    user_message = "以下を一字一句変えずに復唱して：[face:joy]海が見えたよ[face:fun]早く泳ごうよ"

    collected_text = []
    collected_voice = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.text:
            collected_text.append(resp.text)
        if resp.voice_text:
            collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)

    # Verify that text contains control tags
    assert "[face:joy]" in full_text, f"Text should contain '[face:joy]' tag, got: {full_text}"
    assert "[face:fun]" in full_text, f"Text should contain '[face:fun]' tag, got: {full_text}"

    # Should be split into 2 segments (before each control tag)
    assert len(collected_voice) >= 2, f"Expected at least 2 segments, got {len(collected_voice)}: {collected_voice}"

    # First segment should contain first sentence
    assert "海が見えたよ" in collected_voice[0], f"First segment should contain '海が見えたよ', got: {collected_voice[0]}"

    # Second segment should contain second sentence
    assert "早く泳ごうよ" in collected_voice[1], f"Second segment should contain '早く泳ごうよ', got: {collected_voice[1]}"

    # Voice text should not contain control tags
    assert "[face:joy]" not in collected_voice[0], f"Voice text should not contain tags, got: {collected_voice[0]}"
    assert "[face:fun]" not in collected_voice[1], f"Voice text should not contain tags, got: {collected_voice[1]}"

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_no_split_on_tts_tags():
    """
    Test that TTS tags [xxx] (without colon) are NOT split and remain in voice_text,
    while control tags [xxx:yyy] are split and removed from voice_text.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="""ユーザーの指示に従い、入力内容を一字一句変えずに復唱してください。""",
        model=MODEL,
        temperature=0.0,
        split_on_control_tags=True
    )
    context_id = f"test_no_split_tts_tags_{uuid4()}"

    # Text with both TTS tags [xxx] and control tags [xxx:yyy]
    # Should split on control tag but NOT on TTS tags
    user_message = "以下を一字一句変えずに復唱して：[happy]こんにちは[face:joy]元気です[sad]さようなら"

    collected_text = []
    collected_voice = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.text:
            collected_text.append(resp.text)
        if resp.voice_text:
            collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)
    full_voice = "".join(collected_voice)

    # Verify that text contains both TTS tags and control tags
    assert "[happy]" in full_text, f"Text should contain '[happy]' tag, got: {full_text}"
    assert "[face:joy]" in full_text, f"Text should contain '[face:joy]' tag, got: {full_text}"
    assert "[sad]" in full_text, f"Text should contain '[sad]' tag, got: {full_text}"

    # TTS tags should remain in voice_text (not removed)
    assert "[happy]" in full_voice, f"Voice text should contain '[happy]' TTS tag, got: {full_voice}"
    assert "[sad]" in full_voice, f"Voice text should contain '[sad]' TTS tag, got: {full_voice}"

    # Control tags should be removed from voice_text
    assert "[face:joy]" not in full_voice, f"Voice text should NOT contain '[face:joy]' control tag, got: {full_voice}"

    # Should be split into 2 segments (split on control tag [face:joy], not on TTS tags)
    assert len(collected_voice) == 2, f"Expected 2 segments (split on control tag only), got {len(collected_voice)}: {collected_voice}"

    # First segment should contain first TTS tag and first sentence
    assert "[happy]" in collected_voice[0], f"First segment should contain '[happy]', got: {collected_voice[0]}"
    assert "こんにちは" in collected_voice[0], f"First segment should contain 'こんにちは', got: {collected_voice[0]}"

    # Second segment should contain second TTS tag and second sentence
    assert "[sad]" in collected_voice[1], f"Second segment should contain '[sad]', got: {collected_voice[1]}"
    assert "さようなら" in collected_voice[1], f"Second segment should contain 'さようなら', got: {collected_voice[1]}"

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_split_on_control_tags_disabled():
    """
    Test that control tags are NOT split when split_on_control_tags=False.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="""ユーザーの指示に従い、入力内容を一字一句変えずに復唱してください。""",
        model=MODEL,
        temperature=0.0,
        split_on_control_tags=False  # Disable splitting on control tags
    )
    context_id = f"test_split_control_tags_disabled_{uuid4()}"

    # Text with control tags - should NOT split because option is disabled
    user_message = "以下を一字一句変えずに復唱して：[face:joy]海が見えたよ[face:fun]早く泳ごう"

    collected_text = []
    collected_voice = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        if resp.text:
            collected_text.append(resp.text)
        if resp.voice_text:
            collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)

    # Verify that text contains control tags
    assert "[face:joy]" in full_text, f"Text should contain '[face:joy]' tag, got: {full_text}"
    assert "[face:fun]" in full_text, f"Text should contain '[face:fun]' tag, got: {full_text}"

    # Should be 1 segment because split_on_control_tags is disabled
    assert len(collected_voice) == 1, f"Expected 1 segment (split_on_control_tags=False), got {len(collected_voice)}: {collected_voice}"

    # Voice text should contain both sentences together
    assert "海が見えたよ" in collected_voice[0], f"Voice text should contain '海が見えたよ', got: {collected_voice[0]}"
    assert "早く泳ごう" in collected_voice[0], f"Voice text should contain '早く泳ごう', got: {collected_voice[0]}"

    await service.openai_client.close()
