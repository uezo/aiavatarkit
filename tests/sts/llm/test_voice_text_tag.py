import os
import pytest
from uuid import uuid4
from aiavatar.sts.llm.chatgpt import ChatGPTService

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """あなたはAIアシスタントです。

応答は必ず以下の形式で出力してください：

<ack>頷き・第一声の発話内容</ack>
<think>思考内容</think>
<answer>応答本体</answer>

### 内容

- 頷き・第一声: 肯定/否定の一言、フィラーなども含む
- 思考内容: 応答に際しての留意事項や応答すべき内容。どんなに短い応答でもまずは必ず考える
- 応答本体: 最終的にユーザーに伝える文章

それ以外の文言は禁止です。
"""


@pytest.mark.asyncio
async def test_single_voice_text_tag():
    """
    Single voice_text_tag (backward compat): only <answer> is vocalized.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5,
        voice_text_tag="answer"
    )
    context_id = f"test_single_tag_{uuid4()}"

    texts = []
    voices = []
    async for resp in service.chat_stream(context_id, "user1", "1+1は？"):
        texts.append(resp.text or "")
        if resp.voice_text:
            voices.append(resp.voice_text)

    full_text = "".join(texts)
    full_voice = "".join(voices)
    assert len(full_text) > 0
    assert "<answer>" in full_text
    assert "<answer>" not in full_voice
    assert "</answer>" not in full_voice
    assert "<think>" not in full_voice
    assert "<ack>" not in full_voice
    print(f"\n[Single tag] text: {full_text}")
    print(f"[Single tag] voice: {full_voice}")

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_multiple_voice_text_tags():
    """
    Multiple voice_text_tags: both <ack> and <answer> are vocalized, <think> is not.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5,
        voice_text_tag=["ack", "answer"]
    )
    context_id = f"test_multi_tag_{uuid4()}"

    texts = []
    voices = []
    async for resp in service.chat_stream(context_id, "user1", "1+1は？"):
        texts.append(resp.text or "")
        if resp.voice_text:
            voices.append(resp.voice_text)

    full_text = "".join(texts)
    full_voice = "".join(voices)
    assert len(full_text) > 0
    assert "<ack>" not in full_voice
    assert "</ack>" not in full_voice
    assert "<answer>" not in full_voice
    assert "</answer>" not in full_voice
    assert "<think>" not in full_voice
    assert "</think>" not in full_voice
    print(f"\n[Multi tag] text: {full_text}")
    print(f"[Multi tag] voice: {full_voice}")

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_multiple_tags_only_one_present():
    """
    Multiple tags registered but prompt only produces <ack> and <answer>.
    voice_text_tag includes 'speech' which won't appear — should still work.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5,
        voice_text_tag=["ack", "answer", "speech"]
    )
    context_id = f"test_multi_partial_{uuid4()}"

    texts = []
    voices = []
    async for resp in service.chat_stream(context_id, "user1", "1+1は？"):
        texts.append(resp.text or "")
        if resp.voice_text:
            voices.append(resp.voice_text)

    full_text = "".join(texts)
    full_voice = "".join(voices)
    assert len(full_text) > 0
    assert len(full_voice) > 0
    assert "<ack>" not in full_voice
    assert "<answer>" not in full_voice
    assert "<think>" not in full_voice
    print(f"\n[Multi tag, partial] text: {full_text}")
    print(f"[Multi tag, partial] voice: {full_voice}")

    await service.openai_client.close()
