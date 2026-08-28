import logging
import os
import pytest
from uuid import uuid4
from aiavatar.sts.llm.base import LLMResponse, LLMServiceDummy
from aiavatar.sts.llm.chatgpt import ChatGPTService

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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


class ChunkedLLMServiceDummy(LLMServiceDummy):
    def __init__(self, *, response_chunks, **kwargs):
        super().__init__(response_text="", **kwargs)
        self.response_chunks = response_chunks

    async def get_llm_stream_response(
        self,
        context_id,
        user_id,
        messages,
        system_prompt_params=None,
        tools=None,
        inline_llm_params=None,
        session_id=None,
        channel=None,
    ):
        for text in self.response_chunks:
            yield LLMResponse(context_id=context_id, text=text)


async def collect_text_and_voice(service):
    text_parts = []
    voice_parts = []
    async for response in service.chat_stream("context", "user", "test"):
        text_parts.append(response.text or "")
        voice_parts.append(response.voice_text or "")
    return "".join(text_parts), "".join(voice_parts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_chunks",
    [
        ["<answer>first</answer>\n<answer>duplicate</answer>"],
        ["<answer>first</answer>\n", "<answer>duplicate</answer>"],
    ],
    ids=["same_chunk", "later_chunk"],
)
async def test_terminal_voice_text_tag_truncates_after_first_close(
    tmp_path,
    response_chunks,
):
    service = ChunkedLLMServiceDummy(
        response_chunks=response_chunks,
        system_prompt="test",
        model="dummy",
        voice_text_tag=["answer"],
        terminal_voice_text_tag="answer",
        db_connection_str=str(tmp_path / "context.db"),
    )

    full_text, full_voice = await collect_text_and_voice(service)
    histories = await service.context_manager.get_histories("context")

    assert full_text == "<answer>first</answer>"
    assert full_voice == "first"
    assert histories[-1]["content"] == full_text


@pytest.mark.asyncio
async def test_terminal_voice_text_tag_uses_named_tag_with_multiple_voice_tags(tmp_path):
    first_response = (
        "<ack>了解。</ack><think>回答を確認する。</think>"
        "<answer>最初の回答です。</answer>"
    )
    service = ChunkedLLMServiceDummy(
        response_chunks=[
            first_response,
            "garbage<ack>了解。</ack><answer>重複した回答です。</answer>",
        ],
        system_prompt="test",
        model="dummy",
        voice_text_tag=["ack", "answer"],
        terminal_voice_text_tag="answer",
        db_connection_str=str(tmp_path / "context.db"),
    )

    full_text, full_voice = await collect_text_and_voice(service)

    assert full_text == first_response
    assert full_voice == "了解。最初の回答です。"


@pytest.mark.asyncio
async def test_terminal_voice_text_tag_warns_once_with_discarded_buffer(tmp_path, caplog):
    service = ChunkedLLMServiceDummy(
        response_chunks=[
            "<answer>first</answer>same-chunk suffix",
            "second chunk",
            "third chunk",
        ],
        system_prompt="test",
        model="dummy",
        voice_text_tag=["answer"],
        terminal_voice_text_tag="answer",
        db_connection_str=str(tmp_path / "context.db"),
    )

    with caplog.at_level(logging.WARNING, logger="aiavatar.sts.llm.base"):
        await collect_text_and_voice(service)

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "aiavatar.sts.llm.base"
        and record.levelno == logging.WARNING
        and record.getMessage().startswith("Discarded LLM response text")
    ]
    assert warnings == [
        "Discarded LLM response text after terminal voice text tag: "
        "context_id=context, tag=answer, "
        "discarded_text='same-chunk suffixsecond chunkthird chunk'",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_chunks",
    [
        ["<ans", "wer>first</answer>\n<answer>duplicate</answer>"],
        ["<answer>first</ans", "wer>\n<answer>duplicate</answer>"],
    ],
    ids=["split_open", "split_close"],
)
async def test_terminal_voice_text_tag_detects_split_tag(tmp_path, response_chunks):
    service = ChunkedLLMServiceDummy(
        response_chunks=response_chunks,
        system_prompt="test",
        model="dummy",
        voice_text_tag=["answer"],
        terminal_voice_text_tag="answer",
        db_connection_str=str(tmp_path / "context.db"),
    )

    full_text, full_voice = await collect_text_and_voice(service)

    assert full_text == "<answer>first</answer>"
    assert full_voice == "first"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_kwargs",
    [{}, {"terminal_voice_text_tag": None}],
    ids=["default", "explicit_none"],
)
async def test_terminal_voice_text_tag_disabled_preserves_duplicate(
    tmp_path,
    terminal_kwargs,
):
    response_text = "<answer>first</answer>\n<answer>duplicate</answer>"
    service = ChunkedLLMServiceDummy(
        response_chunks=[response_text],
        system_prompt="test",
        model="dummy",
        voice_text_tag=["answer"],
        db_connection_str=str(tmp_path / "context.db"),
        **terminal_kwargs,
    )

    full_text, full_voice = await collect_text_and_voice(service)

    assert full_text == response_text
    assert full_voice == "firstduplicate"


@pytest.mark.asyncio
async def test_terminal_voice_text_tag_missing_close_preserves_response(tmp_path):
    response_text = "<answer>終端タグがありません。後続も保持されます。"
    service = ChunkedLLMServiceDummy(
        response_chunks=[response_text],
        system_prompt="test",
        model="dummy",
        voice_text_tag=["answer"],
        terminal_voice_text_tag="answer",
        db_connection_str=str(tmp_path / "context.db"),
    )

    full_text, full_voice = await collect_text_and_voice(service)

    assert full_text == response_text
    assert full_voice == "終端タグがありません。後続も保持されます。"


@pytest.mark.asyncio
async def test_single_voice_text_tag():
    """
    Single voice_text_tag (backward compat): only <answer> is vocalized.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        reasoning_effort="none",
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
        reasoning_effort="none",
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
        reasoning_effort="none",
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
