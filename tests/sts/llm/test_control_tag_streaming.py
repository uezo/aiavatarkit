import pytest

from aiavatar.sts.llm.base import LLMResponse, LLMServiceDummy


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_chunks",
    [
        [
            '<answer><artifact type="presentation" '
            'src="https://speakerdeck.com/player/b32d37c9c2504d548d792336c291d076?slide=1" />'
            '最初のスライドです。</answer>'
        ],
        [
            '<answer><artifact type="presentation" '
            'src="https://speakerdeck.com/player/b32d37c9c2504d548d792336c291d076?',
            'slide=1" />最初のスライドです。</answer>',
        ],
    ],
)
async def test_punctuation_inside_streamed_control_tag_is_not_spoken(tmp_path, response_chunks):
    service = ChunkedLLMServiceDummy(
        response_chunks=response_chunks,
        system_prompt="test",
        model="dummy",
        voice_text_tag=["ack", "answer"],
        db_connection_str=str(tmp_path / "context.db"),
    )

    text_parts = []
    voice_parts = []
    async for response in service.chat_stream("context", "user", "show slides"):
        text_parts.append(response.text or "")
        voice_parts.append(response.voice_text or "")

    full_text = "".join(text_parts)
    full_voice = "".join(voice_parts)
    assert "<artifact " in full_text
    assert "?slide=1" in full_text
    assert full_voice == "最初のスライドです。"
    assert "artifact" not in full_voice
    assert "speakerdeck.com" not in full_voice


@pytest.mark.asyncio
async def test_math_comparison_is_spoken_as_text(tmp_path):
    service = ChunkedLLMServiceDummy(
        response_chunks=["<answer>x<1かつy>1です。</answer>"],
        system_prompt="test",
        model="dummy",
        voice_text_tag=["answer"],
        db_connection_str=str(tmp_path / "context.db"),
    )

    voice_parts = []
    async for response in service.chat_stream("context", "user", "compare values"):
        voice_parts.append(response.voice_text or "")

    assert "".join(voice_parts) == "x<1かつy>1です。"


@pytest.mark.asyncio
async def test_incomplete_known_control_tag_is_not_spoken(tmp_path):
    service = ChunkedLLMServiceDummy(
        response_chunks=["<answer>こちらです。<artifact type=\"presentation\" src=\"https://example.com?"],
        system_prompt="test",
        model="dummy",
        voice_text_tag=["answer"],
        db_connection_str=str(tmp_path / "context.db"),
    )

    text_parts = []
    voice_parts = []
    async for response in service.chat_stream("context", "user", "show it"):
        text_parts.append(response.text or "")
        voice_parts.append(response.voice_text or "")

    assert "<artifact " in "".join(text_parts)
    assert "".join(voice_parts) == "こちらです。"
