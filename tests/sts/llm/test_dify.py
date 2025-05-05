import os
import pytest
from aiavatar.sts.llm.dify import DifyService

DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_API_KEY_AGENT = os.getenv("DIFY_API_KEY_AGENT")
DIFY_URL = os.getenv("DIFY_URL")
IMAGE_URL = os.getenv("IMAGE_URL")

@pytest.mark.asyncio
async def test_dify_service_simple():
    """
    Test DifyService to check if it can stream responses.
    This test actually calls Dify API, so it may cost tokens.
    """
    service = DifyService(
        api_key=DIFY_API_KEY,
        user="user",
        base_url=DIFY_URL
    )
    context_id = "test_context"

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
    assert context_id in service.conversation_ids
    conversation_id = service.conversation_ids[context_id]

    # Call again
    user_message = "あれっ？私、あなたの何を食べたって言ったっけ？"
    collected_text = []
    collected_voice = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        collected_text.append(resp.text)
        collected_voice.append(resp.voice_text)

    # Check the response content
    assert "プリン" in full_text, "'プリン' doesn't appear in text. Context management is incorrect."


@pytest.mark.skip("Skip dify image")
@pytest.mark.asyncio
async def test_dify_service_image():
    """
    Test DifyService to check if it can handle image and stream responses.
    This test actually calls Dify API, so it may cost tokens.
    """
    service = DifyService(
        api_key=DIFY_API_KEY,
        user="user",
        base_url=DIFY_URL
    )
    context_id = "test_context"

    collected_text = []

    async for resp in service.chat_stream(context_id, "test_user", "これは何ですか？漢字で答えてください。", files=[{"type": "image", "url": IMAGE_URL}]):
        collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "寿司" in full_text, "寿司 is not in text."

    # Check the context
    assert context_id in service.conversation_ids


@pytest.mark.asyncio
async def test_dify_service_agent_mode():
    """
    Test DifyService for Agent.
    This test actually calls Dify API, so it may cost tokens.
    """
    service = DifyService(
        api_key=DIFY_API_KEY_AGENT,
        user="user",
        base_url=DIFY_URL,
        is_agent_mode=True
    )
    context_id = "test_context_agent"

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
    assert context_id in service.conversation_ids
    conversation_id = service.conversation_ids[context_id]

    # Call again
    user_message = "あれっ？私、あなたの何を食べたって言ったっけ？"
    collected_text = []
    collected_voice = []
    async for resp in service.chat_stream(context_id, "test_user", user_message):
        collected_text.append(resp.text)
        collected_voice.append(resp.voice_text)

    # Check the response content
    assert "プリン" in full_text, "'プリン' doesn't appear in text. Context management is incorrect."
