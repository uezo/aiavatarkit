import pytest
from aiavatar.processors import ChatGPTProcessor

@pytest.fixture
def chatgpt_processor():
    return ChatGPTProcessor("YOUR API KEY", temperature=0.0)

@pytest.mark.asyncio
async def test_chat(chatgpt_processor: ChatGPTProcessor):
    resp_iter = chatgpt_processor.chat("おしゃべりしよう")

    async for r in resp_iter:
        assert len(r) > 0

    assert len(chatgpt_processor.histories) == 2

@pytest.mark.asyncio
async def test_reset_histories(chatgpt_processor: ChatGPTProcessor):
    chatgpt_processor.histories.append("a")
    chatgpt_processor.histories.append("b")

    assert len(chatgpt_processor.histories) == 2

    chatgpt_processor.reset_histories()

    assert len(chatgpt_processor.histories) == 0
