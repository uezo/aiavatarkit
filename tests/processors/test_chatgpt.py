import pytest
import io
import time
import pyautogui
from aiavatar.processors.chatgpt import ChatGPTProcessor


async def get_image(source: str=None) -> bytes:
    buffered = io.BytesIO()
    image = pyautogui.screenshot(region=(0, 0, 1280, 720))
    image.save(buffered, format="PNG")
    image.save("test_chatgpt_image.png")   # Save current image for debug
    return buffered.getvalue()


@pytest.fixture
def chatgpt_processor():
    return ChatGPTProcessor(api_key="YOUR_API_KEY", temperature=0.0, history_timeout=5.0)

@pytest.mark.asyncio
async def test_chat(chatgpt_processor: ChatGPTProcessor):
    resp_iter = chatgpt_processor.chat("おしゃべりしよう")
    async for r in resp_iter:
        assert len(r) > 0
    assert len(chatgpt_processor.histories) == 2

    resp_iter = chatgpt_processor.chat("もっとしゃべろう")
    async for r in resp_iter:
        assert len(r) > 0
    assert len(chatgpt_processor.histories) == 4

    # Wait for history timeout
    time.sleep(chatgpt_processor.history_timeout + 1.0)

    resp_iter = chatgpt_processor.chat("さらにしゃべろう")
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

@pytest.mark.asyncio
async def test_build_messages(chatgpt_processor: ChatGPTProcessor):
    # Just current user message
    current_user_message = "current user message"
    messages = await chatgpt_processor.build_messages(current_user_message)
    assert len(messages) == 1
    assert messages[-1]["content"] == current_user_message

    # With system message
    chatgpt_processor.system_message_content = "system message content"
    messages = await chatgpt_processor.build_messages(current_user_message)
    assert len(messages) == 2
    assert messages[0]["content"] == chatgpt_processor.system_message_content
    assert messages[-1]["content"] == current_user_message

    # Make histories
    chatgpt_processor.histories.append({"role": "user", "content": "user message 1"})
    chatgpt_processor.histories.append({"role": "assistant", "content": "assistant message 1"})
    chatgpt_processor.histories.append({"role": "user", "content": "user message 2"})
    chatgpt_processor.histories.append({"role": "assistant", "content": "assistant message 2"})

    # With histories
    chatgpt_processor.system_message_content = None
    messages = await chatgpt_processor.build_messages(current_user_message)
    assert len(messages) == 5
    assert messages[0]["content"] == "user message 1"
    assert messages[-1]["content"] == current_user_message

    # With system message + histories
    chatgpt_processor.system_message_content = "system message content"
    messages = await chatgpt_processor.build_messages(current_user_message)
    assert len(messages) == 6
    assert messages[0]["content"] == chatgpt_processor.system_message_content
    assert messages[1]["content"] == "user message 1"
    assert messages[-1]["content"] == current_user_message

    # After reset histories
    chatgpt_processor.reset_histories()
    messages = await chatgpt_processor.build_messages(current_user_message)
    assert len(messages) == 2
    assert messages[0]["content"] == chatgpt_processor.system_message_content
    assert messages[-1]["content"] == current_user_message

def test_extract_tags(chatgpt_processor: ChatGPTProcessor):
    tags = chatgpt_processor.extract_tags("[face:smile]やっと着いたね！[vision:screenshot]写真を撮ろうよ！")
    assert tags["face"] == "smile"
    assert tags["vision"] == "screenshot"

@pytest.mark.asyncio
async def test_chat_vision(chatgpt_processor: ChatGPTProcessor):
    chatgpt_processor.model = "gpt-4o"
    chatgpt_processor.system_message_content = "ユーザーからの要求を処理するために画像データが必要な場合、[vision:screenshot]を応答メッセージに含めてください。\n\n例\n[vision:screenshot]承知しました。画像を確認しています。"
    chatgpt_processor.use_vision = True
    chatgpt_processor.get_image = get_image
    resp_iter = chatgpt_processor.chat("画面に映っているプログラミング言語は何ですか？")
    response_text = ""
    async for r in resp_iter:
        response_text += r

    assert "Python" in response_text
