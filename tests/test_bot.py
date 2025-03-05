import pytest
import asyncio
import base64
import io
import os
import httpx
import pyautogui
from aiavatar import AIAvatar
from aiavatar.animation import AnimationControllerDummy
from aiavatar.face import FaceControllerDummy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """## 表情

あなたは以下の表情を持っています。

- neutral
- joy
- angry
- sorrow
- fun

特に表情を表現したい場合は、文章の先頭に[face:joy]のように挿入してください。
一覧以外の表情は絶対に表現してはいけません。

例
```
[face:fun]ねえ、海が見えるよ！[face:joy]早く泳ごうよ。
```


## 身振り手振り

あなたは以下の身振り手振りをすることができます。

- joy_hands_up
- angry_hands_on_waist
- sorrow_heads_down
- fun_waving_arm

特に感情を身振り手振りで表現したい場合は、文章に[animation:joy_hands_up]のように挿入してください。
一覧以外の身振り手振りは絶対に表現してはいけません。

例
[animation:joy_hands_up]おーい、こっちだよ！


## 視覚情報

ユーザーとの会話に応答するために視覚情報（画像）が必要な場合は、[vision:camera]のようなタグを応答に含めてください。
画像取得のソースは以下のとおりです。

- screenshot: ユーザーのPC画面を見ることができます
- camera: カメラを通じて現実世界を見ることができます


## 話し方

応答内容は読み上げられます。可能な限り1文で、40文字以内を目処に簡潔に応答してください。
"""


class FaceControllerForTest(FaceControllerDummy):
    def __init__(self, debug = False):
        super().__init__(debug)
        self.histories = []
    
    async def set_face(self, name, duration):
        await super().set_face(name, duration)
        self.histories.append(name)


class AnimationControllerTest(AnimationControllerDummy):
    def __init__(self, animations = None, idling_key = "idling", debug = False):
        super().__init__(animations, idling_key, debug)
        self.animations["joy_hands_up"] = 1
        self.animations["angry_hands_on_waist"] = 2
        self.animations["sorrow_heads_down"] = 3
        self.animations["fun_waving_arm"] = 4
        self.histories = []

    async def animate(self, name, duration):
        await super().animate(name, duration)
        self.histories.append(name)


@pytest.fixture
def aiavatar_app():
    app = AIAvatar(
        openai_api_key=OPENAI_API_KEY,
        openai_model="gpt-4o",
        system_prompt=SYSTEM_PROMPT,
        debug=True
    )
    app.adapter.face_controller = FaceControllerForTest()
    app.adapter.animation_controller = AnimationControllerTest()
    return app


def transcribe(data: bytes, audio_format: str = "wav") -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    form_data = {"model": "whisper-1"}
    files = {"file": (f"voice.{audio_format}", data, f"audio/{audio_format}")}
    resp = httpx.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=headers,
        data=form_data,
        files=files
    )
    return resp.json().get("text")


@pytest.mark.asyncio
async def test_chat(aiavatar_app: AIAvatar):
    await aiavatar_app.chat("こんにちは。", wait_performance=True)
    assert "こんにちは" in aiavatar_app.adapter.last_response.text
    trans_text = transcribe(aiavatar_app.adapter.last_response.audio_data)
    assert "こんにちは" in trans_text or "こんにちわ" in trans_text

    # Context
    await aiavatar_app.chat("旅行で悩んでいます。東京、京都、福岡のいずれかに。", wait_performance=True)
    context_id = aiavatar_app.adapter.last_response.context_id
    await aiavatar_app.chat("おすすめは？", wait_performance=True)
    assert aiavatar_app.adapter.last_response.context_id == context_id
    response_text = aiavatar_app.adapter.last_response.text
    assert "東京" in response_text or "京都" in response_text or "福岡" in response_text
    trans_text = transcribe(aiavatar_app.adapter.last_response.audio_data)
    assert "東京" in trans_text or "京都" in trans_text or "福岡" in trans_text


@pytest.mark.asyncio
async def test_chat_face_animation(aiavatar_app: AIAvatar):
    await aiavatar_app.chat("表情と身振り手振りで喜怒哀楽を表現してください", wait_performance=True)

    face_histories = aiavatar_app.adapter.face_controller.histories
    assert face_histories[0] == "joy"
    assert face_histories[1] == "angry"
    assert face_histories[2] == "sorrow"
    assert face_histories[3] == "fun"

    animation_histories = aiavatar_app.adapter.animation_controller.histories
    assert animation_histories[0] == "joy_hands_up"
    assert animation_histories[1] == "angry_hands_on_waist"
    assert animation_histories[2] == "sorrow_heads_down"
    assert animation_histories[3] == "fun_waving_arm"


@pytest.mark.asyncio
async def test_chat_wakeword(aiavatar_app: AIAvatar):
    aiavatar_app.wakewords = ["こんにちは"]
    aiavatar_app.wakeword_timeout = 10

    # Not triggered chat
    await aiavatar_app.chat("やあ", wait_performance=True)
    assert aiavatar_app.adapter.last_response.type == "final"
    assert aiavatar_app.adapter.last_response.text == ""
    assert aiavatar_app.adapter.last_response.voice_text == ""
    assert aiavatar_app.adapter.last_response.audio_data == b""

    # Start chat
    await aiavatar_app.chat("こんにちは、元気？", wait_performance=True)
    assert "こんにちは" in aiavatar_app.adapter.last_response.text
    # Continue chat not by wakeword
    await aiavatar_app.chat("寿司とラーメンどっちが好き？", wait_performance=True)
    response_text = aiavatar_app.adapter.last_response.text
    assert "寿司" in response_text or "ラーメン" in response_text

    # Wait for wakeword timeout
    await asyncio.sleep(10)

    # Not triggered chat
    await aiavatar_app.chat("そうなんだ", wait_performance=True)
    assert aiavatar_app.adapter.last_response.type == "final"
    assert aiavatar_app.adapter.last_response.text == ""
    assert aiavatar_app.adapter.last_response.voice_text == ""
    assert aiavatar_app.adapter.last_response.audio_data == b""


@pytest.mark.asyncio
async def test_chat_vision(aiavatar_app: AIAvatar):
    @aiavatar_app.adapter.get_image_url
    async def get_image_url(source: str) -> str:
        image_bytes = None

        if source == "screenshot":
            # Capture screenshot
            buffered = io.BytesIO()
            image = pyautogui.screenshot(region=(0, 0, 1280, 720))
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()

        if image_bytes:
            # Upload and get url, or, make base64 encoded url
            b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            b64_url = f"data:image/jpeg;base64,{b64_encoded}"
            return b64_url

    await aiavatar_app.chat("画面を見て。今見えているアプリケーションは何かな？", wait_performance=True)
    assert "visual" in aiavatar_app.adapter.last_response.text.lower()  # Run test on Visual Studio Code


@pytest.mark.asyncio
async def test_chat_function(aiavatar_app: AIAvatar):
    # Register tool
    weather_tool_spec = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
            },
        }
    }
    @aiavatar_app.sts.llm.tool(weather_tool_spec)
    async def get_weather(location: str = None):
        return {"weather": "clear", "temperature": 23.4}

    await aiavatar_app.chat("東京の天気を教えて。", wait_performance=True)
    assert "晴" in aiavatar_app.adapter.last_response.text
    assert "23.4" in aiavatar_app.adapter.last_response.text
