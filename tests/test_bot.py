import pytest
import asyncio
import base64
import io
import os
from time import time
from uuid import uuid4
import httpx
import numpy
import pyautogui
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar import AIAvatar, AIAvatarResponse
from aiavatar.animation import AnimationControllerDummy
from aiavatar.face import FaceControllerDummy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_VOICE_SAMPLE_RATE = 24000 # VOICEVOX
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
    return AIAvatar(
        face_controller=FaceControllerForTest(),
        animation_controller=AnimationControllerTest(),
        openai_api_key=OPENAI_API_KEY,
        openai_model="gpt-4o",
        system_prompt=SYSTEM_PROMPT,
        debug=True
    )


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


async def chat(app: AIAvatar, text: str, session_id: str, context_id: str = None) -> AIAvatarResponse:
    # TTS for input audio instead of human's speech
    async def get_input_voice(text: str):
        voicevox_for_input = VoicevoxSpeechSynthesizer(
            base_url="http://127.0.0.1:50021",
            speaker=2,
            debug=True
        )
        voice = await voicevox_for_input.synthesize(text)
        silence_samples = int(INPUT_VOICE_SAMPLE_RATE * 1.0)
        silence_bytes = numpy.zeros(silence_samples, dtype=numpy.int16).tobytes()
        return voice + silence_bytes

    await app.send_microphone_data(
        await get_input_voice(text),
        session_id=session_id
    )

    # Wait for processing responses
    start_time = time()
    while len(app.last_responses) == 0 or app.last_responses[-1].type != "final":
        await asyncio.sleep(0.1)
        if time() - start_time > 60:
            print("Response timeout (60 sec)")
            break
    while not app.response_queue.empty():
        await asyncio.sleep(0.1)
    while app.audio_player.is_playing:
        await asyncio.sleep(0.1)

    last_response = app.last_responses[-1]
    last_response.audio_data = b""

    # Add audio data
    for r in app.last_responses:
        if r.audio_data:
            last_response.audio_data = r.audio_data
            break

    app.last_responses.clear()
    return last_response

@pytest.mark.asyncio
async def test_chat(aiavatar_app: AIAvatar):
    session_id = f"test_chat_session_{str(uuid4())}"
    user_id = "test_chat_user"
    task = asyncio.create_task(aiavatar_app.start_listening(session_id=session_id, user_id=user_id))

    try:
        # Just chat
        response = await chat(aiavatar_app, text="こんにちは", session_id=session_id)
        assert "こんにちは" in response.text
        assert response.context_id is not None
        context_id = response.context_id

        # Context
        await chat(aiavatar_app, text="旅行で悩んでいます。東京、京都、福岡のいずれかに。", session_id=session_id, context_id=context_id)
        response = await chat(aiavatar_app, text="おすすめはどこ？場所だけ答えて。それ以外は何も言わないで", session_id=session_id, context_id=context_id)
        assert "東京" in response.text or "京都" in response.text or "福岡" in response.text
        trans_text = transcribe(response.audio_data)
        assert response.audio_data != b""
        assert "東京" in trans_text or "京都" in trans_text or "福岡" in trans_text

    finally:
        await aiavatar_app.stop_listening(session_id)
        task.cancel()


@pytest.mark.asyncio
async def test_chat_face_animation(aiavatar_app: AIAvatar):
    session_id = f"test_chat_face_animation_session_{str(uuid4())}"
    user_id = "test_chat_face_animation_user"
    task = asyncio.create_task(aiavatar_app.start_listening(session_id=session_id, user_id=user_id))

    try:
        response = await chat(aiavatar_app, text="表情と身振り手振りで喜怒哀楽を表現してください", session_id=session_id)
        print(f"response: {response.text}")

        face_histories = aiavatar_app.face_controller.histories
        assert face_histories[0] == "joy"
        assert face_histories[1] == "angry"
        assert face_histories[2] == "sorrow"
        assert face_histories[3] == "fun"

        animation_histories = aiavatar_app.animation_controller.histories
        assert animation_histories[0] == "joy_hands_up"
        assert animation_histories[1] == "angry_hands_on_waist"
        assert animation_histories[2] == "sorrow_heads_down"
        assert animation_histories[3] == "fun_waving_arm"

    finally:
        await aiavatar_app.stop_listening(session_id)
        task.cancel()


@pytest.mark.asyncio
async def test_chat_wakeword(aiavatar_app: AIAvatar):
    session_id = f"test_chat_wakeword_session_{str(uuid4())}"
    user_id = "test_chat_wakeword_user"
    task = asyncio.create_task(aiavatar_app.start_listening(session_id=session_id, user_id=user_id))

    aiavatar_app.sts.wakewords = ["こんにちは"]
    aiavatar_app.sts.wakeword_timeout = 10

    try:
        # Not triggered chat
        response = await chat(aiavatar_app, text="やあ", session_id=session_id)
        assert response.type == "final"
        assert response.text == ""
        assert response.voice_text == ""
        assert response.audio_data == b""
        assert response.context_id is None

        # Start chat
        response = await chat(aiavatar_app, text="こんにちは、元気？", session_id=session_id)
        assert "こんにちは" in response.text
        context_id = response.context_id

        # Continue chat not by wakeword
        response = await chat(aiavatar_app, text="寿司とラーメンどっちが好き？", session_id=session_id, context_id=context_id)
        assert "寿司" in response.text or "ラーメン" in response.text

        # Wait for wakeword timeout
        await asyncio.sleep(10)

        # Not triggered chat
        response = await chat(aiavatar_app, text="そうなんだ", session_id=session_id, context_id=context_id)
        assert response.type == "final"
        assert response.text == ""
        assert response.voice_text == ""
        assert response.audio_data == b""
        assert response.context_id == context_id    # Context is still alive

    finally:
        await aiavatar_app.stop_listening(session_id)
        task.cancel()


@pytest.mark.asyncio
async def test_chat_vision(aiavatar_app: AIAvatar):
    session_id = f"test_chat_vision_session_{str(uuid4())}"
    user_id = "test_chat_wakeword_user"
    task = asyncio.create_task(aiavatar_app.start_listening(session_id=session_id, user_id=user_id))

    try:
        @aiavatar_app.get_image_url
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

        # Check `aiavatar_app.last_response`, not response from chat
        response = await chat(aiavatar_app, text="画面を見て。今見えているアプリケーションは何かな？", session_id=session_id)
        assert "visual" in response.text.lower()  # Run test on Visual Studio Code

    finally:
        await aiavatar_app.stop_listening(session_id)
        task.cancel()


@pytest.mark.asyncio
async def test_chat_function(aiavatar_app: AIAvatar):
    session_id = f"test_chat_function_session_{str(uuid4())}"
    user_id = "test_chat_function_user"
    task = asyncio.create_task(aiavatar_app.start_listening(session_id=session_id, user_id=user_id))

    try:
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

        response = await chat(aiavatar_app, text="東京の天気を教えて。", session_id=session_id)
        assert "晴" in response.text
        assert "23.4" in response.text

    finally:
        await aiavatar_app.stop_listening(session_id)
        task.cancel()
