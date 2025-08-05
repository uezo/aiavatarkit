import pytest
import asyncio
import base64
import io
import os
import wave
from uuid import uuid4
import httpx
import numpy
import pyautogui
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient as AIAvatar, AIAvatarResponse
from aiavatar.animation import AnimationControllerDummy
from aiavatar.face import FaceControllerDummy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_VOICE_SAMPLE_RATE = 24000 # VOICEVOX
WS_SERVER_URL = os.getenv("WS_SERVER_URL")


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
        url=WS_SERVER_URL,
        face_controller=FaceControllerForTest(),
        animation_controller=AnimationControllerTest(),
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

    while app.websocket_connection is None:
        await asyncio.sleep(0.1)

    audio_bytes = await get_input_voice(text)
    for i in range(0, len(audio_bytes), 1024):
        chunk = audio_bytes[i:i + 1024]
        await app.send_microphone_data(
            chunk,
            session_id=session_id
        )

    # Wait for processing responses
    while len(app.last_responses) == 0 or app.last_responses[-1].type != "final":
        await asyncio.sleep(0.1)
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


@pytest.mark.skip("Needs server settings for wakeword")
@pytest.mark.asyncio
async def test_chat_wakeword(aiavatar_app: AIAvatar):
    # Before this test, start server with wakewords=["こんにちは"] and wakeword_timeout=10

    session_id = f"test_chat_wakeword_session_{str(uuid4())}"
    user_id = "test_chat_wakeword_user"
    task = asyncio.create_task(aiavatar_app.start_listening(session_id=session_id, user_id=user_id))

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
                image = pyautogui.screenshot(region=(0, 0, 640, 360))
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
        response = await chat(aiavatar_app, text="東京の天気を教えて。", session_id=session_id)
        assert "晴" in response.text
        assert "23.4" in response.text

    finally:
        await aiavatar_app.stop_listening(session_id)
        task.cancel()


@pytest.mark.asyncio
async def test_websocket_multiple_sessions_isolation():
    """Test that multiple AIAvatar instances connecting to the same server with different session IDs maintain separate session data"""
    
    # Create two separate AIAvatar instances for different sessions
    session_id_1 = f"isolation_test_session_1_{str(uuid4())}"
    session_id_2 = f"isolation_test_session_2_{str(uuid4())}"
    user_id_1 = "isolation_user_1"
    user_id_2 = "isolation_user_2"
    
    aiavatar_1 = AIAvatar(
        url=WS_SERVER_URL,
        face_controller=FaceControllerForTest(),
        animation_controller=AnimationControllerTest(),
        debug=True
    )
    
    aiavatar_2 = AIAvatar(
        url=WS_SERVER_URL,
        face_controller=FaceControllerForTest(),
        animation_controller=AnimationControllerTest(),
        debug=True
    )
    
    # Start both sessions simultaneously
    task_1 = asyncio.create_task(aiavatar_1.start_listening(session_id=session_id_1, user_id=user_id_1))
    task_2 = asyncio.create_task(aiavatar_2.start_listening(session_id=session_id_2, user_id=user_id_2))
    
    try:
        # Wait a bit for connections to establish
        await asyncio.sleep(1)
        
        # Session 1: First conversation
        response_1_1 = await chat(aiavatar_1, text="私の名前は田中です。覚えてください。", session_id=session_id_1)
        assert response_1_1.context_id is not None
        context_id_1 = response_1_1.context_id
        
        # Session 2: First conversation (different topic)
        response_2_1 = await chat(aiavatar_2, text="私の名前は鈴木です。覚えてください。", session_id=session_id_2)
        assert response_2_1.context_id is not None
        context_id_2 = response_2_1.context_id
        
        # Verify different context IDs for different sessions
        assert context_id_1 != context_id_2
        
        # Session 1: Ask about the name (should remember Alice)
        response_1_2 = await chat(aiavatar_1, text="私の名前は何ですか？", session_id=session_id_1, context_id=context_id_1)
        
        # Session 2: Ask about the name (should remember Bob)
        response_2_2 = await chat(aiavatar_2, text="私の名前は何ですか？", session_id=session_id_2, context_id=context_id_2)
        
        # Verify session isolation: each session should only know its own context
        # Session 1 should respond about Alice, not Bob
        response_1_text = response_1_2.text.lower()
        assert "田中" in response_1_text
        assert "鈴木" not in response_1_text
        
        # Session 2 should respond about Bob, not Alice
        response_2_text = response_2_2.text.lower()
        assert "鈴木" in response_2_text
        assert "田中" not in response_2_text
        
        # Verify that session IDs are correctly maintained
        assert response_1_1.session_id == session_id_1
        assert response_1_2.session_id == session_id_1
        assert response_2_1.session_id == session_id_2
        assert response_2_2.session_id == session_id_2
        
        # Verify that user IDs are correctly maintained
        assert response_1_1.user_id == user_id_1
        assert response_1_2.user_id == user_id_1
        assert response_2_1.user_id == user_id_2
        assert response_2_2.user_id == user_id_2
        
        # Verify that context IDs remain consistent within each session
        assert response_1_1.context_id == context_id_1
        assert response_1_2.context_id == context_id_1
        assert response_2_1.context_id == context_id_2
        assert response_2_2.context_id == context_id_2

    finally:
        await aiavatar_1.stop_listening(session_id_1)
        await aiavatar_2.stop_listening(session_id_2)
        task_1.cancel()
        task_2.cancel()


@pytest.mark.skip("Needs server settings for chunked audio response")
@pytest.mark.asyncio
async def test_chunked_audio_response():
    """Test that audio responses are chunked when response_audio_chunk_size is set to a positive value"""
    session_id = f"test_chunked_audio_session_{str(uuid4())}"
    user_id = "test_chunked_audio_user"
    
    # Create a custom AIAvatar client to capture raw responses
    chunked_responses = []
    pcm_format = None
    
    class ChunkedAudioTestClient(AIAvatar):
        async def perform_response(self, response: AIAvatarResponse):
            # Capture all responses to analyze chunking behavior
            chunked_responses.append(response)
            
            # Store PCM format from first response
            nonlocal pcm_format
            if response.metadata and "pcm_format" in response.metadata:
                pcm_format = response.metadata["pcm_format"]
                
            await super().perform_response(response)
    
    test_client = ChunkedAudioTestClient(
        url=WS_SERVER_URL,
        face_controller=FaceControllerForTest(),
        animation_controller=AnimationControllerTest(),
        debug=True
    )
    
    task = asyncio.create_task(test_client.start_listening(session_id=session_id, user_id=user_id))

    try:
        # Send a request that will generate audio response
        response = await chat(test_client, text="こんにちは、元気ですか？", session_id=session_id)
        
        audio_chunks = []
        
        for resp in chunked_responses:
            if resp.metadata and "pcm_format" in resp.metadata and not resp.audio_data:
                # This is the initial response with format info
                initial_response = resp
            elif resp.audio_data and isinstance(resp.audio_data, (str, bytes)):
                # This is an audio chunk
                audio_chunks.append(resp)

        # Verify we got a response with audio
        assert response.text is not None and len(response.text) > 0
        assert len(audio_chunks) > 10

        # Verify PCM format is provided
        assert pcm_format is not None
        assert "sample_rate" in pcm_format
        assert "channels" in pcm_format  
        assert "sample_width" in pcm_format
        
        # Reconstruct PCM audio from chunks
        reconstructed_pcm = b""
        for chunk_resp in audio_chunks:
            if isinstance(chunk_resp.audio_data, str):
                # Decode base64 chunk
                chunk_data = base64.b64decode(chunk_resp.audio_data)
            else:
                chunk_data = chunk_resp.audio_data
            reconstructed_pcm += chunk_data
        
        # Verify reconstructed audio has reasonable length
        assert len(reconstructed_pcm) > 0
        
        # Verify audio parameters are reasonable
        assert pcm_format["sample_rate"] > 0
        assert pcm_format["channels"] in [1, 2]  # Mono or stereo
        assert pcm_format["sample_width"] in [1, 2, 3, 4]  # Common bit depths
        
        # Create a WAV file from the PCM data to verify it's valid
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(pcm_format["channels"])
            wav_file.setsampwidth(pcm_format["sample_width"])
            wav_file.setframerate(pcm_format["sample_rate"])
            wav_file.writeframes(reconstructed_pcm)
        
        wav_data = wav_buffer.getvalue()
        assert len(wav_data) > 44  # WAV header is 44 bytes, so audio data should exist
        
        # Verify the WAV can be parsed back
        wav_buffer.seek(0)
        with wave.open(wav_buffer, 'rb') as wav_file:
            assert wav_file.getnchannels() == pcm_format["channels"]
            assert wav_file.getsampwidth() == pcm_format["sample_width"] 
            assert wav_file.getframerate() == pcm_format["sample_rate"]
            frames = wav_file.readframes(wav_file.getnframes())
            assert len(frames) == len(reconstructed_pcm)
    
    finally:
        await test_client.stop_listening(session_id)
        task.cancel()
