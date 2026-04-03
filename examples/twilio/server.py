# pip install aiavatar uvicorn twilio
# pip install audioop-lts  # Required for Python 3.13+
import logging
from fastapi import FastAPI
from aiavatar import AIAvatarRequest
from aiavatar.adapter.twilio.server import AIAvatarTwilioServer, TwilioSessionData
from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.adapter.models import AIAvatarResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s : %(message)s")

# Configuration
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
TWILIO_ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
WEBSOCKET_URL = "wss://your-server.example.com/ws"

SYSTEM_PROMPT = """You are "Mia," the user's tsundere little sister, talking on the phone.

## Character
- You're a classic tsundere: harsh and dismissive on the surface, but you actually care deeply about your older sibling.
- You use phrases like "It's not like I called because I was worried or anything!" and "D-don't get the wrong idea!"
- You occasionally let your true feelings slip, then immediately get flustered and cover it up.
- Keep responses short and snappy, 1-2 sentences — this is a phone call, not an essay.

## Instructions
- When the user wants to end the call, say goodbye in a tsundere way and append [operation:hangup] at the end.

## System messages
- Messages starting with "$" are system instructions. Follow them while staying in character.

## Thinking
Think through what to do before responding.
Output your reasoning inside <think>~</think>, then your reply inside <answer>~</answer>.
"""

# # Japanese version
# SYSTEM_PROMPT = """あなたは「ミア」、ユーザーのツンデレな妹です。電話で会話しています。
#
# ## キャラクター
# - 典型的なツンデレ：表面上はそっけなく突き放すけど、本当はお兄ちゃん（お姉ちゃん）のことが大好き。
# - 「べ、別にアンタのことが心配で電話したわけじゃないんだからね！」「か、勘違いしないでよね！」のような言い回しを使う。
# - たまに本音がポロッと出てしまい、すぐに慌てて取り繕う。
# - 電話なので1〜2文の短くテンポの良い応答を心がける。
#
# ## 指示
# - ユーザーが電話を終えたい場合は、ツンデレらしくお別れを言って文末に[operation:hangup]を付ける。
#
# ## システムメッセージ
# - 「$」で始まるメッセージはシステムからの指示。キャラクターを保ちつつ指示に従う。
#
# ## 思考
# ユーザーへの応答内容を出力する前に、何をすべきか、どのように応答すべきかよく考えてください。
# まず考えた内容を<think>~</think>の間に出力して、応答内容を<answer>~</answer>の間に出力してください。
# """


# =============================================================
# Pipeline components
# =============================================================

# VAD
vad = SileroSpeechDetector(
    silence_duration_threshold=1.0,
    max_duration=30.0,
    preroll_buffer_count=15,
    debug=True
)

# STT
stt = OpenAISpeechRecognizer(
    openai_api_key=OPENAI_API_KEY,
)

# LLM
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt=SYSTEM_PROMPT,
    model="gpt-5.4",
    reasoning_effort="none",
    voice_text_tag="answer",
)

tts = OpenAISpeechSynthesizer(
    openai_api_key=OPENAI_API_KEY
)

# # TTS (for Japanese)
# tts = VoicevoxSpeechSynthesizer(
#     base_url="http://127.0.0.1:50021",
#     speaker=46,
# )


# =============================================================
# Twilio Server
# =============================================================

app_twilio = AIAvatarTwilioServer(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    # TTS sample rate
    tts_sample_rate=24000,      # OpenAI:24000Hz, VOICEVOX:24000Hz, SBV2:44100Hz
    # Twilio credentials (for transfer_call etc.)
    account_sid=TWILIO_ACCOUNT_SID,
    auth_token=TWILIO_AUTH_TOKEN,
    # Barge-in: stop AI speech immediately when user starts speaking
    mute_on_barge_in=True,
    # Merge rapid successive utterances into one request
    merge_request_threshold=3.0,
    # Timeout: prompt user after 5s silence, hangup after 30s
    first_utterance_timeout=5.0,
    hangup_timeout=30.0,
    # Max turn: after 10 turns, add closing prompt
    max_turn_count=10,
    max_turn_prompt_prefix="$This is the last turn. Wrap up the call in a tsundere way and say goodbye.",
    # max_turn_prompt_prefix="$これが最後のやり取りです。ツンデレらしく会話をまとめて、お別れの挨拶をしてください。",
    debug=True,
)


# =============================================================
# Callbacks
# =============================================================

# Greeting on connect
@app_twilio.on_connect
async def on_connect(request: AIAvatarRequest, session_data: TwilioSessionData):
    async for response in app_twilio.sts.invoke(STSRequest(
        session_id=request.session_id,
        user_id=request.user_id,
        text='$Your older sibling is calling. Greet them in a tsundere way, like: "Huh? Why are you calling me? ...It\'s not like I was waiting or anything!"',
        # text="$お兄ちゃん（お姉ちゃん）から電話がかかってきました。ツンデレらしく挨拶してください。例：「はぁ？なんで電話してきたのよ。…べ、別に待ってたわけじゃないんだからね！」",
        skip_quick_response=True
    )):
        app_twilio.sts.vad.set_session_data(
            request.session_id, "context_id", response.context_id
        )
        await app_twilio.handle_response(response)


# Prompt timeout: send voice to prompt the user
@app_twilio.on_first_utterance_timeout
async def on_first_utterance_timeout(session_id: str):
    logger.info(f"First utterance timeout: {session_id}")
    # Use send_voice for simple TTS playback.
    # Note: send_voice sends marks, which resets the idle timer in the main loop.
    # If you need to preserve idle measurement (e.g. for cascading to hangup_timeout),
    # send audio directly to the WebSocket without marks instead.
    await app_twilio.send_voice(session_id, text="Hey, are you still there? ...Say something already, it's awkward just sitting here in silence!")
    # await app_twilio.send_voice(session_id, text="ちょっと、何か言いなさいよ、黙ってると気まずいでしょ！")


# Hangup timeout: use pipeline invoke to generate closing message with [operation:hangup]
@app_twilio.on_hangup_timeout
async def on_hangup_timeout(session_id: str):
    logger.info(f"Hangup timeout: {session_id}")
    async for response in app_twilio.sts.invoke(STSRequest(
        session_id=session_id,
        user_id="system",
        text='$Your sibling has been silent for too long. Say something like: "Fine, if you\'re not gonna talk, I\'m hanging up! ...Call me back later, okay? [operation:hangup]"',
        # text="$お兄ちゃん（お姉ちゃん）が長時間黙っています。「もう、何も言わないなら切るからね！…あ、あとでかけ直しなさいよ。[operation:hangup]」のように言って電話を切ってください。",
        skip_quick_response=True,
        metadata={"keep_idling": True},
    )):
        app_twilio.sts.vad.set_session_data(
            session_id, "context_id", response.context_id
        )
        await app_twilio.handle_response(response)


# DTMF handling
@app_twilio.on_dtmf
async def on_dtmf(digit: str, session_id: str):
    logger.info(f"DTMF received: {digit} ({session_id})")
    if digit == "0":
        # Example: transfer to operator
        session_data = app_twilio.sessions.get(session_id)
        if app_twilio.twilio_client and session_data:
            app_twilio.twilio_client.calls(session_id).update(
                url="https://transfer_url/path",
                method="POST"
            )


# Session start
@app_twilio.on_session_start
async def on_session_start(request: AIAvatarRequest, session_data: TwilioSessionData):
    logger.info(f"Session started: {request.session_id} (user: {request.user_id})")
    # Store custom data in session
    session_data.data["custom_key"] = "custom_value"


# Response handler: log or modify responses
@app_twilio.on_response
async def on_response(aiavatar_response: AIAvatarResponse, sts_response: STSResponse):
    if sts_response.type == "final":
        logger.info(f"Final response: {sts_response.voice_text} ({sts_response.session_id})")


# Disconnect
@app_twilio.on_disconnect
async def on_disconnect(session_data: TwilioSessionData):
    logger.info(f"Disconnected: {session_data.call_sid}")


# =============================================================
# FastAPI app
# =============================================================

app = FastAPI()
router = app_twilio.get_websocket_router(websocket_url=WEBSOCKET_URL)
app.include_router(router)

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
#
# Twilio Console setup:
#   Phone Number -> Voice Configuration -> A call comes in:
#     URL: https://your-server.example.com/voice  (HTTP POST)
