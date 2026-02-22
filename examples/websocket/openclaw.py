# pip install aiavatar uvicorn fastapi websockets
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.admin import setup_admin_panel
from aiavatar.util import download_example

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENCLAW_TOKEN = "YOUR_OPENCLAW_TOKEN"
OPENCLAW_BASE_URL = "http://127.0.0.1:18789/v1"
AIAVATAR_ADMIN_USER = "admin"
AIAVATAR_API_KEY = None     # Set API key if you protect this service
OPENCLAW_REQUEST_PREFIX = "[channel:voice]"
SSL_CERT_PATH = None
SSL_KEY_PATH = None

logger = logging.getLogger("aiavatar")
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
logger.addHandler(streamHandler)


# =============================================================
# STT (Streaming: VAD + STT)
# =============================================================

stt = OpenAISpeechRecognizer(
    openai_api_key=OPENAI_API_KEY,
    language="ja",      # <- Set `en` for English
)

# # We recommend using Azure for dramatically faster performance
# from aiavatar.sts.stt.azure import AzureSpeechRecognizer
# stt = AzureSpeechRecognizer(
#     azure_api_key=AZURE_API_KEY,
#     azure_region=AZURE_REGION,
#     language="ja-JP"    # <- Set `en-US` for English
# )

vad = SileroStreamSpeechDetector(
    speech_recognizer=stt,
    segment_silence_threshold=0.05,
    use_vad_iterator=True,
)


# =============================================================
# LLM (OpenClaw as a backend)
# =============================================================

llm = ChatGPTService(
    openai_api_key=OPENCLAW_TOKEN,
    base_url=OPENCLAW_BASE_URL,
    model="openclaw",
)

# Add the instruction for voice channel
@llm.request_filter
def request_filter(text: str):
    if text is not None:
        return OPENCLAW_REQUEST_PREFIX + text
    return text

# Edit params
@llm.edit_chat_completion_params
def edit_chat_completion_params(chat_completion_params: dict, context_id: str, user_id: str):
    user_message = chat_completion_params["messages"][-1]
    if not any(isinstance(c, str) or c.get("type") == "text" for c in user_message["content"]):
        user_message["content"].append({"type": "text", "text": OPENCLAW_REQUEST_PREFIX})

    # Edit chat_completion_params
    chat_completion_params["messages"] = user_message,
    chat_completion_params["extra_headers"] = {
        "x-openclaw-session-key": user_id
    }


# =============================================================
# TTS
# =============================================================

tts = OpenAISpeechSynthesizer(
    openai_api_key=OPENAI_API_KEY,
    speaker="coral"
)

# # VOICEVOX for Japanese users
# from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
# tts = VoicevoxSpeechSynthesizer(
#     base_url="http://127.0.0.1:50021",
#     speaker=46     # Sayo
# )

# # Aivis Cloud API for Japanese users
# from aiavatar.sts.tts import create_instant_synthesizer
# from aiavatar.sts.tts import AudioConverter
# aivis_tts = create_instant_synthesizer(
#     method="POST",
#     url="https://api.aivis-project.com/v1/tts/synthesize",
#     headers={
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {AIVIS_API_KEY}"
#     },
#     json={
#         "model_uuid": "22e8ed77-94fe-4ef2-871f-a86f94e9a579",   # Kohaku
#         "text": "{text}"
#     },
#     response_parser=AudioConverter(debug=True).convert
# )

# # ElevenLabs
# from aiavatar.sts.tts import create_instant_synthesizer
# elevenlabs_tts = create_instant_synthesizer(
#     method="POST",
#     url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
#     headers={
#         "xi-api-key": ELEVENLABS_API_KEY
#     },
#     json={
#         "text": "{text}",
#         "model_id": "eleven_v3",
#         "output_format": "pcm_16000"
#     }
# )


# =============================================================
# Speech-to-Speech pipeline
# =============================================================

# AIAvatar
aiavatar_app = AIAvatarWebSocketServer(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    merge_request_threshold=3.0,
    use_invoke_queue=True,          # Enabled for vision sequence
    api_key=AIAVATAR_API_KEY,
    debug=True
)


# =============================================================
# WebSocket Server and Admin Panel
# =============================================================

# Download example UI if not exists
download_example("websocket/html")

# Set router to FastAPI app
app = FastAPI()
router = aiavatar_app.get_websocket_router()
app.include_router(router)
app.mount("/static", StaticFiles(directory="html"), name="static")

# Admin panel
setup_admin_panel(
    app,
    adapter=aiavatar_app,
    title="AIAvatarKit Admin Panel",
    api_key=AIAVATAR_API_KEY,
    basic_auth_username=AIAVATAR_ADMIN_USER,
    basic_auth_password=AIAVATAR_API_KEY
)

# Run
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    ssl_certfile=SSL_CERT_PATH,
    ssl_keyfile=SSL_KEY_PATH
)

# open http://localhost:8000/static/vrm.html
