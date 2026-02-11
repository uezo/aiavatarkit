import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.stt.azure import AzureSpeechRecognizer
from aiavatar.adapter.stt import StreamSpeechRecognitionServer

# Configuration
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_REGION = os.environ.get("AZURE_REGION")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup
    await speech_recognizer.close()


# Initialize SpeechRecognizer component
speech_recognizer = AzureSpeechRecognizer(
    azure_api_key=AZURE_API_KEY,
    azure_region=AZURE_REGION,
    language="ja-JP",
    alternative_languages=["en-US", "zh-CN"]
)

# Stream VAD with real-time recognition
vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    silence_duration_threshold=0.5,
    segment_silence_threshold=0.05,  # Send partial results after 0.05s silence
    max_duration=30.0,
    preroll_buffer_count=10,
    debug=True
)

# Create STT WebSocket server
stt_server = StreamSpeechRecognitionServer(vad=vad, debug=True)

# Optional: Add callbacks
@stt_server.on_connect
async def on_connect(request, session_data):
    print(f"Client connected: {request.session_id}")

@stt_server.on_disconnect
async def on_disconnect(session_data):
    print(f"Client disconnected: {session_data.id}")


# FastAPI app
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="html"), name="static")
app.include_router(stt_server.get_websocket_router("/ws/stt"))

# Run `uvicorn server:app` and open http://localhost:8000/static/index.html
