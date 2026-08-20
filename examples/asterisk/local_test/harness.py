import math
import struct
from collections import defaultdict
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from aiavatar.adapter.asterisk import (
    AIAvatarAsteriskServer,
    AsteriskARIClient,
    AsteriskCallManager,
)
from aiavatar.sts.models import STSResponse


class LocalTestVAD:
    def __init__(self):
        self._on_recording_started = []
        self._session_data = {}
        self.bytes_received = defaultdict(int)
        self.frames_received = defaultdict(int)

    def on_recording_started(self, func):
        self._on_recording_started.append(func)
        return func

    def set_session_data(
        self,
        session_id,
        key,
        value,
        create_session=False,
    ):
        if create_session:
            self._session_data.setdefault(session_id, {})
        if session_id in self._session_data:
            self._session_data[session_id][key] = value

    def get_session_data(self, session_id, key):
        return self._session_data.get(session_id, {}).get(key)

    async def process_samples(self, samples, session_id):
        self.bytes_received[session_id] += len(samples)
        self.frames_received[session_id] += 1
        return False

    async def finalize_session(self, session_id):
        self._session_data.pop(session_id, None)


class LocalTestTTS:
    async def synthesize(self, text):
        return make_tone()


class LocalTestPipeline:
    def __init__(self):
        self.vad = LocalTestVAD()
        self.tts = LocalTestTTS()
        self.response_handlers = []
        self.finalized = []

    def add_response_handler(self, handler):
        self.response_handlers.append(handler)

    async def finalize(self, session_id):
        if session_id not in self.finalized:
            self.finalized.append(session_id)
        await self.vad.finalize_session(session_id)


def make_tone(
    frequency=440.0,
    duration=1.0,
    sample_rate=16_000,
    amplitude=8_000,
):
    samples = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(int(sample_rate * duration))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


pipeline = LocalTestPipeline()
adapter = AIAvatarAsteriskServer(
    sts=pipeline,
    tts_sample_rate=16_000,
    mute_on_barge_in=True,
    api_username="aiavatarkit-local",
    api_password="local-only-change-me",
    debug=True,
)
ari = AsteriskARIClient(
    base_url="http://127.0.0.1:18088/ari",
    username="aiavatar-local",
    password="local-only-change-me",
    reconnect_delay=0.25,
)
call_manager = AsteriskCallManager(
    adapter=adapter,
    ari_client=ari,
    bridge_endpoint="unused-local-test",
    transfer_destinations={},
    external_media_host="aiavatarkit-media",
    transfer_strategy="bridge",
)
dtmf_events = []


@adapter.on_dtmf
async def record_dtmf(digit, session_id):
    dtmf_events.append({"session_id": session_id, "digit": digit})


@asynccontextmanager
async def lifespan(app):
    await call_manager.start()
    try:
        yield
    finally:
        await call_manager.close()


app = FastAPI(lifespan=lifespan)
app.include_router(adapter.get_router(path="/asterisk/media"))


@app.get("/test/state")
async def state():
    return {
        "calls": {
            session_id: {
                "caller_channel_id": session.ari_caller_channel_id,
                "media_channel_id": session.media_channel_id,
                "pipeline_session_id": session.pipeline_session_id,
                "bridge_id": session.bridge_id,
                "media_connected": session.media_connected,
                "last_mark": session.last_mark,
                "bytes_received": pipeline.vad.bytes_received[
                    session.pipeline_session_id
                ],
                "frames_received": pipeline.vad.frames_received[
                    session.pipeline_session_id
                ],
            }
            for session_id, session in call_manager.sessions.items()
        },
        "adapter_sessions": sorted(adapter.sessions),
        "finalized": list(pipeline.finalized),
        "dtmf": list(dtmf_events),
    }


@app.post("/test/tone/{session_id}")
async def send_tone(session_id: str):
    session = adapter.sessions.get(session_id)
    if not session or not adapter.can_handle(session.pipeline_session_id):
        raise HTTPException(status_code=404, detail="Media session is not connected")
    pipeline_session_id = session.pipeline_session_id
    transaction_id = f"local-test-{uuid4()}"
    await adapter.handle_response(STSResponse(
        type="accepted",
        session_id=pipeline_session_id,
        transaction_id=transaction_id,
    ))
    await adapter.handle_response(STSResponse(
        type="start",
        session_id=pipeline_session_id,
        transaction_id=transaction_id,
    ))
    await adapter.handle_response(STSResponse(
        type="chunk",
        session_id=pipeline_session_id,
        transaction_id=transaction_id,
        audio_data=make_tone(),
    ))
    await adapter.handle_response(STSResponse(
        type="final",
        session_id=pipeline_session_id,
        transaction_id=transaction_id,
        text="local round-trip test",
    ))
    return {"session_id": session_id, "sent_pcm_bytes": 32_000}


@app.post("/test/hangup/{session_id}")
async def hangup(session_id: str):
    if session_id not in call_manager.sessions:
        raise HTTPException(status_code=404, detail="Call is not active")
    await call_manager.hangup(session_id)
    return {"session_id": session_id, "hung_up": True}
