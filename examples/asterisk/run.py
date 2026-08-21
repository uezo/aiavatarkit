import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aiavatar.adapter.asterisk import (
    AIAvatarAsteriskServer,
    AsteriskARIClient,
    AsteriskCallManager,
)
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer


pipeline = STSPipeline(
    stt=OpenAISpeechRecognizer(
        openai_api_key=os.environ["OPENAI_API_KEY"],
    ),
    llm_openai_api_key=os.environ["OPENAI_API_KEY"],
)

asterisk = AIAvatarAsteriskServer(
    sts=pipeline,
    tts_sample_rate=int(os.getenv("AIAVATAR_TTS_SAMPLE_RATE", "24000")),
    mute_on_barge_in=True,
    channel="phone",
    api_username=os.environ["AIAVATAR_MEDIA_USERNAME"],
    api_password=os.environ["AIAVATAR_MEDIA_PASSWORD"],
)

ari = AsteriskARIClient(
    base_url=os.environ["ASTERISK_ARI_BASE_URL"],
    username=os.environ["ASTERISK_ARI_USERNAME"],
    password=os.environ["ASTERISK_ARI_PASSWORD"],
)

call_manager = AsteriskCallManager(
    adapter=asterisk,
    ari_client=ari,
    bridge_endpoint=os.getenv("ASTERISK_BRIDGE_ENDPOINT", "avaya-trunk"),
    external_media_host=os.getenv(
        "ASTERISK_MEDIA_CONNECTION",
        "aiavatarkit-media",
    ),
    transfer_destinations={
        "operator": os.getenv("AVAYA_OPERATOR_EXTENSION", "1234"),
        "sales": os.getenv("AVAYA_SALES_EXTENSION", "2345"),
    },
    transfer_strategy=os.getenv(
        "ASTERISK_TRANSFER_STRATEGY",
        "refer_then_bridge",
    ),
    refer_timeout=float(os.getenv("ASTERISK_REFER_TIMEOUT", "30")),
    media_start_timeout=float(os.getenv("ASTERISK_MEDIA_START_TIMEOUT", "10")),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await call_manager.start()
    try:
        yield
    finally:
        await call_manager.close()
        await pipeline.shutdown()


app = FastAPI(lifespan=lifespan)
app.include_router(asterisk.get_router(path="/asterisk/media"))


@asterisk.on_transfer_prepare
async def on_transfer_prepare(request, session):
    # These are Asterisk channel variables. Map them to a SIP header, UUI, or
    # Refer-To parameter in trusted Asterisk/Avaya configuration.
    request.variables["AIAVATAR_USER_ID"] = request.user_id
    if request.context_id:
        request.variables["AIAVATAR_CONTEXT_ID"] = request.context_id


@asterisk.on_dtmf
async def on_dtmf(digit: str, session_id: str):
    print(f"DTMF received: session={session_id} digit={digit}")


@asterisk.on_transfer_completed
async def on_transfer_completed(
    session_id: str,
    destination: str,
    method: str,
):
    print(
        f"Transfer completed: session={session_id} "
        f"destination={destination} method={method}"
    )


@asterisk.on_transfer_failed
async def on_transfer_failed(
    session_id: str,
    destination: str,
    reason: str,
):
    print(
        f"Transfer failed: session={session_id} "
        f"destination={destination} reason={reason}"
    )


@asterisk.on_transfer_unknown
async def on_transfer_unknown(
    session_id: str,
    destination: str,
    reason: str,
):
    print(
        f"Transfer outcome unknown: session={session_id} "
        f"destination={destination} reason={reason}"
    )
