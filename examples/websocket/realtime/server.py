"""Run the shared avatar UI with a Realtime or GPT-Live pipeline.

From the repository root:
    python -m examples.websocket.realtime.server --pipeline realtime
    python -m examples.websocket.realtime.server --pipeline live
"""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from aiavatar.sts.tts import SpeechSynthesizer
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from .adapter import RealtimeWebSocketServer
from .pipeline.live import OpenAILivePipeline
from .pipeline.realtime import OpenAIRealtimePipeline


def create_app(
    pipeline: str = "realtime",
    *,
    openai_api_key: str | None = None,
    model: str | None = None,
    instructions: str | None = None,
    voice: str = "marin",
    tts: SpeechSynthesizer | None = None,
    voicevox_url: str = "http://127.0.0.1:50021",
    voicevox_speaker: int = 46,
    api_key: str | None = None,
) -> FastAPI:
    """Create the comparison app; an injected TTS remains caller-owned.

    OPENAI_API_KEY is read by the selected pipeline when no key is supplied.
    Realtime uses external TTS, while Live returns native continuous PCM.
    """
    if pipeline not in ("realtime", "live"):
        raise ValueError("pipeline must be 'realtime' or 'live'")
    if pipeline == "live" and tts is not None:
        raise ValueError("Live uses native audio and does not accept tts")

    owned_tts = None
    options = {"openai_api_key": openai_api_key, "instructions": instructions,
               "input_sample_rate": 16000}
    if model is not None:
        options["model"] = model
    if pipeline == "realtime":
        if tts is None:
            owned_tts = tts = VoicevoxSpeechSynthesizer(
                base_url=voicevox_url, speaker=voicevox_speaker,
            )
        sts = OpenAIRealtimePipeline(
            tts=tts, context_db_path=None, restore_context=False, **options,
        )
    else:
        sts = OpenAILivePipeline(voice=voice, **options)

    adapter = RealtimeWebSocketServer(sts=sts, api_key=api_key)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            try:
                await sts.shutdown()
            finally:
                if owned_tts is not None:
                    await owned_tts.close()

    app = FastAPI(lifespan=lifespan)
    app.state.adapter = adapter
    app.include_router(adapter.get_websocket_router("/ws"))
    # Use the maintained pages and avatar assets, without copying or rewriting HTML.
    html_dir = Path(__file__).resolve().parents[1] / "html"
    app.mount("/html", StaticFiles(directory=html_dir), name="html")

    @app.get("/")
    async def index():
        return RedirectResponse("html/index.html")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("realtime", "live"), default="realtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--instructions", default=None)
    parser.add_argument("--voice", default="marin", help="GPT-Live voice")
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--voicevox-speaker", type=int, default=46)
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        args.pipeline, model=args.model, instructions=args.instructions,
        voice=args.voice, voicevox_url=args.voicevox_url,
        voicevox_speaker=args.voicevox_speaker,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
