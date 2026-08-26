import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.admin import setup_admin_panel
from aiavatar.sts.models import STSResponse
from aiavatar.sts.pipeline import STSPipeline
from aiavatar.util import download_example

from .components import ComponentSet, build_components
from .config import AppConfig


logger = logging.getLogger(__name__)


def create_app(
    *,
    config: AppConfig | None = None,
    components: ComponentSet | None = None,
    download_ui: bool = True,
    use_namo_turn: bool = True,
) -> FastAPI:
    """Assemble the CLI's built-in WebSocket application."""
    config = config or AppConfig.from_env()
    components = components or build_components(
        config,
        use_namo_turn=use_namo_turn,
    )
    vad, stt, llm, tts = components

    pipeline = STSPipeline(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        timestamp_interval_seconds=config.timestamp_interval_seconds,
        timestamp_timezone=config.timestamp_timezone,
        merge_request_threshold=config.merge_request_threshold,
        use_invoke_queue=config.use_invoke_queue,
        debug=config.debug,
    )
    adapter = AIAvatarWebSocketServer(
        sts=pipeline,
        mute_on_barge_in=config.mute_on_barge_in,
        debug=config.debug,
    )

    @vad.on_speech_detecting
    async def forward_partial_request_text(text, session):
        await pipeline.handle_response(STSResponse(
            type="info",
            session_id=session.session_id,
            metadata={"partial_request_text": text},
        ))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            try:
                await pipeline.shutdown()
            except Exception:
                logger.exception("Error shutting down STS pipeline")
            await components.close()

    app = FastAPI(title="AIAvatarKit", lifespan=lifespan)
    app.state.aiavatar_components = components
    app.state.aiavatar_pipeline = pipeline
    app.state.aiavatar_adapter = adapter
    app.include_router(adapter.get_websocket_router())
    setup_admin_panel(app, adapter=adapter)

    ui_mounted = False
    if download_ui:
        try:
            html_dir = download_example("websocket/html")
            # This catch-all mount must remain after /ws and /admin routes.
            app.mount(
                "/",
                StaticFiles(directory=html_dir, html=True),
                name="websocket-example",
            )
            ui_mounted = True
        except Exception as ex:
            logger.warning("Could not prepare the WebSocket example UI: %s", ex)
    if not ui_mounted:
        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse("/admin/")
    return app
