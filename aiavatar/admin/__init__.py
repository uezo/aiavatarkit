from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from .auth import create_api_key_dependency
from .character import setup_character_api
from .config import setup_config_api
from .control import setup_control_api
from .evaluation import setup_evaluation_api
from .logs import setup_logs_api
from .metrics import setup_metrics_api
from ..adapter import Adapter
from ..eval import DialogEvaluator
from ..sts.llm.chatgpt import ChatGPTService

_STATIC_DIR = Path(__file__).parent / "static"


def _setup_admin_html(app: FastAPI, *, title: str = "AIAvatarKit Admin Panel", html: str = None):
    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    async def admin_page():
        if html is not None:
            return html
        default_html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return default_html.replace("{{ADMIN_TITLE}}", title)


def setup_admin_panel(
    app: FastAPI,
    *,
    adapter: Adapter,
    title: str = "AIAvatarKit Admin Panel",
    html: str = None,
    evaluator=None,
    character_service=None,
    character_id: str = None,
    default_session_id: str = None,
    api_key: str = None,
):
    """Convenience function to set up the full admin panel.

    Required:
        adapter: Adapter instance (provides sts pipeline, performance_recorder, voice_recorder)

    Optional:
        title: Admin panel title
        html: Custom HTML string for the admin page (overrides built-in template)
        evaluator: DialogEvaluator for evaluation tab
        character_service: CharacterService for character tab
        character_id: Character ID for character tab (required if character_service is set)
        default_session_id: Default session ID for control API
        api_key: API key for all admin endpoints
    """
    # Admin HTML page
    _setup_admin_html(app, title=title, html=html)

    # Metrics
    setup_metrics_api(
        app,
        recorder=adapter.sts.performance_recorder,
        api_key=api_key
    )

    # Logs
    setup_logs_api(
        app,
        recorder=adapter.sts.performance_recorder,
        voice_recorder=adapter.sts.voice_recorder if adapter.sts.voice_recorder_enabled else None,
        api_key=api_key,
    )

    # Control (perform / conversation)
    setup_control_api(
        app,
        adapter=adapter,
        default_session_id=default_session_id,
        api_key=api_key
    )

    # Config
    setup_config_api(
        app,
        adapter=adapter,
        api_key=api_key
    )

    # Evaluation
    if not evaluator:
        if isinstance(adapter.sts.llm, ChatGPTService):
            eval_llm = ChatGPTService(
                openai_api_key=adapter.sts.llm.openai_client.api_key,
                base_url=adapter.sts.llm.openai_client.base_url,
                model=adapter.sts.llm.model,
                temperature=adapter.sts.llm.temperature,
                reasoning_effort=adapter.sts.llm.reasoning_effort
            )
            evaluator = DialogEvaluator(llm=adapter.sts.llm, evaluation_llm=eval_llm)
    if evaluator:
        setup_evaluation_api(app, evaluator=evaluator, api_key=api_key)

    # Character (optional)
    if character_service and character_id:
        setup_character_api(
            app,
            character_service=character_service,
            character_id=character_id,
            api_key=api_key,
        )
