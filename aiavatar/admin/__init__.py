import base64
from pathlib import Path
import secrets
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from .character import setup_character_api
from .config import setup_config_api, _adapter_key
from .control import setup_control_api
from .evaluation import setup_evaluation_api
from .logs import setup_logs_api
from .metrics import setup_metrics_api
from ..adapter import Adapter
from ..eval import DialogEvaluator
from ..sts.llm.chatgpt import ChatGPTService

_STATIC_DIR = Path(__file__).parent / "static"


class AdminPanel:
    """Lightweight handle returned by setup_admin_panel for post-setup configuration."""

    def __init__(self, adapters: dict):
        self._adapters = adapters

    def add_adapter(self, adapter: Adapter, *, name: Optional[str] = None):
        """Register an additional adapter for the Config API."""
        key = name or _adapter_key(adapter)
        self._adapters[key] = adapter


def _setup_admin_html(app: FastAPI, *, title: str = "AIAvatarKit Admin Panel", html: str = None):
    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    async def admin_page():
        if html is not None:
            return html
        default_html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return default_html.replace("{{ADMIN_TITLE}}", title)


def _setup_admin_basic_auth(app: FastAPI, *, username: str, password: str, path_prefix: str = "/admin"):
    @app.middleware("http")
    async def admin_basic_auth(request: Request, call_next):
        if request.url.path.startswith(path_prefix):
            auth_header = request.headers.get("Authorization")
            if auth_header is None or not auth_header.startswith("Basic "):
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic realm='Admin'"},
                    content="Unauthorized"
                )

            try:
                encoded = auth_header.split(" ")[1]
                decoded = base64.b64decode(encoded).decode("utf-8")
                req_username, req_password = decoded.split(":", 1)

                if not (secrets.compare_digest(req_username, username) and
                        secrets.compare_digest(req_password, password)):
                    return Response(
                        status_code=401,
                        headers={"WWW-Authenticate": "Basic realm='Admin'"},
                        content="Invalid credentials"
                    )
            except Exception:
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic realm='Admin'"},
                    content="Invalid credentials"
                )

        return await call_next(request)


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
    basic_auth_username: str = None,
    basic_auth_password: str = None,
) -> AdminPanel:
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
        basic_auth_username: Username for Basic authentication (requires basic_auth_password)
        basic_auth_password: Password for Basic authentication (requires basic_auth_username)

    Returns:
        AdminPanel instance for post-setup configuration (e.g. add_adapter)
    """
    # Basic auth middleware (must be registered before routes)
    if basic_auth_username and basic_auth_password:
        _setup_admin_basic_auth(app, username=basic_auth_username, password=basic_auth_password)

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
    adapters = setup_config_api(
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

    return AdminPanel(adapters=adapters)
