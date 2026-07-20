from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ..adapter import Adapter
from ..eval import DialogEvaluator
from ..sts.llm.chatgpt import ChatGPTService
from .auth import AdminAuthenticator, BasicAdminAuthenticator, create_auth_dependency
from .config import _adapter_key, create_config_router
from .evaluation import EvaluationAPI
from .logs import LogsAPI
from .metrics import MetricsAPI

_STATIC_DIR = Path(__file__).parent / "static"


class AdminPanel:
    """Handle for adding adapters to the Config view after setup."""

    def __init__(self, adapters: dict):
        self._adapters = adapters

    def add_adapter(self, adapter: Adapter, *, name: Optional[str] = None):
        self._adapters[name or _adapter_key(adapter)] = adapter


def _default_evaluator(adapter: Adapter) -> Optional[DialogEvaluator]:
    if not isinstance(adapter.sts.llm, ChatGPTService):
        return None
    source = adapter.sts.llm
    evaluation_llm = ChatGPTService(
        openai_api_key=source.openai_client.api_key,
        base_url=str(source.openai_client.base_url),
        model=source.model,
        temperature=source.temperature,
        reasoning_effort=source.reasoning_effort,
    )
    return DialogEvaluator(llm=source, evaluation_llm=evaluation_llm)


def setup_admin_panel(
    app: FastAPI,
    *,
    adapter: Adapter,
    title: str = "AIAvatarKit Admin Panel",
    authenticator: AdminAuthenticator = None,
    evaluator: DialogEvaluator = None,
) -> AdminPanel:
    """Install the new Admin UI and API under ``/admin``.

    The same authenticator protects HTML, assets, and API calls. Pass
    ``BasicAdminAuthenticator`` for Basic auth or any async/sync callable taking
    a FastAPI ``Request`` for an SSO integration.
    """
    evaluator = evaluator or _default_evaluator(adapter)
    auth = create_auth_dependency(authenticator)
    router = APIRouter(prefix="/admin", dependencies=[Depends(auth)])

    def render_page():
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("{{ADMIN_TITLE}}", title))

    def redirect_to_page():
        return RedirectResponse("admin/")

    router.add_api_route("", redirect_to_page, methods=["GET"], include_in_schema=False)
    router.add_api_route("/", render_page, methods=["GET"], include_in_schema=False)

    @router.get("/assets/{asset_path:path}", include_in_schema=False)
    async def asset(asset_path: str):
        static_root = _STATIC_DIR.resolve()
        target = (static_root / asset_path).resolve()
        if static_root not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(target)

    @router.get("/api/capabilities", tags=["Admin"])
    async def capabilities():
        return {"evaluation": evaluator is not None}

    router.include_router(MetricsAPI(adapter.sts.performance_recorder).get_router(), prefix="/api")
    router.include_router(
        LogsAPI(
            adapter.sts.performance_recorder,
            adapter.sts.voice_recorder if adapter.sts.voice_recorder_enabled else None,
        ).get_router(),
        prefix="/api",
    )
    config_router, adapters = create_config_router(adapter=adapter)
    router.include_router(config_router, prefix="/api")
    if evaluator is not None:
        router.include_router(EvaluationAPI(evaluator).get_router(), prefix="/api")

    app.include_router(router)
    return AdminPanel(adapters)


__all__ = [
    "AdminAuthenticator",
    "AdminPanel",
    "BasicAdminAuthenticator",
    "setup_admin_panel",
]
