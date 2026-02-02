import re
from typing import Dict, List, Union
from fastapi import Depends, FastAPI
from ...sts.pipeline import STSPipeline
from ...adapter.base import Adapter
from ..auth import create_api_key_dependency
from .adapter import AdapterConfigAPI
from .pipeline import PipelineConfigAPI
from .vad import VadConfigAPI
from .stt import SttConfigAPI
from .llm import LlmConfigAPI
from .tts import TtsConfigAPI


def _adapter_key(adapter: Adapter) -> str:
    """Derive a short key from adapter class name.

    AIAvatarWebSocketServer -> websocket
    AIAvatarHttpServer      -> http
    AIAvatarLineBotServer   -> linebot
    AIAvatarLocalServer     -> local
    """
    name = adapter.__class__.__name__
    # Remove AIAvatar prefix and Server suffix
    name = re.sub(r"^AIAvatar", "", name)
    name = re.sub(r"Server$", "", name)
    return name.lower()


def setup_config_api(
    app: FastAPI,
    *,
    adapter: Union[Adapter, List[Adapter], Dict[str, Adapter], None] = None,
    sts: STSPipeline = None,
    api_key: str = None
):
    deps = [Depends(create_api_key_dependency(api_key))] if api_key else []

    if isinstance(adapter, dict):
        _adapters = adapter
    elif isinstance(adapter, list):
        _adapters = {_adapter_key(a): a for a in adapter}
    elif isinstance(adapter, Adapter):
        _adapters = {_adapter_key(adapter): adapter}
    else:
        _adapters = None

    if _adapters:
        adapter_router = AdapterConfigAPI(adapters=_adapters).get_router()
        app.include_router(adapter_router, dependencies=deps)

    _sts = sts or (next(iter(_adapters.values())).sts if _adapters else None)

    if _sts is None:
        raise ValueError("Either 'sts' or 'adapter' must be provided")

    app.include_router(PipelineConfigAPI(pipeline=_sts).get_router(), dependencies=deps)
    app.include_router(VadConfigAPI(vad=_sts.vad).get_router(), dependencies=deps)
    app.include_router(SttConfigAPI(stt=_sts.stt).get_router(), dependencies=deps)
    app.include_router(LlmConfigAPI(llm=_sts.llm).get_router(), dependencies=deps)
    app.include_router(TtsConfigAPI(tts=_sts.tts).get_router(), dependencies=deps)
