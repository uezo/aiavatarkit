from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, List
from ..sts.models import STSResponse
from ..sts.pipeline import STSPipeline
from .models import AIAvatarRequest, AIAvatarResponse


class Adapter(ABC):
    def __init__(self, sts: STSPipeline):
        self.sts = sts
        self.sts.handle_response = self.handle_response
        self.sts.stop_response = self.stop_response

        # Callbacks
        self._on_session_start_handlers: List[Callable[[AIAvatarRequest, Any], Awaitable[None]]] = []
        self._on_request_handlers: List[Callable[[AIAvatarRequest], Awaitable[None]]] = []
        self._on_response_handlers: List[Callable[[AIAvatarResponse, STSResponse], Awaitable[None]]] = []

    @abstractmethod
    async def handle_response(self, response: STSResponse):
        pass

    @abstractmethod
    async def stop_response(self, session_id: str, context_id: str):
        pass

    def on_session_start(self, func: Callable[[AIAvatarRequest, Any], Awaitable[None]]):
        self._on_session_start_handlers.append(func)
        return func

    def on_request(self, func: Callable[[AIAvatarRequest], Awaitable[None]]):
        self._on_request_handlers.append(func)
        return func

    def on_response(self, func: Callable[[AIAvatarResponse, STSResponse], Awaitable[None]]):
        self._on_response_handlers.append(func)
        return func
