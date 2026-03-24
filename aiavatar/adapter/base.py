import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, List, Optional
from ..sts.models import STSResponse
from ..sts.pipeline import STSPipeline
from .models import AIAvatarRequest, AIAvatarResponse, AvatarControlRequest


class Adapter(ABC):
    def __init__(self, sts: STSPipeline):
        self.sts = sts
        self.sts.handle_response = self.handle_response
        self.sts.stop_response = self.stop_response

        # Control tag pattern: receives (tag, attr) and returns a regex with two capture groups
        self.control_tag_pattern = r'\[{tag}:(\w+)\]|<{tag}\s[^>]*{attr}=["\'](\w+)["\']'

        # Callbacks
        self._on_session_start_handlers: List[Callable[[AIAvatarRequest, Any], Awaitable[None]]] = []
        self._on_request_handlers: List[Callable[[AIAvatarRequest], Awaitable[None]]] = []
        self._on_response_handlers: List[Callable[[AIAvatarResponse, STSResponse], Awaitable[None]]] = []

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict) -> dict:
        allowed_keys = self.get_config().keys()
        updated = {}
        for k, v in config.items():
            if v is None:
                continue
            if k not in allowed_keys:
                continue
            try:
                setattr(self, k, v)
                updated[k] = v
            except Exception:
                pass
        return updated

    @abstractmethod
    async def handle_response(self, response: STSResponse):
        pass

    @abstractmethod
    async def stop_response(self, session_id: str, context_id: str):
        pass

    def parse_control_tag(self, text: str, tag: str, attr: str = "name") -> Optional[str]:
        if not text:
            return None
        pattern = self.control_tag_pattern.format(tag=tag, attr=attr)
        match = re.search(pattern, text)
        if match:
            return match.group(1) or match.group(2)
        return None

    def parse_face_name(self, text: str) -> Optional[str]:
        return self.parse_control_tag(text, "face")

    def parse_animation_name(self, text: str) -> Optional[str]:
        return self.parse_control_tag(text, "animation")

    def parse_vision_source(self, text: str) -> Optional[str]:
        return self.parse_control_tag(text, "vision", "source")

    def parse_avatar_control_request(self, text: str) -> AvatarControlRequest:
        avreq = AvatarControlRequest()
        face_name = self.parse_face_name(text)
        if face_name:
            avreq.face_name = face_name
            avreq.face_duration = 4.0
        animation_name = self.parse_animation_name(text)
        if animation_name:
            avreq.animation_name = animation_name
            avreq.animation_duration = 4.0
        return avreq

    def on_session_start(self, func: Callable[[AIAvatarRequest, Any], Awaitable[None]]):
        self._on_session_start_handlers.append(func)
        return func

    def on_request(self, func: Callable[[AIAvatarRequest], Awaitable[None]]):
        self._on_request_handlers.append(func)
        return func

    def on_response(self, func: Callable[[AIAvatarResponse, STSResponse], Awaitable[None]]):
        self._on_response_handlers.append(func)
        return func
