import re
from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Awaitable, Dict, List, Optional
from ..sts.models import STSResponse
from ..sts.pipeline import STSPipeline, ResponseHandler
from .control_tags import ControlTagConfigResolver
from .models import AIAvatarRequest, AIAvatarResponse, AvatarControlRequest, ControlTag

logger = logging.getLogger(__name__)

_CONTROL_TAG_PATTERN = re.compile(
    r'''\[(?P<bracket_name>[A-Za-z_]\w*):(?P<bracket_value>[^\]]+)\]'''
    r'''|<(?P<xml_name>[A-Za-z_]\w*)\b(?P<attributes>(?:"[^"]*"|'[^']*'|[^'"<>])*)>''',
    re.IGNORECASE,
)
_CONTROL_ATTRIBUTE_PATTERN = re.compile(
    r'''([A-Za-z_][\w-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')'''
)


class Adapter(ABC):
    def __init__(self, sts: STSPipeline):
        self.sts = sts
        self.sts.add_response_handler(ResponseHandler(
            can_handle=self.can_handle,
            handle_response=self.handle_response,
            stop_response=self.stop_response
        ))

        # Control tag pattern: receives (tag, attr) and returns a regex with two capture groups
        self.control_tag_pattern = r'\[{tag}:(\w+)\]|<{tag}\s[^>]*{attr}=["\'](\w+)["\']'
        self._control_tag_parsers = {}
        self.register_control_tag("face")
        self.register_control_tag("animation")
        self.register_control_tag("vision", value_attribute="source")
        self._artifact_resolver = ControlTagConfigResolver()
        self.register_control_tag("artifact", parser=self._artifact_resolver)

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

    def can_handle(self, session_id) -> bool:
        logger.warning(f"Adapter {self.__class__.__name__} doesn't handle any responses. Implement `can_handle` and return `True` when the session_id is owned by this adapter.")
        return False

    @abstractmethod
    async def handle_response(self, response: STSResponse):
        pass

    @abstractmethod
    async def stop_response(self, session_id: str, context_id: str):
        pass

    def register_control_tag(
        self,
        name: str,
        *,
        value_attribute: str = "name",
        parser: Optional[Callable[[Dict[str, str]], Optional[Dict[str, Any]]]] = None,
    ):
        """Register a control tag and an optional attribute normalizer."""
        normalized_name = name.lower()
        if not re.fullmatch(r"[A-Za-z_]\w*", normalized_name):
            raise ValueError(f"Invalid control tag name: {name}")
        self._control_tag_parsers[normalized_name] = (value_attribute.lower(), parser)

    def set_artifacts(self, artifacts: Dict[str, Dict[str, Any]]):
        """Replace the application-owned artifact catalog."""
        self._artifact_resolver.set_configs(artifacts)
        self.register_control_tag("artifact", parser=self._artifact_resolver)

    def update_artifacts(self, artifacts: Dict[str, Dict[str, Any]]):
        """Add or replace artifact catalog entries without clearing other IDs."""
        self._artifact_resolver.update_configs(artifacts)
        self.register_control_tag("artifact", parser=self._artifact_resolver)

    def add_artifact(self, artifact_id: str, config: Dict[str, Any]):
        """Add or replace one artifact catalog entry."""
        self._artifact_resolver.set_config(artifact_id, config)
        self.register_control_tag("artifact", parser=self._artifact_resolver)

    def parse_control_tags(self, text: str) -> List[ControlTag]:
        """Parse registered bracket or XML-style control tags in source order."""
        tags = []
        parsers = getattr(self, "_control_tag_parsers", {})
        for match in _CONTROL_TAG_PATTERN.finditer(text or ""):
            name = (match.group("bracket_name") or match.group("xml_name")).lower()
            registration = parsers.get(name)
            if not registration:
                continue

            value_attribute, parser = registration
            if match.group("bracket_name"):
                attributes = {value_attribute: match.group("bracket_value").strip()}
            else:
                attributes = self._parse_control_attributes(match.group("attributes"))
                if attributes is None:
                    logger.warning("Ignored malformed control tag: %s", match.group(0))
                    continue

            if parser:
                try:
                    attributes = parser(attributes)
                except (TypeError, ValueError) as ex:
                    logger.warning("Ignored invalid %s control tag: %s", name, ex)
                    continue
                if attributes is None:
                    continue
            tags.append(ControlTag(name=name, attributes=attributes))
        return tags

    @staticmethod
    def _parse_control_attributes(source: str) -> Optional[Dict[str, str]]:
        source = source.strip()
        if source.endswith("/"):
            source = source[:-1].rstrip()

        attributes = {}
        position = 0
        for match in _CONTROL_ATTRIBUTE_PATTERN.finditer(source):
            if source[position:match.start()].strip():
                return None
            name = match.group(1).lower()
            if name in attributes:
                return None
            attributes[name] = match.group(2) if match.group(2) is not None else match.group(3)
            position = match.end()
        if source[position:].strip():
            return None
        return attributes

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
