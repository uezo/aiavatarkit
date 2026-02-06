from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Callable, Optional
import inspect
import httpx
import logging
from .preprocessor import TTSPreprocessor

logger = logging.getLogger(__name__)


class SpeechSynthesizer(ABC):
    def __init__(
        self,
        *,
        style_mapper: Dict[str, str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        preprocessors: List[TTSPreprocessor] = None,
        follow_redirects: bool = False,
        debug: bool = False
    ):
        self.http_client = httpx.AsyncClient(
            follow_redirects=follow_redirects,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )
        self.style_mapper = style_mapper or {}
        self.preprocessors = preprocessors or []
        self.debug = debug

    def parse_style(self, style_info: dict = None) -> str:
        if not style_info:
            return None

        styled_text = style_info.get("styled_text", "")
        for k, v in self.style_mapper.items():
            if k in styled_text:
                return v
        return None

    async def preprocess(self, text: str, style_info: dict = None, language: str = None):
        processed_text = text
        for p in self.preprocessors:
            processed_text = await p.process(processed_text, style_info, language)
        return processed_text

    @abstractmethod
    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        pass

    def get_config(self) -> dict:
        return {
            "style_mapper": self.style_mapper,
            "timeout": getattr(self.http_client.timeout, "read", None) if self.http_client else None,
            "debug": self.debug,
        }

    def set_config(self, config: dict) -> dict:
        allowed_keys = self.get_config().keys()
        updated = {}
        for k, v in config.items():
            if v is None:
                continue
            if k not in allowed_keys:
                continue
            if k == "timeout":
                if self.http_client:
                    self.http_client = httpx.AsyncClient(
                        follow_redirects=self.http_client._follow_redirects,
                        timeout=httpx.Timeout(v),
                        limits=httpx.Limits(
                            max_connections=self.http_client._pool._max_connections,
                            max_keepalive_connections=self.http_client._pool._max_keepalive_connections
                        )
                    )
                    updated[k] = v
            else:
                try:
                    setattr(self, k, v)
                    updated[k] = v
                except Exception:
                    pass
        return updated

    async def close(self):
        await self.http_client.aclose()


class SpeechSynthesizerDummy(SpeechSynthesizer):
    def __init__(
        self,
        *,
        synthesized_bytes: bytes = None,
        wait_sec: float = 0.0,
        style_mapper: Dict[str, str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        preprocessors: List[TTSPreprocessor] = None,
        follow_redirects: bool = False,
        debug: bool = False
    ):
        super().__init__(
            style_mapper=style_mapper,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            preprocessors=preprocessors,
            debug=debug
        )
        self.synthesized_bytes = synthesized_bytes
        self.wait_sec = wait_sec

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        await asyncio.sleep(self.wait_sec)
        return self.synthesized_bytes


def create_instant_synthesizer(
    *,
    method: str = None, url: str = None, params: dict = None, headers: dict = None, json: dict = None,
    request_maker: Callable[[str, Optional[dict], Optional[str]], httpx.Request] = None,
    response_parser: Callable[[httpx.Response], bytes] = None,
    style_mapper: Dict[str, str] = None,
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
    timeout: float = 10,
    preprocessors: List[TTSPreprocessor] = None,
    follow_redirects: bool = False,
    debug = False
) -> SpeechSynthesizer:
    class InstantSynthesizer(SpeechSynthesizer):
        async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
            if not text or not text.strip():
                return bytes()

            logger.info(f"Speech synthesize: {text}")

            # Preprocess
            processed_text = await self.preprocess(text, style_info, language)

            # Make HTTP request
            if request_maker:
                http_request = request_maker(processed_text, style_info, language)
            else:
                # Replace placeholders with processed_text and language
                def replace_placeholders(obj: dict, text: str, lang: str):
                    return {k: v.format(text=text, language=lang or "") if isinstance(v, str) else v for k, v in obj.items()}

                dynamic_params = replace_placeholders(params, processed_text, language) if params else None
                dynamic_headers = replace_placeholders(headers, processed_text, language) if headers else None
                dynamic_json = replace_placeholders(json, processed_text, language) if json else None

                http_request = httpx.Request(
                    method=method,
                    url=url,
                    params=dynamic_params,
                    headers=dynamic_headers,
                    json=dynamic_json
                )

            # Synthesize
            http_response = await self.http_client.send(http_request)
            http_response.raise_for_status()

            # Parse HTTP response and return audio bytes
            if response_parser:
                result = response_parser(http_response)
                if inspect.iscoroutine(result):
                    return await result
                return result
            return http_response.content

    return InstantSynthesizer(
        style_mapper=style_mapper,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        timeout=timeout,
        preprocessors=preprocessors,
        follow_redirects=follow_redirects,
        debug=debug
    )
