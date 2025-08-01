from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Optional
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
        debug: bool = False
    ):
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
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

    async def close(self):
        await self.http_client.aclose()


class SpeechSynthesizerDummy(SpeechSynthesizer):
    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        return None


def create_instant_synthesizer(
    *,
    method: str, url: str, params: dict = None, headers: dict = None, json: dict = None,
    request_maker: Callable[[str, Optional[dict], Optional[str]], httpx.Request] = None,
    response_parser: Callable[[httpx.Response], bytes] = None,
    style_mapper = None,
    max_connections = 100,
    max_keepalive_connections = 20,
    timeout = 10,
    preprocessors = None,
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
            return response_parser(http_response) if response_parser else http_response.content

    return InstantSynthesizer(
        style_mapper=style_mapper,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        timeout=timeout,
        preprocessors=preprocessors,
        debug=debug
    )
