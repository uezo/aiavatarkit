from abc import ABC, abstractmethod
import asyncio
import hashlib
import json as json_mod
import os
from typing import Dict, List, Callable, Optional, final
import inspect
import aiofiles
import httpx
import logging
from .preprocessor import TTSPreprocessor
from .postprocessor import TTSPostprocessor, WavSampleRatePostprocessor

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
        postprocessors: List[TTSPostprocessor] = None,
        sample_rate: int = None,
        follow_redirects: bool = False,
        cache_dir: str = None,
        cache_ext: str = "wav",
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
        self.postprocessors = [
            *(postprocessors or []),
            WavSampleRatePostprocessor(),
        ]
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self.cache_ext = cache_ext
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
            kwargs = {
                "style_info": style_info,
                "language": language,
            }
            try:
                parameters = inspect.signature(p.process).parameters.values()
                if any(
                    parameter.name == "synthesizer"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                ):
                    kwargs["synthesizer"] = self
            except (TypeError, ValueError):
                pass
            processed_text = await p.process(processed_text, **kwargs)
        return processed_text

    async def postprocess(self, audio: bytes) -> bytes:
        if not audio:
            return audio
        processed_audio = audio
        for p in self.postprocessors:
            processed_audio = await p.process(
                processed_audio,
                synthesizer=self,
            )
        return processed_audio

    def make_cache_key(
        self,
        url: str,
        headers: dict = None,
        params: dict = None,
        json_body: dict = None,
        data: bytes = None,
    ) -> Optional[str]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256()
        h.update(url.encode())
        if headers:
            h.update(json_mod.dumps(headers, sort_keys=True).encode())
        if params:
            h.update(json_mod.dumps(params, sort_keys=True).encode())
        if json_body:
            h.update(json_mod.dumps(json_body, sort_keys=True).encode())
        if data:
            h.update(data)
        postprocess_config = [
            config
            for processor in self.postprocessors
            if (config := processor.get_cache_config(self)) is not None
        ]
        if postprocess_config:
            h.update(json_mod.dumps(postprocess_config, sort_keys=True).encode())
        return h.hexdigest()

    async def read_cache(self, cache_key: str) -> Optional[bytes]:
        if not cache_key:
            return None
        path = os.path.join(self.cache_dir, f"{cache_key}.{self.cache_ext}")
        if not os.path.exists(path):
            return None
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def write_cache(self, cache_key: str, data: bytes):
        if not cache_key or not data:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        async with aiofiles.open(os.path.join(self.cache_dir, f"{cache_key}.{self.cache_ext}"), "wb") as f:
            await f.write(data)

    @final
    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        """Run the common preprocessing, caching, generation, and postprocessing flow."""
        if not text or not text.strip():
            return bytes()

        if self.sample_rate is not None and (
            not isinstance(self.sample_rate, int)
            or isinstance(self.sample_rate, bool)
            or self.sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer")

        if self.debug:
            logger.info(f"Speech synthesize: {text}")

        processed_text = await self.preprocess(text, style_info, language)
        cache_key = None
        if self.cache_dir:
            cache_key = await self.make_synthesis_cache_key(
                processed_text,
                style_info,
                language,
            )
            if cached := await self.read_cache(cache_key):
                return cached

        audio = await self.generate(processed_text, style_info, language)
        audio = await self.postprocess(audio)
        await self.write_cache(cache_key, audio)
        return audio

    async def make_synthesis_cache_key(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> Optional[str]:
        """Return a cache key for already-preprocessed synthesis input."""
        synthesizer_type = f"{type(self).__module__}.{type(self).__qualname__}"
        return self.make_cache_key(
            url=f"tts://{synthesizer_type}",
            json_body={
                "text": text,
                "style_info": style_info,
                "language": language,
            },
        )

    @abstractmethod
    async def generate(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        """Generate audio from text that has already passed through preprocessors."""
        pass

    def get_config(self) -> dict:
        return {
            "style_mapper": self.style_mapper,
            "sample_rate": self.sample_rate,
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
        postprocessors: List[TTSPostprocessor] = None,
        sample_rate: int = None,
        follow_redirects: bool = False,
        cache_dir: str = None,
        cache_ext: str = "wav",
        debug: bool = False
    ):
        super().__init__(
            style_mapper=style_mapper,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            sample_rate=sample_rate,
            follow_redirects=follow_redirects,
            cache_dir=cache_dir,
            cache_ext=cache_ext,
            debug=debug
        )
        self.synthesized_bytes = synthesized_bytes
        self.wait_sec = wait_sec

    async def make_synthesis_cache_key(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> Optional[str]:
        return None

    async def generate(self, text: str, style_info: dict = None, language: str = None) -> bytes:
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
    postprocessors: List[TTSPostprocessor] = None,
    sample_rate: int = None,
    follow_redirects: bool = False,
    cache_dir: str = None,
    cache_ext: str = "wav",
    debug = False
) -> SpeechSynthesizer:
    class InstantSynthesizer(SpeechSynthesizer):
        def _make_http_request(
            self,
            text: str,
            style_info: dict = None,
            language: str = None,
        ) -> httpx.Request:
            if request_maker:
                return request_maker(text, style_info, language)
            else:
                def replace_placeholders(obj: dict, text: str, lang: str):
                    return {k: v.format(text=text, language=lang or "") if isinstance(v, str) else v for k, v in obj.items()}

                dynamic_params = replace_placeholders(params, text, language) if params else None
                dynamic_headers = replace_placeholders(headers, text, language) if headers else None
                dynamic_json = replace_placeholders(json, text, language) if json else None

                return httpx.Request(
                    method=method,
                    url=url,
                    params=dynamic_params,
                    headers=dynamic_headers,
                    json=dynamic_json
                )

        async def make_synthesis_cache_key(
            self,
            text: str,
            style_info: dict = None,
            language: str = None,
        ) -> Optional[str]:
            http_request = self._make_http_request(text, style_info, language)
            return self.make_cache_key(
                url=str(http_request.url),
                headers=dict(http_request.headers),
                data=http_request.content,
            )

        async def generate(self, text: str, style_info: dict = None, language: str = None) -> bytes:
            http_request = self._make_http_request(text, style_info, language)
            http_response = await self.http_client.send(http_request)
            http_response.raise_for_status()

            if response_parser:
                result = response_parser(http_response)
                if inspect.iscoroutine(result):
                    result = await result
            else:
                result = http_response.content
            return result

    return InstantSynthesizer(
        style_mapper=style_mapper,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        timeout=timeout,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        sample_rate=sample_rate,
        follow_redirects=follow_redirects,
        cache_dir=cache_dir,
        cache_ext=cache_ext,
        debug=debug
    )
