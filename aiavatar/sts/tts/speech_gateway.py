import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .postprocessor import TTSPostprocessor
from .preprocessor import TTSPreprocessor

try:
    from speech_gateway.gateway import SpeechGateway
    from speech_gateway.gateway.unified import UnifiedGateway
    from speech_gateway.gateway import UnifiedTTSRequest
except ImportError:
    UnifiedGateway = None
    UnifiedTTSRequest = None

logger = logging.getLogger(__name__)


class SpeechGatewaySpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        service_name: str = None,
        speaker: str = None,
        speed: float = None,
        style_mapper: Dict[str, str] = None,
        tts_url: str = "http://127.0.0.1:8000/tts",
        audio_format: str = None,
        use_local_gateway: bool = False,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        preprocessors: List[TTSPreprocessor] = None,
        postprocessors: List[TTSPostprocessor] = None,
        sample_rate: int = None,
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
            cache_dir=cache_dir,
            cache_ext=cache_ext,
            debug=debug
        )
        self.service_name = service_name
        self.speaker = speaker
        self.speed = speed
        self.tts_url = tts_url
        self.audio_format = audio_format
        self.use_local_gateway = use_local_gateway
        if self.use_local_gateway:
            if UnifiedGateway is None:
                raise ImportError(
                    "speech_gateway is required for use_local_gateway=True. "
                    "Install it with: pip install speech-gateway"
                )
            self.unified_gateway = UnifiedGateway()
        else:
            self.unified_gateway = None

    def get_config(self) -> dict:
        config = super().get_config()
        config["service_name"] = self.service_name
        config["speaker"] = self.speaker
        config["speed"] = self.speed
        config["tts_url"] = self.tts_url
        config["audio_format"] = self.audio_format
        return config

    def add_local_gateway(
        self, name: str,
        gateway,
        *,
        speaker: str = None,
        languages: List[str] = None,
        default: bool = False
    ):
        if not self.unified_gateway:
            self.unified_gateway = UnifiedGateway()
        self.unified_gateway.add_gateway(
            name, gateway,
            languages=languages,
            default_speaker=speaker
        )
        if default:
            self.service_name = name
            self.speaker = speaker

    def _build_request_json(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> dict:
        request_json = {"text": text}
        if self.service_name:
            request_json["service_name"] = self.service_name
        if self.speaker:
            request_json["speaker"] = self.speaker
        if self.speed:
            request_json["speed"] = self.speed

        # Apply style
        if style := self.parse_style(style_info):
            request_json["style"] = style

        # Apply speed
        if speed := (style_info or {}).get("info", {}).get("speed"):
            request_json["speed"] = speed

        # Apply language
        if language and language != "ja-JP":
            request_json["language"] = language
            request_json.pop("service_name", None)
            request_json.pop("speaker", None)

        # Apply audio format
        if self.audio_format:
            request_json["audio_format"] = self.audio_format
        return request_json

    async def make_synthesis_cache_key(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> str:
        request_json = self._build_request_json(text, style_info, language)
        url = "local://speech-gateway" if self.use_local_gateway else self.tts_url
        return self.make_cache_key(url=url, json_body=request_json)

    async def generate(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        request_json = self._build_request_json(text, style_info, language)

        if self.debug:
            if "style" in request_json:
                logger.info(f"Apply style: {request_json['style']}")
            if "speed" in request_json:
                logger.info(f"Apply speed: {request_json['speed']}")
            if language and language != "ja-JP":
                logger.info(f"Apply language: {language}")
            if self.audio_format:
                logger.info(f"Apply audio format: {self.audio_format}")

        # Synthesize
        if self.use_local_gateway:
            resp = await self.unified_gateway.tts(
                UnifiedTTSRequest(**request_json)
            )
            return resp.audio_data
        else:
            resp = await self.http_client.post(
                url=self.tts_url,
                json=request_json
            )
            return resp.content
