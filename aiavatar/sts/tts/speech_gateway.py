import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .preprocessor import TTSPreprocessor

try:
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

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        if self.debug:
            logger.info(f"Speech synthesize: {text}")

        # Preprocess
        processed_text = await self.preprocess(text, style_info, language)

        # Make basic params
        request_json = {"text": processed_text}
        if self.service_name:
            request_json["service_name"] = self.service_name
        if self.speaker:
            request_json["speaker"] = self.speaker
        if self.speed:
            request_json["speed"] = self.speed

        # Apply style
        if style := self.parse_style(style_info):
            request_json["style"] = style
            if self.debug:
                logger.info(f"Apply style: {style}")

        # Apply speed
        if speed := (style_info or {}).get("info", {}).get("speed"):
            request_json["speed"] = speed
            if self.debug:
                logger.info(f"Apply speed: {speed}")

        # Apply language
        if language and language != "ja-JP":
            logger.info(f"Apply language: {language}")
            request_json["language"] = language
            del request_json["service_name"]
            del request_json["speaker"]

        # Apply audio format
        if self.audio_format:
            request_json["audio_format"] = self.audio_format
            if self.debug:
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
