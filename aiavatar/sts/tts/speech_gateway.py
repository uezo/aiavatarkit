import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .preprocessor import TTSPreprocessor

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

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        logger.info(f"Speech synthesize: {text}")

        # Preprocess
        processed_text = await self.preprocess(text, style_info, language)

        # Audio format
        query_params = {"x_audio_format": self.audio_format} if self.audio_format else {}

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
            logger.info(f"Apply style: {style}")

        # Apply language
        if language and language != "ja-JP":
            logger.info(f"Apply language: {language}")
            request_json["language"] = language
            del request_json["service_name"]
            del request_json["speaker"]

        # Synthesize
        resp = await self.http_client.post(
            url=self.tts_url,
            params=query_params,
            json=request_json
        )

        return resp.content
