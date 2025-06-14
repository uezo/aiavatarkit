import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .preprocessor import TTSPreprocessor

logger = logging.getLogger(__name__)


class AzureSpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        azure_api_key: str,
        azure_region: str,
        speaker: str,
        style_mapper: Dict[str, str] = None,
        default_language: str = "ja-JP",
        audio_format: str = "riff-16khz-16bit-mono-pcm",
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
        self.azure_api_key = azure_api_key
        self.azure_region = azure_region
        self.speaker = speaker
        self.default_language = default_language
        self.audio_format = audio_format
        self.voice_map = {self.default_language: self.speaker}

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        logger.info(f"Speech synthesize: {text}")

        # Preprocess
        processed_text = await self.preprocess(text, style_info, language)

        headers = {
            "X-Microsoft-OutputFormat": self.audio_format,
            "Content-Type": "application/ssml+xml",
            "Ocp-Apim-Subscription-Key": self.azure_api_key
        }

        speaker = self.voice_map[language or self.default_language]
        ssml_text = f"<speak version='1.0' xml:lang='{language or self.default_language}'><voice xml:lang='{language or self.default_language}' name='{speaker}'>{processed_text}</voice></speak>"
        data = ssml_text.encode("utf-8")

        # Synthesize
        # https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/language-support?tabs=tts
        resp = await self.http_client.post(
            url=f"https://{self.azure_region}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers=headers,
            data=data
        )

        return resp.content
