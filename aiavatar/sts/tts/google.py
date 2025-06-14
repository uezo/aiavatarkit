import base64
import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .preprocessor import TTSPreprocessor


logger = logging.getLogger(__name__)


class GoogleSpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        google_api_key: str,
        speaker: str,
        default_language: str = "ja-JP",
        style_mapper: Dict[str, str] = None,
        audio_format: str = "LINEAR16",
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
        self.google_api_key = google_api_key
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

        # Set language and speaker
        voice = {"languageCode": self.default_language, "name": self.speaker}
        if language:
            if language.startswith("zh-"):
                language = language.replace("zh-", "cmn-CN")
            if language in self.voice_map:
                voice = {"languageCode": language, "name": self.voice_map[language]}

        # Synthesize
        # https://cloud.google.com/text-to-speech/docs/voices
        resp = await self.http_client.post(
            url=f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_api_key}",
            json={
                "input": {"text": processed_text},
                "voice": voice,
                "audioConfig": {"audioEncoding": self.audio_format}
            }
        )
        resp_json = resp.json()

        return base64.b64decode(resp_json["audioContent"])
