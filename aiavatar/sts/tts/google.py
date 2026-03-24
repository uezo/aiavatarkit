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
        cache_dir: str = None,
        cache_ext: str = "pcm",
        debug: bool = False
    ):
        super().__init__(
            style_mapper=style_mapper,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            preprocessors=preprocessors,
            cache_dir=cache_dir,
            cache_ext=cache_ext,
            debug=debug
        )
        self.google_api_key = google_api_key
        self.speaker = speaker
        self.default_language = default_language
        self.audio_format = audio_format
        self.voice_map = {self.default_language: self.speaker}

    def get_config(self) -> dict:
        config = super().get_config()
        config["speaker"] = self.speaker
        config["default_language"] = self.default_language
        config["audio_format"] = self.audio_format
        return config

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        if self.debug:
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

        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.google_api_key}"
        json_body = {
            "input": {"text": processed_text},
            "voice": voice,
            "audioConfig": {"audioEncoding": self.audio_format}
        }

        # Check cache
        cache_key = self.make_cache_key(url=url, json_body=json_body)
        if cached := await self.read_cache(cache_key):
            return cached

        # Synthesize
        # https://cloud.google.com/text-to-speech/docs/voices
        resp = await self.http_client.post(url=url, json=json_body)
        resp_json = resp.json()

        audio_data = base64.b64decode(resp_json["audioContent"])
        await self.write_cache(cache_key, audio_data)
        return audio_data
