import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .preprocessor import TTSPreprocessor

logger = logging.getLogger(__name__)


class VoicevoxSpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:50021",
        speaker: int = 46,
        style_mapper: Dict[str, str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        preprocessors: List[TTSPreprocessor] = None,
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
            cache_dir=cache_dir,
            cache_ext=cache_ext,
            debug=debug
        )
        self.base_url = base_url
        self.speaker = speaker

    def get_config(self) -> dict:
        config = super().get_config()
        config["base_url"] = self.base_url
        config["speaker"] = self.speaker
        return config

    async def get_audio_query(self, text: str, speaker: int):
        url = f"{self.base_url}/audio_query"
        response = await self.http_client.post(url, params={"speaker": speaker, "text": text})
        response.raise_for_status()
        return response.json()

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        if self.debug:
            logger.info(f"Speech synthesize: {text}")

        # Preprocess
        processed_text = await self.preprocess(text, style_info, language)

        speaker = self.speaker

        # Apply style
        if style := self.parse_style(style_info):
            speaker = int(style)
            logger.info(f"Apply style: {speaker}")

        url = self.base_url + "/synthesis"
        params = {"speaker": speaker}

        # Check cache (audio_query is deterministic for text + speaker)
        cache_key = self.make_cache_key(url=url, params=params, data=processed_text.encode())
        if cached := await self.read_cache(cache_key):
            return cached

        # Make query
        audio_query = await self.get_audio_query(processed_text, speaker)

        # Synthesize
        response = await self.http_client.post(url=url, params=params, json=audio_query)

        await self.write_cache(cache_key, response.content)
        return response.content
