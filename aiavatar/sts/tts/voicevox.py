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
        self.base_url = base_url
        self.speaker = speaker

    async def get_audio_query(self, text: str, speaker: int):
        url = f"{self.base_url}/audio_query"
        response = await self.http_client.post(url, params={"speaker": speaker, "text": text})
        response.raise_for_status()
        return response.json()

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        logger.info(f"Speech synthesize: {text}")

        # Preprocess
        processed_text = await self.preprocess(text, style_info, language)

        speaker = self.speaker

        # Apply style
        if style := self.parse_style(style_info):
            speaker = int(style)
            logger.info(f"Apply style: {speaker}")

        # Make query
        audio_query = await self.get_audio_query(processed_text, speaker)

        # Synthesize
        response = await self.http_client.post(
            url=self.base_url + "/synthesis",
            params={"speaker": speaker},
            json=audio_query
        )
        return response.content
