import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .postprocessor import TTSPostprocessor
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

    def _get_speaker(self, style_info: dict = None) -> int:
        speaker = self.speaker
        if style := self.parse_style(style_info):
            speaker = int(style)
        return speaker

    async def make_synthesis_cache_key(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> str:
        speaker = self._get_speaker(style_info)
        return self.make_cache_key(
            url=self.base_url + "/synthesis",
            params={"speaker": speaker},
            data=text.encode(),
        )

    async def generate(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        speaker = self._get_speaker(style_info)
        if self.parse_style(style_info):
            logger.info(f"Apply style: {speaker}")

        url = self.base_url + "/synthesis"
        params = {"speaker": speaker}

        # Make query
        audio_query = await self.get_audio_query(text, speaker)

        # Synthesize
        response = await self.http_client.post(url=url, params=params, json=audio_query)
        return response.content
