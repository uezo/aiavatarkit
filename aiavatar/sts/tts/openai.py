import logging
from typing import Dict, List
from . import SpeechSynthesizer
from .preprocessor import TTSPreprocessor

logger = logging.getLogger(__name__)


class OpenAISpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        openai_api_key: str,
        base_url: str = "https://api.openai.com/v1",
        speaker: str = "sage",
        model: str = "tts-1",
        instructions: str = None,
        style_mapper: Dict[str, str] = None,
        audio_format: str = "wav",
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
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.speaker = speaker
        self.model = model
        self.instructions = instructions
        self.audio_format = audio_format

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        logger.info(f"Speech synthesize: {text}")

        # Preprocess
        processed_text = await self.preprocess(text, style_info, language)

        # Headers and params
        if "azure" in self.base_url:
            url = self.base_url
            headers = {"api-key": self.openai_api_key}
        else:
            url = f"{self.base_url}/audio/speech"
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}

        # Synthesize
        resp = await self.http_client.post(
            url=url,
            headers=headers,
            json= {
                "model": self.model,
                "voice": self.speaker,
                "input": processed_text,
                "instructions": self.instructions,
                # "speed": self.speed,
                "response_format": "wav"
            }
        )

        return resp.content
