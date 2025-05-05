import logging
from typing import Dict
from . import SpeechSynthesizer

logger = logging.getLogger(__name__)


class OpenAISpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        openai_api_key: str,
        speaker: str,
        model: str = "tts-1",
        style_mapper: Dict[str, str] = None,
        audio_format: str = "wav",
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        super().__init__(
            style_mapper=style_mapper,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        self.openai_api_key = openai_api_key
        self.speaker = speaker
        self.model = model
        self.audio_format = audio_format

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        logger.info(f"Speech synthesize: {text}")

        # Synthesize
        resp = await self.http_client.post(
            url="https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}"
            },
            json= {
                "model": self.model,
                "voice": self.speaker,
                "input": text,
                # "speed": self.speed,
                "response_format": "wav"
            }
        )

        return resp.content
