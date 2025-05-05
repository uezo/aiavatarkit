import io
import logging
from typing import List
import wave
from . import SpeechRecognizer

logger = logging.getLogger(__name__)


class OpenAISpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        openai_api_key: str,
        sample_rate: int = 16000,
        language: str = "ja",
        alternative_languages: List[str] = None,
        *,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        super().__init__(
            language=language,
            alternative_languages=alternative_languages,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        self.openai_api_key = openai_api_key
        self.sample_rate = sample_rate

    def to_wave_file(self, raw_audio: bytes):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16bit
            wf.setframerate(self.sample_rate)  # sample rate
            wf.writeframes(raw_audio)
        buffer.seek(0)
        return buffer

    async def transcribe(self, data: bytes) -> str:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        form_data = {
            "model": "whisper-1",
        }

        if self.language and not self.alternative_languages:
            form_data["language"] = self.language.split("-")[0] if "-" in self.language else self.language

        files = {
            "file": ("voice.wav", self.to_wave_file(data), "audio/wav"),
        }

        resp = await self.http_client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            data=form_data,
            files=files
        )

        try:
            resp_json = resp.json()
        except:
            resp_json = {}
            return None

        if resp.status_code != 200:
            logger.error(f"Failed in recognition: {resp.status_code}\n{resp_json}")

        return resp_json.get("text")
