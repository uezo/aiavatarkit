import io
import json
import logging
from typing import List
import wave
from . import SpeechRecognizer

logger = logging.getLogger(__name__)


class AzureSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        azure_api_key: str,
        azure_region: str,
        sample_rate: int = 16000,
        language: str = "ja-JP",
        alternative_languages: List[str] = None,
        use_classic: bool = False,
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
        self.azure_api_key = azure_api_key
        self.azure_region = azure_region
        self.sample_rate = sample_rate
        self.use_classic = use_classic
        if self.use_classic and self.alternative_languages:
            logger.warning("Auto language detection is not available in Azure STT v1. Set `use_classic=False` to enable this feature.")

    async def transcribe(self, data: bytes) -> str:
        if self.use_classic:
            return await self.transcribe_classic(data)
        else:
            return await self.transcribe_fast(data)

    async def transcribe_classic(self, data: bytes) -> str:
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_api_key
        }

        resp = await self.http_client.post(
            f"https://{self.azure_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={self.language}",
            headers=headers,
            content=data
        )

        try:
            resp_json = resp.json()
        except:
            resp_json = {}

        if resp.status_code != 200:
            logger.error(f"Failed in recognition: {resp.status_code}\n{resp_json}")

        if recognized_text := resp_json.get("DisplayText"):
            if self.debug:
                logger.info(f"Recognized: {recognized_text}")
            return recognized_text

    def to_wave_file(self, raw_audio: bytes):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16bit
            wf.setframerate(self.sample_rate)  # sample rate
            wf.writeframes(raw_audio)
        buffer.seek(0)
        return buffer

    async def transcribe_fast(self, data: bytes) -> str:
        # Using Fast Transcription
        # https://learn.microsoft.com/en-us/rest/api/speechtotext/transcriptions/transcribe?view=rest-speechtotext-2024-11-15&tabs=HTTP
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_api_key,
        }

        # https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create?tabs=locale-specified#request-configuration-options
        locales = [self.language] + self.alternative_languages
        files = {
            "audio": self.to_wave_file(data),
            "definition": (None, json.dumps({"locales": locales, "channels": [0,1]}), "application/json"),
        }

        resp = await self.http_client.post(
            f"https://{self.azure_region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15",
            headers=headers,
            files=files
        )

        try:
            resp.raise_for_status()
            resp_json = resp.json()
        except:
            logger.error(f"Failed in recognition: {resp.status_code}\n{resp.content}")
            return None

        if recognized_text := resp_json["combinedPhrases"][0]["text"]:
            if self.debug:
                logger.info(f"Recognized: {recognized_text}")
            return recognized_text
