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
        cid: str = None,
        *,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 5.0,
        max_retries: int = 2,
        debug: bool = False
    ):
        super().__init__(
            language=language,
            alternative_languages=alternative_languages,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            max_retries=max_retries,
            debug=debug
        )
        self.azure_api_key = azure_api_key
        self.azure_region = azure_region
        self.sample_rate = sample_rate
        self.cid = cid
        self.use_classic = use_classic
        if self.cid and not self.use_classic:
            self.use_classic = True
            logger.warning(f"Switch to classic mode to use custom model: cid={self.cid}")
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

        params = {"language": self.language}
        if self.cid:
            params["cid"] = self.cid

        resp = await self.http_request_with_retry(
            method="POST",
            url=f"https://{self.azure_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
            headers=headers,
            params=params,
            content=data
        )

        try:
            recognized_text = resp.json()["DisplayText"]
            if self.debug:
                logger.info(f"Recognized: {recognized_text}")
            return recognized_text
        except:
            return None

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

        resp = await self.http_request_with_retry(
            method="POST",
            url=f"https://{self.azure_region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15",
            headers=headers,
            files=files
        )

        try:
            recognized_text = resp.json()["combinedPhrases"][0]["text"]
            if self.debug:
                logger.info(f"Recognized: {recognized_text}")
            return recognized_text
        except:
            return None
