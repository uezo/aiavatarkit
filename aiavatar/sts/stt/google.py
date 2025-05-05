import base64
import logging
from typing import List
from . import SpeechRecognizer

logger = logging.getLogger(__name__)


class GoogleSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        google_api_key: str,
        sample_rate: int = 16000,
        language: str = "ja-JP",
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
        self.google_api_key = google_api_key
        self.sample_rate = sample_rate

    async def transcribe(self, data: bytes) -> str:
        request_body = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.sample_rate,
                "languageCode": self.language,
            },
            "audio": {
                "content": base64.b64encode(data).decode("utf-8")
            },
        }
        if self.alternative_languages:
            request_body["config"]["alternativeLanguageCodes"] = self.alternative_languages

        resp = await self.http_client.post(
            f"https://speech.googleapis.com/v1/speech:recognize?key={self.google_api_key}",
            json=request_body
        )

        try:
            resp_json = resp.json()
        except:
            resp_json = {}

        if resp.status_code != 200:
            logger.error(f"Failed in recognition: {resp.status_code}\n{resp_json}")

        if resp_json.get("results"):
            if recognized_text := resp_json["results"][0]["alternatives"][0].get("transcript"):
                if self.debug:
                    logger.info(f"Recognized: {recognized_text}")
                return recognized_text
