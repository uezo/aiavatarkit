import logging
from . import SpeechRecognizer

logger = logging.getLogger(__name__)


class AmiVoiceSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        amivoice_api_key: str,
        engine: str = "-a2-ja-general",
        sample_rate: int = 16000,
        target_sample_rate: int = 0,
        *,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 30.0,
        debug: bool = False
    ):
        super().__init__(
            language=None,
            alternative_languages=None,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        self.amivoice_api_key = amivoice_api_key
        self.engine = engine
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.url = "https://acp-api.amivoice.com/v1/recognize"

    async def transcribe(self, data: bytes) -> str:
        form_data = {
            "u": self.amivoice_api_key,
            "d": f"grammarFileNames={self.engine}",
        }

        if self.sample_rate > self.target_sample_rate and self.target_sample_rate > 0:
            sample_rate = self.target_sample_rate
            samples = self.downsample(data, self.sample_rate, self.target_sample_rate)
        else:
            sample_rate = self.sample_rate
            samples = data

        files = {
            "a": ("audio.wav", self.to_wave_file(samples, sample_rate), "audio/wav"),
        }

        resp = await self.http_request_with_retry(
            method="POST",
            url=self.url,
            data=form_data,
            files=files
        )

        try:
            recognized_text = resp.json()["text"]
            if self.debug:
                logger.info(f"Recognized: {recognized_text}")
            return recognized_text
        except:
            return None
