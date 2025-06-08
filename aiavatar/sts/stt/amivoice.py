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

        resp = await self.http_client.post(
            url=self.url,
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
            return None

        return resp_json.get("text")
