import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np

from .base import SpeechRecognizer


DEFAULT_MODEL = "mlx-community/whisper-turbo"
WHISPER_SAMPLE_RATE = 16000


def _transcribe(audio: np.ndarray, **kwargs) -> dict:
    try:
        import mlx_whisper
    except ImportError as ex:
        raise RuntimeError(
            "MLX speech recognition requires the optional MLX dependencies; "
            "install aiavatar[mlx-stt]"
        ) from ex
    return mlx_whisper.transcribe(audio, **kwargs)


class MLXSpeechRecognizer(SpeechRecognizer):
    """Recognize 16 kHz mono PCM speech locally with MLX Whisper."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        min_data_length: int = 4096,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: str = None,
        alternative_languages: List[str] = None,
        initial_prompt: str = None,
        debug: bool = False,
    ):
        super().__init__(
            language=language,
            alternative_languages=alternative_languages,
            debug=debug,
        )
        if not model:
            raise ValueError("model must not be empty")
        if sample_rate != WHISPER_SAMPLE_RATE:
            raise ValueError("MLX Whisper requires 16000 Hz input audio")
        if (
            not isinstance(min_data_length, int)
            or isinstance(min_data_length, bool)
            or min_data_length < 0
        ):
            raise ValueError("min_data_length must be a non-negative integer")

        self._model_id = model
        self.min_data_length = min_data_length
        self.sample_rate = sample_rate
        self.initial_prompt = initial_prompt
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-whisper-stt",
        )
        self._closed = False

    @property
    def model(self) -> str:
        return self._model_id

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "model": self._model_id,
            "min_data_length": self.min_data_length,
            "sample_rate": self.sample_rate,
            "initial_prompt": self.initial_prompt,
        })
        return config

    def _transcribe_sync(self, data: bytes) -> str:
        audio = np.frombuffer(data, dtype="<i2").astype(np.float32)
        audio /= 32768.0
        language = None
        if self.language and not self.alternative_languages:
            language = self.language.split("-", 1)[0].split("_", 1)[0].lower()

        result = _transcribe(
            audio,
            path_or_hf_repo=self._model_id,
            language=language,
            initial_prompt=self.initial_prompt,
            task="transcribe",
            verbose=None,
        )
        text = result.get("text") if isinstance(result, dict) else None
        return text.strip() if isinstance(text, str) and text.strip() else None

    async def transcribe(self, data: bytes) -> str:
        if len(data) < self.min_data_length:
            return None
        if len(data) % 2:
            raise ValueError("16-bit PCM audio must contain an even number of bytes")
        if self._closed:
            raise RuntimeError("MLX speech recognizer is closed")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            data,
        )

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await super().close()
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
