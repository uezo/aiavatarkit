import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from .base import SpeechSynthesizer
from .postprocessor import TTSPostprocessor
from .preprocessor import TTSPreprocessor


DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"

_LANGUAGE_NAMES = {
    "auto": "Auto",
    "zh": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}


def _load_runtime(model_id: str):
    try:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model
    except ImportError as ex:
        raise RuntimeError(
            "Qwen3 MLX TTS requires the optional MLX dependencies; "
            "install aiavatar[mlx-tts]"
        ) from ex
    return load_model(model_id), mx


class Qwen3MLXSpeechSynthesizer(SpeechSynthesizer):
    """Synthesize Qwen3 CustomVoice audio locally through MLX Audio."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = "Vivian",
        instruct: str = None,
        language: str = "Auto",
        max_tokens: int = 1200,
        seed: int = 7,
        style_mapper: Dict[str, str] = None,
        sample_rate: int = None,
        preprocessors: List[TTSPreprocessor] = None,
        postprocessors: List[TTSPostprocessor] = None,
        cache_dir: str = None,
        cache_ext: str = "wav",
        debug: bool = False,
    ):
        super().__init__(
            style_mapper=style_mapper,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            sample_rate=sample_rate,
            cache_dir=cache_dir,
            cache_ext=cache_ext,
            debug=debug,
        )
        if not model:
            raise ValueError("model must not be empty")
        if not voice:
            raise ValueError("voice must not be empty")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer or None")

        self._model_id = model
        self.voice = voice
        self.instruct = instruct
        self.language = language
        self.max_tokens = max_tokens
        self.seed = seed
        self._runtime = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qwen3-mlx-tts",
        )
        self._closed = False

    @property
    def model(self) -> str:
        return self._model_id

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "model": self._model_id,
            "voice": self.voice,
            "instruct": self.instruct,
            "language": self.language,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        })
        return config

    @staticmethod
    def _resolve_language(language: str) -> str:
        normalized = (language or "Auto").strip().lower()
        base = normalized.split("-", 1)[0].split("_", 1)[0]
        return _LANGUAGE_NAMES.get(normalized) or _LANGUAGE_NAMES.get(base, "Auto")

    def _request_config(
        self,
        style_info: dict = None,
        language: str = None,
    ) -> dict:
        return {
            "model": self._model_id,
            "voice": self.parse_style(style_info) or self.voice,
            "instruct": self.instruct,
            "language": self._resolve_language(language or self.language),
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

    async def make_synthesis_cache_key(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> str:
        return self.make_cache_key(
            url=f"qwen3-mlx://{self._model_id}",
            json_body={
                "text": text,
                **self._request_config(style_info, language),
            },
        )

    @staticmethod
    def _wav_bytes(audio_chunks: list, sample_rate: int) -> bytes:
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise RuntimeError("Qwen3 MLX TTS returned an invalid sample rate")

        frames = []
        for audio in audio_chunks:
            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
            if samples.size:
                frames.append(
                    (np.clip(samples, -1.0, 1.0) * 32767.0)
                    .astype("<i2")
                    .tobytes()
                )
        if not frames:
            raise RuntimeError("Qwen3 MLX TTS produced no audio")

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(frames))
        return output.getvalue()

    def _generate_sync(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> bytes:
        if self._runtime is None:
            self._runtime = _load_runtime(self._model_id)
        model, mx = self._runtime
        request = self._request_config(style_info, language)
        if request["seed"] is not None:
            mx.random.seed(request["seed"])

        results = model.generate_custom_voice(
            text=text,
            speaker=request["voice"],
            language=request["language"],
            instruct=request["instruct"],
            max_tokens=request["max_tokens"],
            verbose=False,
        )
        return self._wav_bytes(
            [result.audio for result in results if getattr(result, "audio", None) is not None],
            model.sample_rate,
        )

    async def generate(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> bytes:
        if self._closed:
            raise RuntimeError("Qwen3 MLX TTS synthesizer is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            text,
            style_info,
            language,
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
