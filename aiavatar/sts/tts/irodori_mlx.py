import asyncio
import io
import os
import wave
from numbers import Real
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from .base import SpeechSynthesizer
from .postprocessor import TTSPostprocessor
from .preprocessor import TTSPreprocessor


DEFAULT_MODEL = "mlx-community/Irodori-TTS-v4.1-Small-fp16"


def _load_model(model_id: str):
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as ex:
        raise RuntimeError(
            "Irodori MLX TTS requires the optional MLX dependencies; "
            "install aiavatar[mlx-tts]"
        ) from ex
    return load_model(model_id)


class IrodoriMLXSpeechSynthesizer(SpeechSynthesizer):
    """Synthesize Japanese speech locally with Irodori-TTS through MLX Audio."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        ref_audio: str | List[str] = None,
        instruct: str = None,
        seed: int = 7,
        num_steps: int = 6,
        t_schedule_mode: str = "sway",
        sway_coeff: float = -1.0,
        duration_scale: float = 1.0,
        max_seconds: float = 30.0,
        max_ref_seconds: float = None,
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
        if not isinstance(model, str) or not model:
            raise ValueError("model must not be empty")
        if isinstance(ref_audio, str) and not ref_audio:
            raise ValueError("ref_audio must be a non-empty path or None")
        if isinstance(ref_audio, list) and (
            not ref_audio or any(not isinstance(path, str) or not path for path in ref_audio)
        ):
            raise ValueError("ref_audio must contain non-empty paths")
        if ref_audio is not None and not isinstance(ref_audio, (str, list)):
            raise ValueError("ref_audio must be a path, a list of paths, or None")
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(t_schedule_mode, str) or not t_schedule_mode:
            raise ValueError("t_schedule_mode must not be empty")
        if not isinstance(sway_coeff, Real) or isinstance(sway_coeff, bool):
            raise ValueError("sway_coeff must be a number")
        if (
            not isinstance(duration_scale, Real)
            or isinstance(duration_scale, bool)
            or duration_scale <= 0
        ):
            raise ValueError("duration_scale must be positive")
        if (
            not isinstance(max_seconds, Real)
            or isinstance(max_seconds, bool)
            or max_seconds <= 0
        ):
            raise ValueError("max_seconds must be positive")
        if max_ref_seconds is not None and (
            not isinstance(max_ref_seconds, Real)
            or isinstance(max_ref_seconds, bool)
            or max_ref_seconds <= 0
        ):
            raise ValueError("max_ref_seconds must be positive or None")

        self._model_id = model
        self.ref_audio = ref_audio
        self.instruct = instruct
        self.seed = seed
        self.num_steps = num_steps
        self.t_schedule_mode = t_schedule_mode
        self.sway_coeff = sway_coeff
        self.duration_scale = duration_scale
        self.max_seconds = max_seconds
        self.max_ref_seconds = max_ref_seconds
        self._model = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="irodori-mlx-tts",
        )
        self._closed = False

    @property
    def model(self) -> str:
        return self._model_id

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "model": self._model_id,
            "ref_audio": self.ref_audio,
            "instruct": self.instruct,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "t_schedule_mode": self.t_schedule_mode,
            "sway_coeff": self.sway_coeff,
            "duration_scale": self.duration_scale,
            "max_seconds": self.max_seconds,
            "max_ref_seconds": self.max_ref_seconds,
        })
        return config

    def _request_config(self, style_info: dict = None) -> dict:
        return {
            "ref_audio": self.ref_audio,
            "instruct": self.parse_style(style_info) or self.instruct,
            "rng_seed": self.seed,
            "num_steps": self.num_steps,
            "t_schedule_mode": self.t_schedule_mode,
            "sway_coeff": self.sway_coeff,
            "duration_scale": self.duration_scale,
            "max_seconds": self.max_seconds,
            "max_ref_seconds": self.max_ref_seconds,
        }

    def _reference_cache_config(self):
        if self.ref_audio is None:
            return None
        paths = self.ref_audio if isinstance(self.ref_audio, list) else [self.ref_audio]
        result = []
        for path in paths:
            stat = os.stat(path)
            result.append({
                "path": os.path.abspath(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        return result

    async def make_synthesis_cache_key(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> str:
        request = self._request_config(style_info)
        request["ref_audio"] = await asyncio.to_thread(self._reference_cache_config)
        return self.make_cache_key(
            url=f"irodori-mlx://{self._model_id}",
            json_body={"text": text, **request},
        )

    @staticmethod
    def _wav_bytes(results: list) -> bytes:
        if not results:
            raise RuntimeError("Irodori MLX TTS produced no audio")
        sample_rates = {getattr(result, "sample_rate", None) for result in results}
        if len(sample_rates) != 1:
            raise RuntimeError("Irodori MLX TTS returned inconsistent sample rates")
        sample_rate = sample_rates.pop()
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise RuntimeError("Irodori MLX TTS returned an invalid sample rate")

        frames = []
        for result in results:
            samples = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            if samples.size:
                frames.append(
                    (np.clip(samples, -1.0, 1.0) * 32767.0)
                    .astype("<i2")
                    .tobytes()
                )
        if not frames:
            raise RuntimeError("Irodori MLX TTS produced no audio")

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
        if self._model is None:
            self._model = _load_model(self._model_id)
        request = {
            key: value
            for key, value in self._request_config(style_info).items()
            if value is not None
        }
        return self._wav_bytes(list(self._model.generate(text=text, **request)))

    async def generate(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> bytes:
        if self._closed:
            raise RuntimeError("Irodori MLX TTS synthesizer is closed")
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
