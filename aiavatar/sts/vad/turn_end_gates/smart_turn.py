import asyncio
import logging
import threading
from typing import Any, Optional, Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

from .base import TurnEndDecision, TurnEndGate, TurnEndGateContext

logger = logging.getLogger(__name__)


class SmartTurnEndGate(TurnEndGate):
    """Turn-end confirmation gate backed by pipecat-ai Smart Turn.

    This gate is intended to run after a normal VAD silence threshold is met.
    It accepts Linear16 PCM bytes from the current turn and runs Smart Turn on
    the last max_audio_duration seconds, padding at the beginning when shorter.
    """

    def __init__(
        self,
        *,
        name: str = "smart_turn",
        # Local ONNX model path. When set, Smart Turn does not download the model.
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        max_audio_duration: float = 8.0,
        target_sample_rate: int = 16000,
        timeout: Optional[float] = 1.5,
        providers: Optional[Sequence[str]] = None,
        inter_op_num_threads: int = 1,
        feature_extractor: Any = None,
        session: Any = None,
        hf_repo_id: str = "pipecat-ai/smart-turn-v3",
        hf_filename: str = "smart-turn-v3.2-cpu.onnx",
        debug: bool = False,
    ):
        self.name = name
        self.threshold = threshold
        self.max_audio_duration = max_audio_duration
        self.target_sample_rate = target_sample_rate
        self.timeout = timeout
        self.debug = debug
        self._lock = threading.Lock()

        if feature_extractor is None:
            feature_extractor = WhisperFeatureExtractor(chunk_length=max_audio_duration)
        self.feature_extractor = feature_extractor

        if session is None:
            if model_path is None:
                model_path = self._download_model_path(hf_repo_id, hf_filename)
            session = self._build_session(
                model_path=model_path,
                providers=providers,
                inter_op_num_threads=inter_op_num_threads,
            )
        self.session = session

    def _download_model_path(self, repo_id: str, filename: str) -> str:
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _build_session(
        self,
        *,
        model_path: str,
        providers: Optional[Sequence[str]],
        inter_op_num_threads: int,
    ):
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = inter_op_num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        kwargs = {"sess_options": options}
        if providers:
            kwargs["providers"] = list(providers)
        return ort.InferenceSession(model_path, **kwargs)

    def _linear16_to_float32(self, audio: bytes, channels: int) -> np.ndarray:
        usable_len = len(audio) - (len(audio) % 2)
        if usable_len <= 0:
            return np.zeros(0, dtype=np.float32)

        audio_int16 = np.frombuffer(audio[:usable_len], dtype=np.int16)
        if channels > 1:
            frame_count = len(audio_int16) // channels
            audio_int16 = audio_int16[:frame_count * channels]
            audio_float = audio_int16.reshape(-1, channels).astype(np.float32).mean(axis=1)
            return audio_float / 32768.0

        return audio_int16.astype(np.float32) / 32768.0

    def _resample_if_needed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.target_sample_rate or len(audio) == 0:
            return audio.astype(np.float32, copy=False)

        duration = len(audio) / float(sample_rate)
        target_len = max(1, int(round(duration * self.target_sample_rate)))
        source_positions = np.linspace(0.0, len(audio) - 1, num=len(audio), dtype=np.float32)
        target_positions = np.linspace(0.0, len(audio) - 1, num=target_len, dtype=np.float32)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    def _fit_audio_duration(self, audio: np.ndarray) -> np.ndarray:
        max_samples = int(self.max_audio_duration * self.target_sample_rate)
        if max_samples <= 0:
            raise ValueError("max_audio_duration must be greater than 0")

        if len(audio) > max_samples:
            return audio[-max_samples:].astype(np.float32, copy=False)
        if len(audio) < max_samples:
            padding = max_samples - len(audio)
            return np.pad(audio, (padding, 0), mode="constant", constant_values=0).astype(np.float32)
        return audio.astype(np.float32, copy=False)

    def _prepare_audio(self, audio: bytes, sample_rate: int, channels: int) -> np.ndarray:
        waveform = self._linear16_to_float32(audio, max(1, channels))
        waveform = self._resample_if_needed(waveform, sample_rate)
        return self._fit_audio_duration(waveform)

    def _predict_probability(self, waveform: np.ndarray) -> float:
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=int(self.max_audio_duration * self.target_sample_rate),
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        with self._lock:
            outputs = self.session.run(None, {"input_features": input_features})
        return float(np.asarray(outputs[0]).squeeze().item())

    async def should_end_turn(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        channels: int,
        recorded_duration: float,
        silence_duration: float,
        session_id: str,
        text: Optional[str] = None,
        session: Any = None,
        context: Optional[TurnEndGateContext] = None,
    ) -> TurnEndDecision:
        waveform = self._prepare_audio(audio, sample_rate, channels)
        probability = await asyncio.to_thread(self._predict_probability, waveform)
        should_end = probability >= self.threshold
        if self.debug:
            logger.info(
                "Smart Turn: %s session=%s, probability=%.3f, threshold=%.3f, recorded_duration=%.3f, silence_duration=%.3f, audio_bytes=%s",
                "PASS complete" if should_end else "WAIT incomplete",
                session_id,
                probability,
                self.threshold,
                recorded_duration,
                silence_duration,
                len(audio),
            )
        return TurnEndDecision(
            should_end=should_end,
            confidence=probability,
            reason="smart_turn_complete" if should_end else "smart_turn_incomplete",
            timeout=None if should_end else self.timeout,
        )
