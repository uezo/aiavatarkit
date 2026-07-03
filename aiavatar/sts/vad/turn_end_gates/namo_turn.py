import asyncio
import logging
import threading
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from .base import TurnEndDecision, TurnEndGate

logger = logging.getLogger(__name__)


_LANGUAGE_MAP = {
    "ar": "Arabic",
    "bn": "Bengali",
    "zh": "Chinese",
    "da": "Danish",
    "nl": "Dutch",
    "de": "German",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
}


def get_namo_repo_id(language: Optional[str] = "ja") -> str:
    if language is None:
        return "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
    lang_name = _LANGUAGE_MAP.get(language.lower(), language.capitalize())
    return f"videosdk-live/Namo-Turn-Detector-v1-{lang_name}"


class NamoTurnEndGate(TurnEndGate):
    """Text-based turn-end confirmation gate backed by Namo Turn Detector v1."""

    def __init__(
        self,
        *,
        language: Optional[str] = "ja",
        repo_id: Optional[str] = None,
        model_path: Optional[str] = None,
        model_filename: str = "model_quant.onnx",
        tokenizer: Any = None,
        session: Any = None,
        threshold: float = 0.5,
        max_length: Optional[int] = None,
        truncation_side: str = "left",
        no_text_should_end: bool = True,
        providers: Optional[list[str]] = None,
        inter_op_num_threads: int = 1,
        debug: bool = False,
    ):
        self.language = language
        self.repo_id = repo_id or get_namo_repo_id(language)
        self.threshold = threshold
        self.no_text_should_end = no_text_should_end
        self.max_length = max_length if max_length is not None else (8192 if language is None else 512)
        self.truncation_side = truncation_side
        self.debug = debug
        self._lock = threading.Lock()

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.repo_id,
            truncation_side=self.truncation_side,
        )

        if session is None:
            if model_path is None:
                model_path = hf_hub_download(repo_id=self.repo_id, filename=model_filename)
            session = self._build_session(
                model_path=model_path,
                providers=providers,
                inter_op_num_threads=inter_op_num_threads,
            )
        self.session = session

    def _build_session(
        self,
        *,
        model_path: str,
        providers: Optional[list[str]],
        inter_op_num_threads: int,
    ):
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = inter_op_num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        kwargs = {"sess_options": options}
        if providers:
            kwargs["providers"] = providers
        return ort.InferenceSession(model_path, **kwargs)

    def _predict(self, text: str) -> tuple[int, float]:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed_dict = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        with self._lock:
            outputs = self.session.run(None, feed_dict)
        probabilities = self._softmax(np.asarray(outputs[0])[0])
        predicted_label = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        return predicted_label, confidence

    def _softmax(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        if axis is None:
            axis = -1
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

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
    ) -> TurnEndDecision:
        normalized_text = (text or "").strip()
        if not normalized_text:
            if self.debug:
                logger.info(
                    "Namo Turn: %s no_text session=%s, recorded_duration=%.3f, silence_duration=%.3f",
                    "PASS complete" if self.no_text_should_end else "WAIT incomplete",
                    session_id,
                    recorded_duration,
                    silence_duration,
                )
            return TurnEndDecision(
                should_end=self.no_text_should_end,
                confidence=None,
                reason="namo_no_text",
            )

        predicted_label, confidence = await asyncio.to_thread(self._predict, normalized_text)
        should_end = predicted_label == 1 and confidence >= self.threshold
        if self.debug:
            logger.info(
                "Namo Turn: %s session=%s, label=%s, confidence=%.3f, threshold=%.3f, text=%r",
                "PASS complete" if should_end else "WAIT incomplete",
                session_id,
                predicted_label,
                confidence,
                self.threshold,
                normalized_text,
            )
        return TurnEndDecision(
            should_end=should_end,
            confidence=confidence,
            reason="namo_end_of_turn" if should_end else "namo_not_end_of_turn",
        )
