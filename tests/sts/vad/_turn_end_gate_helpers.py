import asyncio
import configparser
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import numpy as np
import openai
import pytest

from aiavatar.sts.stt.base import SpeechRecognizer
from aiavatar.sts.vad.turn_end_gates import TurnEndDecision, TurnEndGate


class DummyVADIterator:
    def __init__(self, *args, **kwargs):
        pass

    def reset_states(self):
        pass

def fake_init_silero_model(self, model_path=None, hub_cache_path=None):
    self.model_pool = [object()]
    self.model_locks = [threading.Lock()]
    self.VADIteratorClass = DummyVADIterator

class AlwaysHoldGate(TurnEndGate):
    def __init__(self, timeout=0.3):
        self.calls = []
        self.timeout = timeout

    async def should_end_turn(self, **kwargs):
        self.calls.append(kwargs)
        return TurnEndDecision(should_end=False, confidence=0.1, reason="hold", timeout=self.timeout)

class AlwaysHoldForeverGate(AlwaysHoldGate):
    def __init__(self):
        super().__init__(timeout=None)

class HoldThenEndGate(TurnEndGate):
    def __init__(self, timeout=0.3):
        self.calls = 0
        self.timeout = timeout

    async def should_end_turn(self, **kwargs):
        self.calls += 1
        return TurnEndDecision(
            should_end=self.calls > 1,
            confidence=0.9 if self.calls > 1 else 0.1,
            reason="jitter",
            timeout=None if self.calls > 1 else self.timeout,
        )

class AlwaysPassGate(TurnEndGate):
    def __init__(self):
        self.calls = []

    async def should_end_turn(self, **kwargs):
        self.calls.append(kwargs)
        return TurnEndDecision(should_end=True, confidence=0.9, reason="pass")

class BackgroundHoldGate(TurnEndGate):
    run_in_background = True

    def __init__(self, future: asyncio.Future, timeout=0.6):
        self.future = future
        self.calls = []
        self.timeout = timeout
        self.name = "background_hold"

    async def should_end_turn(self, **kwargs):
        self.calls.append(kwargs)
        await self.future
        return TurnEndDecision(
            should_end=False,
            confidence=0.9,
            reason="background_hold",
            timeout=self.timeout,
        )

class BackgroundPassGate(TurnEndGate):
    run_in_background = True

    def __init__(self, future: asyncio.Future, timeout=0.6):
        self.future = future
        self.calls = []
        self.timeout = timeout
        self.name = "background_pass"

    async def should_end_turn(self, **kwargs):
        self.calls.append(kwargs)
        await self.future
        return TurnEndDecision(
            should_end=True,
            confidence=0.9,
            reason="background_pass",
            timeout=None,
        )

class DummySpeechRecognizer(SpeechRecognizer):
    async def transcribe(self, data: bytes) -> str:
        return "こんにちは"

def detect_non_silent(audio_bytes: bytes, session) -> bool:
    return any(audio_bytes)

class FakeFeatureExtractor:
    def __init__(self):
        self.waveforms = []

    def __call__(self, waveform, **kwargs):
        self.waveforms.append((waveform, kwargs))
        return SimpleNamespace(input_features=np.ones((1, 80, 3000), dtype=np.float32))

class FakeOnnxSession:
    def __init__(self, probability: float):
        self.probability = probability
        self.inputs = []

    def run(self, output_names, inputs):
        self.inputs.append(inputs)
        return [np.asarray([[self.probability]], dtype=np.float32)]

class FakeNamoTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {
            "input_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
        }

class FakeNamoSession:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.inputs = []

    def run(self, output_names, inputs):
        self.inputs.append(inputs)
        return [self.logits]

def import_smart_turn_with_fake_dependencies(monkeypatch):
    fake_ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="ORT_SEQUENTIAL"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL"),
        InferenceSession=lambda *args, **kwargs: FakeOnnxSession(probability=0.5),
    )
    fake_transformers = SimpleNamespace(WhisperFeatureExtractor=FakeFeatureExtractor)
    fake_huggingface_hub = SimpleNamespace(hf_hub_download=lambda repo_id, filename: filename)

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    sys.modules.pop("aiavatar.sts.vad.turn_end_gates.smart_turn", None)
    return importlib.import_module("aiavatar.sts.vad.turn_end_gates.smart_turn").SmartTurnEndGate

def import_namo_turn_with_fake_dependencies(monkeypatch):
    tokenizer_calls = []

    fake_ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="ORT_SEQUENTIAL"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL"),
        InferenceSession=lambda *args, **kwargs: FakeNamoSession([0.0, 1.0]),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda repo_id, **kwargs: (
                tokenizer_calls.append((repo_id, kwargs)) or FakeNamoTokenizer()
            )
        )
    )
    fake_huggingface_hub = SimpleNamespace(hf_hub_download=lambda repo_id, filename: filename)

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    sys.modules.pop("aiavatar.sts.vad.turn_end_gates.namo_turn", None)
    NamoTurnEndGate = importlib.import_module("aiavatar.sts.vad.turn_end_gates.namo_turn").NamoTurnEndGate
    return NamoTurnEndGate, tokenizer_calls

def create_openai_client_for_test():
    api_key = os.getenv("OPENAI_API_KEY") or read_pytest_env_value("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for LLMTurnEndGate integration tests")
    return openai.AsyncOpenAI(api_key=api_key)

def read_pytest_env_value(name: str) -> str | None:
    config = configparser.ConfigParser()
    config.read(Path(__file__).parents[3] / "pytest.ini")
    env_block = config.get("pytest", "env", fallback="")
    for line in env_block.splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key.strip() == name:
            return value.strip()
    return None
