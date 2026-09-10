"""Opt-in session-isolation regression using packaged, real Silero models.

Prerequisites: torch, torchaudio, silero-vad and onnxruntime installed locally.
No downloads, credentials, devices, or external STT services are used. Run:

    AIAVATAR_TEST_REAL_SILERO=1 python -m pytest -c /dev/null --rootdir=. \
        -p no:cacheprovider tests/sts/vad/test_silero_session_isolation_real.py -q

Set AIAVATAR_SILERO_ARTIFACT_DIR to an explicit temporary directory to retain
stitched WAV inputs, probability traces, and comparison summaries. Otherwise
pytest's tmp_path owns these outputs. Compare isolated and interleaved runs of
the *same backend*: JIT-versus-ONNX numerical equality is not a requirement.
"""

import os

import pytest

if os.environ.get("AIAVATAR_TEST_REAL_SILERO") != "1":
    pytest.skip(
        "Set AIAVATAR_TEST_REAL_SILERO=1 to run local real-model isolation checks",
        allow_module_level=True,
    )

import asyncio
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import torch
from silero_vad import load_silero_vad
from silero_vad.utils_vad import (
    VADIterator,
    collect_chunks,
    get_speech_timestamps,
    read_audio,
    save_audio,
)

from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
CHUNK_BYTES = CHUNK_SAMPLES * 2
PROBABILITY_ATOL = 1e-6


class RecordingModel:
    """Observe genuine inference without changing the underlying model state."""

    def __init__(self, model, recording):
        self.model = model
        self.recording = recording

    def __call__(self, *args, **kwargs):
        probability = self.model(*args, **kwargs)
        self.recording["active"]["probabilities"].append(probability.item())
        return probability

    def reset_states(self, *args, **kwargs):
        return self.model.reset_states(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __copy__(self):
        return type(self)(copy.copy(self.model), self.recording)


class LocalRecognizer:
    """Exercise stream-VAD bookkeeping without constructing an HTTP client."""

    async def recognize(self, session_id, data):
        return SimpleNamespace(text="local test speech")


def stitched_inputs():
    source = Path(__file__).parents[1] / "stt" / "data" / "hello.wav"
    with wave.open(str(source), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
            SAMPLE_RATE, 1, 2
        )
        speech = wav.readframes(wav.getnframes())
    speech += bytes((-len(speech)) % CHUNK_BYTES)
    silence = bytes(CHUNK_BYTES)
    # Same source is sufficient: shift speech, attenuate it, and swap its halves
    # so each session presents a different waveform/history at the same instant.
    speech_samples = np.frombuffer(speech, dtype="<i2")
    split = (len(speech_samples) // (2 * CHUNK_SAMPLES)) * CHUNK_SAMPLES
    alternate = np.concatenate((speech_samples[split:], speech_samples[:split]))
    alternate = (alternate.astype(np.float32) * 0.35).astype("<i2").tobytes()
    raw = {
        "A": silence * 8 + speech + silence * 24 + speech + silence * 40,
        "B": silence * 40 + alternate + silence * 8 + alternate + silence * 24,
    }
    length = max(map(len, raw.values()))
    raw["C"] = bytes(length)
    return {
        key: [audio[i:i + CHUNK_BYTES] for i in range(0, length, CHUNK_BYTES)]
        for key, audio in ((key, value.ljust(length, b"\0")) for key, value in raw.items())
    }


def new_trace():
    return {"probabilities": [], "speech": [], "recording": [], "utterances": []}


def compare_trace(reference, actual):
    assert len(reference["probabilities"]) == len(actual["probabilities"])
    differences = np.abs(
        np.asarray(reference["probabilities"]) - np.asarray(actual["probabilities"])
    )
    mismatch_indices = np.flatnonzero(differences > PROBABILITY_ATOL)
    return {
        "max_probability_difference": float(differences.max(initial=0)),
        "probability_mismatch_chunks": int(len(mismatch_indices)),
        "first_probability_mismatch_chunk": int(mismatch_indices[0]) if len(mismatch_indices) else None,
        "speech_mismatch_chunks": sum(a != b for a, b in zip(reference["speech"], actual["speech"])),
        "recording_mismatch_chunks": sum(a != b for a, b in zip(reference["recording"], actual["recording"])),
        "utterances_match": reference["utterances"] == actual["utterances"],
        "reference_utterances": len(reference["utterances"]),
        "actual_utterances": len(actual["utterances"]),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("detector_kind", ["base", "stream"])
@pytest.mark.parametrize("use_vad_iterator", [False, True], ids=["direct", "iterator"])
async def test_real_audio_session_isolation(monkeypatch, tmp_path, detector_kind, use_vad_iterator):
    audio = stitched_inputs()
    output_base = Path(os.environ.get("AIAVATAR_SILERO_ARTIFACT_DIR", tmp_path))
    output = output_base / f"{detector_kind}-{'iterator' if use_vad_iterator else 'direct'}"
    output.mkdir(parents=True, exist_ok=True)
    for session_id, chunks in audio.items():
        with wave.open(str(output / f"{session_id}.wav"), "wb") as wav:
            wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            wav.writeframes(b"".join(chunks))

    recording = {"active": None}
    loaded_backends = set()

    def local_hub_load(*args, **kwargs):
        # Preserve the detector's chosen backend, changing only model discovery
        # to the installed package so torch.hub never contacts the network.
        onnx = kwargs["onnx"]
        loaded_backends.add("onnx" if onnx else "jit")
        return RecordingModel(load_silero_vad(onnx=onnx), recording), (
            get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks
        )

    monkeypatch.setattr(torch.hub, "load", local_hub_load)

    async def run(order, lifecycle=False):
        kwargs = dict(
            use_vad_iterator=use_vad_iterator,
            model_pool_size=1,
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SAMPLES,
            silence_duration_threshold=0.3,
            min_duration=0.1,
            max_duration=30,
        )
        if detector_kind == "stream":
            detector = SileroStreamSpeechDetector(speech_recognizer=LocalRecognizer(), **kwargs)
        else:
            detector = SileroSpeechDetector(**kwargs)
        traces = {session_id: new_trace() for session_id in order}
        original_detect = detector._detect_speech_silero

        def observe_detection(data, session):
            result = original_detect(data, session)
            traces[session.session_id]["speech"].append(bool(result))
            return result

        detector._detect_speech_silero = observe_detection

        @detector.on_speech_detected
        async def on_detected(data, text, metadata, recorded_duration, session_id):
            traces[session_id]["utterances"].append({
                "audio_sha256": hashlib.sha256(data).hexdigest(),
                "duration": round(recorded_duration, 9),
                "text": text,
            })

        # Inject lifecycle operations during A's first speech, with no B audio:
        # any change to A therefore comes from B's lifecycle alone.
        lifecycle_events = {20: "create", 28: "reset", 36: "delete"}
        try:
            for chunk_index in range(len(audio["A"])):
                if lifecycle and chunk_index in lifecycle_events:
                    action = lifecycle_events[chunk_index]
                    if action == "create":
                        detector.get_session("B")
                    elif action == "reset":
                        detector.reset_session_audio_state("B")
                    else:
                        detector.delete_session("B")
                for session_id in order:
                    recording["active"] = traces[session_id]
                    is_recording = await detector.process_samples(audio[session_id][chunk_index], session_id)
                    traces[session_id]["recording"].append(bool(is_recording))
                    await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            pending = [
                session.pending_recognition_task
                for session in detector.recording_sessions.values()
                if getattr(session, "pending_recognition_task", None) is not None
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for session_id in list(detector.recording_sessions):
                detector.delete_session(session_id)
            recording["active"] = None
        for trace in traces.values():
            # Detection swallows inference exceptions, so missing probabilities
            # must fail explicitly rather than resemble successful silence.
            assert len(trace["probabilities"]) == len(audio["A"])
        return traces

    runs = {}
    references = {}
    for session_id in audio:
        result = await run([session_id])
        references.update(result)
        runs[f"isolated-{session_id}"] = result
    runs["interleaved-ABC"] = await run(["A", "B", "C"])
    runs["interleaved-CBA"] = await run(["C", "B", "A"])
    runs["B-lifecycle"] = await run(["A"], lifecycle=True)

    comparisons = {
        f"{schedule}/{session_id}": compare_trace(references[session_id], trace)
        for schedule, traces in runs.items() if not schedule.startswith("isolated-")
        for session_id, trace in traces.items()
    }
    summary = {
        "detector": detector_kind,
        "use_vad_iterator": use_vad_iterator,
        "backends": sorted(loaded_backends),
        "versions": {name: importlib.metadata.version(name) for name in ("silero-vad", "torch", "onnxruntime")},
        "sample_rate": SAMPLE_RATE,
        "chunk_samples": CHUNK_SAMPLES,
        "chunks_per_session": len(audio["A"]),
        "probability_absolute_tolerance": PROBABILITY_ATOL,
        "lifecycle_events": {"20": "create B", "28": "reset B audio state", "36": "delete B"},
        "comparisons": comparisons,
        "runs": runs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "probabilities.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["schedule", "session", "chunk", "time_seconds", "probability", "speech", "recording"])
        for schedule, traces in runs.items():
            for session_id, trace in traces.items():
                for index, (probability, speech, active) in enumerate(zip(
                    trace["probabilities"], trace["speech"], trace["recording"]
                )):
                    writer.writerow([schedule, session_id, index, index * CHUNK_SAMPLES / SAMPLE_RATE, probability, speech, active])

    # Confirm the fixture actually exercises speech as well as silence.
    assert any(references["A"]["speech"])
    assert references["A"]["utterances"]
    assert not any(references["C"]["speech"])
    failures = {
        key: result for key, result in comparisons.items()
        if result["probability_mismatch_chunks"]
        or result["speech_mismatch_chunks"]
        or result["recording_mismatch_chunks"]
        or not result["utterances_match"]
    }
    assert not failures, f"Session state interference; artifacts: {output}\n{json.dumps(failures, indent=2)}"
