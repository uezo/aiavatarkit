"""Hermetic Python-backend tests: synthetic PCM, fake inference, no model download."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from aiavatar.sts.vad.filters.hush import HushAudioFilter
from aiavatar.sts.vad.filters.hush.model_backend.dsp import DspState
from aiavatar.sts.vad.filters.hush.model_backend.runtime import PythonModel


def _pcm(values):
    return np.asarray(values, dtype="<i2").tobytes()


class _BackendControl:
    def __init__(self):
        self.models = []
        self.states = []
        self.threads = []
        self.load_started = threading.Event()
        self.load_release = threading.Event()
        self.load_release.set()
        self.load_error = None
        self.before_frame = None
        self.failure = None


@pytest.fixture
def python_filter(tmp_path, monkeypatch):
    """Exercise the public queue with a stateful, one-hop-delay Python backend stand-in."""
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.model_backend.runtime")
    control = _BackendControl()

    class FakePythonModel:
        def __init__(self, model_path, atten_lim_db):
            control.threads.append(threading.get_ident())
            control.load_started.set()
            assert control.load_release.wait(5), "Test did not release model loading"
            if control.load_error:
                raise control.load_error
            self.atten_lim_db = atten_lim_db
            control.models.append(self)

        def create_state(self):
            state = SimpleNamespace(previous=np.zeros(160, dtype=np.float32),
                                    frames=[], active=False, threads=[])
            control.states.append(state)
            return state

        def process_frame(self, state, audio):
            assert not state.active, "Concurrent use of one conversation state"
            state.active = True
            try:
                control.threads.append(threading.get_ident())
                state.threads.append(threading.get_ident())
                state.frames.append(audio.copy())
                if control.before_frame:
                    control.before_frame(state, audio)
                if control.failure == "exception":
                    raise RuntimeError("fake Python inference failed")
                result = state.previous.copy()
                state.previous[:] = audio
                if control.failure == "nonfinite":
                    result[0] = np.nan
                return result
            finally:
                state.active = False

        def close(self):
            assert not any(state.active for state in control.states)

    def reject_native_load(*args, **kwargs):
        pytest.fail("The explicitly selected Python backend attempted to load a native C library")

    monkeypatch.setattr(module, "PythonModel", FakePythonModel)
    monkeypatch.setattr(ctypes, "CDLL", reject_native_load)
    model = tmp_path / "stateful.onnx"
    model.touch()
    filters = []

    def factory(**kwargs):
        instance = HushAudioFilter(backend="python", model_path=model, **kwargs)
        filters.append(instance)
        return instance

    yield factory, control
    control.load_release.set()
    for instance in filters:
        instance.close()


@pytest.mark.asyncio
async def test_explicit_python_backend_loads_off_loop_without_native_library(python_filter):
    factory, control = python_filter
    control.load_release.clear()
    instance = factory(atten_lim_db=36)
    assert instance.get_config()["backend"] == "python"
    assert instance.set_config({"backend": "native"}) == {}
    assert instance.get_config()["backend"] == "python"
    assert not control.load_started.is_set()
    preparing = asyncio.create_task(instance.prepare())
    try:
        assert await asyncio.to_thread(control.load_started.wait, 2)
        assert not preparing.done()
        assert threading.get_ident() not in control.threads
    finally:
        control.load_release.set()
        await preparing
    assert len(control.models) == 1
    assert control.models[0].atten_lim_db == 36


@pytest.mark.asyncio
async def test_python_filter_preserves_pcm_across_interleaved_streams_and_flush(python_filter):
    factory, control = python_filter
    instance = factory(max_workers=2)
    await instance.prepare()
    left = _pcm([-32768, -1, 0, 1, 32767] * 79)
    right = _pcm([0] * 323 + [1200] * 97)
    result = {"left": bytearray(), "right": bytearray()}
    for offset in range(0, max(len(left), len(right)), 137):
        for name, audio in (("left", left), ("right", right)):
            result[name].extend(await instance.process_async(audio[offset:offset + 137], name))
    for name, audio in (("left", left), ("right", right)):
        result[name].extend(await instance.flush_async(name))
        actual = np.frombuffer(result[name], dtype="<i2").astype(np.int32)
        expected = np.frombuffer(audio, dtype="<i2").astype(np.int32)
        assert actual.shape == expected.shape
        if name == "left":
            np.testing.assert_array_equal(actual, expected)
        else:
            # Both backends clock native near-silence with a tiny signal.
            assert np.max(np.abs(actual - expected)) <= 14
        assert await instance.flush_async(name) == b""
    assert len(control.models) == 1
    assert len(control.states) == 2
    assert threading.get_ident() not in control.threads


@pytest.mark.asyncio
async def test_python_reset_discards_inflight_state_without_blocking_peer(python_filter):
    factory, control = python_filter
    instance = factory(max_workers=2)
    await instance.prepare()
    entered, release = threading.Event(), threading.Event()

    def hold_old_state(state, audio):
        if state is control.states[0]:
            entered.set()
            assert release.wait(5), "Test did not release the old conversation"

    control.before_frame = hold_old_state
    instance.process(_pcm([2000] * 160), "speaker")
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        instance.reset_session("speaker")
        fresh = _pcm([-6000] * 321)
        output = await asyncio.wait_for(instance.process_async(fresh, "speaker"), 2)
        output += await asyncio.wait_for(instance.flush_async("speaker"), 2)
        assert output == fresh
        assert not release.is_set()
    finally:
        release.set()
    await instance.aclose()
    assert len(control.states) == 2
    assert all(not state.active for state in control.states)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
async def test_python_failure_is_not_passthrough_and_reset_recovers(python_filter, failure):
    factory, control = python_filter
    instance = factory()
    await instance.prepare()
    control.failure = failure
    with pytest.raises(RuntimeError, match="inference|non-finite|failed"):
        await instance.process_async(_pcm([3000] * 320), "speaker")
    instance.reset_session("speaker")
    control.failure = None
    fresh = _pcm([-1000] * 161)
    output = await instance.process_async(fresh, "speaker")
    output += await instance.flush_async("speaker")
    assert output == fresh


@pytest.mark.asyncio
async def test_python_load_failure_is_visible_and_close_is_idempotent(python_filter):
    factory, control = python_filter
    control.load_error = ImportError("onnxruntime is required")
    instance = factory()
    with pytest.raises(ImportError, match="onnxruntime"):
        await instance.prepare()
    with pytest.raises(ImportError, match="onnxruntime"):
        instance.process(_pcm([1000] * 160), "speaker")
    await instance.aclose()
    await instance.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        instance.process(b"", "speaker")


@pytest.mark.asyncio
async def test_python_close_waits_for_active_inference(python_filter):
    factory, control = python_filter
    instance = factory(max_workers=2)
    await instance.prepare()
    entered, release = threading.Event(), threading.Event()

    def block(state, audio):
        entered.set()
        assert release.wait(5), "Test did not release inference before close"

    control.before_frame = block
    instance.process(_pcm([1234] * 160), "speaker")
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        closing = asyncio.create_task(instance.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        assert control.states[0].active
    finally:
        release.set()
        if closing:
            await asyncio.wait_for(closing, 3)
    assert not control.states[0].active


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    """A model with real DSP but deterministic explicit recurrent outputs."""
    specs = [
        ("enc_conv_erb", [1, 1, 2, 32], "enc"),
        ("enc_conv_spec", [1, 2, 2, 64], "enc"),
        ("enc_gru_0", [1, 1, 256], "enc"),
        ("erb_gru_0", [1, 1, 256], "erb"),
        ("df_gru_0", [1, 1, 256], "df"),
        ("df_gru_1", [1, 1, 256], "df"),
        ("df_gru_2", [1, 1, 256], "df"),
    ]
    states = [dict(input="state_" + name, output="next_state_" + name,
                   shape=shape, dtype="float32", stage=stage)
              for name, shape, stage in specs]
    config = dict(sr=16000, hop_size=160, fft_size=320, nb_erb=32,
                  nb_df=64, df_order=5, min_nb_erb_freqs=2, norm_tau=1.0,
                  conv_lookahead=0, df_lookahead=0, conv_ch=16)
    metadata = {"hush.format_version": "1", "hush.config": json.dumps(config),
                "hush.states": json.dumps(states)}

    def node(name, shape):
        return SimpleNamespace(name=name, shape=shape, type="tensor(float)")

    class Session:
        def __init__(self):
            self.inputs = [node("feat_erb", [1, 1, 1, 32]), node("feat_spec", [1, 2, 1, 64])]
            self.inputs += [node(s["input"], s["shape"]) for s in states]
            self.outputs = [node("mask", [1, 1, 1, 32]), node("coefs", [1, 1, 64, 10]),
                            node("lsnr", [1, 1, 1])]
            self.outputs += [node(s["output"], s["shape"]) for s in states]
            self.options = None
            self.providers = None
            self.snr = 0.0
            self.calls = []
            self.threads = []

        def load(self, model, options, providers):
            self.options, self.providers = options, providers
            return self

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map=metadata)

        def get_inputs(self):
            return self.inputs

        def get_outputs(self):
            return self.outputs

        def run(self, names, inputs):
            self.calls.append({name: value.copy() for name, value in inputs.items()})
            self.threads.append(threading.get_ident())
            mask = np.ones((1, 1, 1, 32), dtype=np.float32)
            coefs = np.zeros((1, 1, 64, 10), dtype=np.float32)
            coefs.reshape(64, 5, 2)[:, -1, 0] = 1
            outputs = [mask, coefs, np.full((1, 1, 1), self.snr, dtype=np.float32)]
            outputs += [inputs[s["input"]] + np.float32(1) for s in states]
            return outputs

    session = Session()
    runtime = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=session.load,
                              ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
                              GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"))
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    graph = importlib.import_module("aiavatar.sts.vad.filters.hush.model_backend.graph")

    def reject_conversion(*args, **kwargs):
        pytest.fail("A preconverted ONNX must not invoke the optional graph converter")

    monkeypatch.setattr(graph, "build_streaming_model", reject_conversion)
    model_path = tmp_path / "fake-streaming.onnx"
    model_path.touch()
    return SimpleNamespace(path=model_path, session=session, metadata=metadata, states=states)


def test_python_session_uses_shared_single_thread_runtime_without_cpu_arena(fake_runtime):
    model = PythonModel(fake_runtime.path, 100)
    session = fake_runtime.session
    assert model.session is session
    assert session.options.intra_op_num_threads == session.options.inter_op_num_threads == 1
    assert session.options.execution_mode == "sequential"
    assert session.options.graph_optimization_level == "all"
    assert session.options.enable_cpu_mem_arena is False
    assert session.providers == ["CPUExecutionProvider"]


def test_recurrent_and_dsp_state_survive_worker_migration_and_interleaving(fake_runtime):
    model = PythonModel(fake_runtime.path, 100)
    left, right = model.create_state(), model.create_state()
    frame = np.linspace(-.1, .1, 160, dtype=np.float32)
    for name in left.inputs:
        assert not np.shares_memory(left.inputs[name], right.inputs[name])
    # Keep both dedicated threads alive so a reused OS thread id cannot mask
    # accidental thread-local storage. One conversation deliberately migrates.
    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        out0 = first.submit(model.process_frame, left, frame).result().copy()
        second.submit(model.process_frame, right, -frame).result()
        out1 = second.submit(model.process_frame, left, frame).result().copy()
        out2 = first.submit(model.process_frame, right, -frame).result().copy()
    assert fake_runtime.session.threads[0] != fake_runtime.session.threads[2]
    for info in fake_runtime.states:
        np.testing.assert_array_equal(left.inputs[info["input"]], 2)
        np.testing.assert_array_equal(right.inputs[info["input"]], 2)
    assert np.max(np.abs(out0)) < 1e-6
    np.testing.assert_allclose(out1, frame, atol=1e-6)
    np.testing.assert_allclose(out2, -frame, atol=1e-6)
    assert left.dsp is not right.dsp
    assert all(np.isfinite(value).all() for call in fake_runtime.session.calls for value in call.values())


@pytest.mark.parametrize("snr,decoder_updates", [(-16, False), (-15, True), (35, True), (36, False)])
def test_decoder_state_pauses_only_outside_native_snr_thresholds(fake_runtime, snr, decoder_updates):
    model = PythonModel(fake_runtime.path, 100)
    state = model.create_state()
    fake_runtime.session.snr = snr
    model.process_frame(state, np.full(160, .05, dtype=np.float32))
    for info in fake_runtime.states:
        expected = 1 if info["stage"] == "enc" or decoder_updates else 0
        np.testing.assert_array_equal(state.inputs[info["input"]], expected)
    # Re-entering the normal range resumes each decoder from its own retained
    # state, while the encoder has advanced throughout.
    fake_runtime.session.snr = 0
    model.process_frame(state, np.full(160, -.05, dtype=np.float32))
    for info in fake_runtime.states:
        expected = 2 if info["stage"] == "enc" or decoder_updates else 1
        np.testing.assert_array_equal(state.inputs[info["input"]], expected)


def test_python_attenuation_limit_preserves_requested_dry_signal(fake_runtime):
    model = PythonModel(fake_runtime.path, 20)
    state = model.create_state()
    fake_runtime.session.snr = -20  # Native low-SNR path suppresses every band.
    frame = np.linspace(-.2, .2, 160, dtype=np.float32)
    model.process_frame(state, frame)
    output = model.process_frame(state, np.zeros(160, dtype=np.float32)).copy()
    np.testing.assert_allclose(output, frame * .1, atol=1e-6)


@pytest.mark.parametrize("invalid", ["version", "geometry", "input_shape", "output_order", "states"])
def test_python_rejects_incompatible_export_before_audio(fake_runtime, invalid):
    if invalid == "version":
        fake_runtime.metadata["hush.format_version"] = "0"
    elif invalid == "geometry":
        config = json.loads(fake_runtime.metadata["hush.config"])
        config["hop_size"] = 480
        fake_runtime.metadata["hush.config"] = json.dumps(config)
    elif invalid == "input_shape":
        fake_runtime.session.inputs[0].shape = [1, 1, 2, 32]
    elif invalid == "output_order":
        fake_runtime.session.outputs[:2] = fake_runtime.session.outputs[1::-1]
    else:
        fake_runtime.metadata["hush.states"] = json.dumps(fake_runtime.states[:-1])
    with pytest.raises(ValueError, match="Hush|streaming"):
        PythonModel(fake_runtime.path, 100)
    assert fake_runtime.session.calls == []


def test_python_rejects_nonfinite_snr(fake_runtime):
    model = PythonModel(fake_runtime.path, 100)
    fake_runtime.session.snr = np.nan
    with pytest.raises(RuntimeError, match="non-finite SNR"):
        model.process_frame(model.create_state(), np.zeros(160, dtype=np.float32))


@pytest.mark.parametrize("deep_filter", [False, True])
def test_dsp_identity_has_one_hop_delay_and_exact_tail_length(deep_filter):
    state = DspState()
    rng = np.random.default_rng(431)
    audio = rng.uniform(-.1, .1, size=8 * 160).astype(np.float32)
    audio[2 * 160:4 * 160] = 0
    mask = np.ones(32, dtype=np.float32)
    coefs = np.zeros((64, 5, 2), dtype=np.float32) if deep_filter else None
    if coefs is not None:
        coefs[:, -1, 0] = 1
    output = []
    for frame in np.r_[audio, np.zeros(160, dtype=np.float32)].reshape(-1, 160):
        features = state.analyze(frame)
        assert features["feat_erb"].shape == (1, 1, 1, 32)
        assert features["feat_spec"].shape == (1, 2, 1, 64)
        assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in features.values())
        output.append(state.synthesize(mask, coefs).copy())
    result = np.concatenate(output)
    np.testing.assert_allclose(result[:160], 0, atol=1e-6)
    np.testing.assert_allclose(result[160:], audio, atol=1e-6)
