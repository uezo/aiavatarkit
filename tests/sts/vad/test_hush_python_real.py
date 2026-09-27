"""Opt-in, local Hush Python-backend regressions; no downloads or asset discovery.

Set HUSH_TEST_MODEL_BUNDLE to the supported 16 kHz model tar.gz to run the
Python backend checks. Set HUSH_TEST_NATIVE_LIBRARY as well to compare the native backend.
Prerequisites: numpy, onnx, onnxruntime, pytest and pytest-asyncio installed.

    HUSH_TEST_MODEL_BUNDLE=/explicit/model.tar.gz \
    HUSH_TEST_NATIVE_LIBRARY=/explicit/libweya_nc.dylib \
    python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
        tests/sts/vad/test_hush_python_real.py -q

Only the explicitly supplied local assets and synthetic audio are used. Any
converted model belongs to pytest's temporary directory. No credentials,
microphone, application configuration, model cache search, or network is used.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tarfile
import threading

import httpx
import pytest

if not os.environ.get("HUSH_TEST_MODEL_BUNDLE"):
    pytest.skip("Set HUSH_TEST_MODEL_BUNDLE for local real-model checks", allow_module_level=True)

np = pytest.importorskip("numpy")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from aiavatar.sts.vad.filters.hush.model_backend.graph import build_streaming_model
from aiavatar.sts.vad.filters.hush.model_backend.runtime import PythonModel
from aiavatar.sts.vad.filters.hush import HushAudioFilter


def _session(data):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    bundle = Path(os.environ["HUSH_TEST_MODEL_BUNDLE"]).expanduser()
    assert bundle.is_file(), "HUSH_TEST_MODEL_BUNDLE must name an existing local file"
    compiled = tmp_path_factory.mktemp("hush-real") / "streaming.onnx"
    compiled.write_bytes(build_streaming_model(bundle))
    return bundle, compiled


@pytest.fixture(scope="module")
def graphs(assets):
    bundle, compiled = assets
    with tarfile.open(bundle) as archive:
        original = tuple(_session(archive.extractfile(name).read())
                         for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx"))
    streaming = _session(str(compiled))
    states = json.loads(streaming.get_modelmeta().custom_metadata_map["hush.states"])
    return original, streaming, states


def _features(length, seed):
    rng = np.random.default_rng(seed)
    return {"feat_erb": rng.normal(0, .3, (1, 1, length, 32)).astype(np.float32),
            "feat_spec": rng.normal(0, .04, (1, 2, length, 64)).astype(np.float32)}


def _full_sequence(models, features):
    enc, erb, df = models
    e0, e1, e2, e3, emb, c0, lsnr = enc.run(None, features)
    mask = erb.run(None, {"e0": e0, "e1": e1, "e2": e2, "e3": e3, "emb": emb})[0]
    coefs = df.run(None, {"emb": emb, "c0": c0})[0]
    return mask, coefs, lsnr


def _frame_sequence(model, specs, features, reset_each_frame=False):
    state = {s["input"]: np.zeros(s["shape"], dtype=np.float32) for s in specs}
    frames = []
    for index in range(features["feat_erb"].shape[2]):
        if reset_each_frame:
            for value in state.values():
                value.fill(0)
        inputs = {k: np.ascontiguousarray(v[:, :, index:index + 1]) for k, v in features.items()}
        mask, coefs, lsnr, *next_states = model.run(None, {**inputs, **state})
        frames.append((mask, coefs, lsnr))
        state = {s["input"]: value for s, value in zip(specs, next_states)}
    return tuple(np.concatenate([frame[i] for frame in frames], axis=axis)
                 for i, axis in enumerate((2, 1, 1)))


@pytest.mark.parametrize("length", [1, 7, 32])
def test_streaming_states_match_original_full_sequence(graphs, length):
    original, streaming, specs = graphs
    features = _features(length, seed=47)
    reference = _full_sequence(original, features)
    observed = _frame_sequence(streaming, specs, features)
    for expected, actual in zip(reference, observed):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-4)
    if length == 32:
        # Negative control: the test must detect the previous history-free
        # implementation, which started every hop with zero recurrent/cache state.
        reset = _frame_sequence(streaming, specs, features, reset_each_frame=True)
        assert max(float(np.max(np.abs(a - b))) for a, b in zip(reference, reset)) > 1e-3


def _audio(frames, seed):
    t = np.arange(frames * 160, dtype=np.float64) / 16000
    rng = np.random.default_rng(seed)
    return (.1 * np.sin(2 * np.pi * (137 + seed) * t)
            + .035 * np.sin(2 * np.pi * (411 + seed) * t)
            + rng.normal(0, .003, size=t.size)).astype(np.float32).reshape(frames, 160)


def test_model_state_isolation_and_sequential_worker_migration(assets):
    model = PythonModel(assets[1], atten_lim_db=100)
    audio = [_audio(40, 5), _audio(40, 53)]
    expected = []
    for samples in audio:
        state = model.create_state()
        expected.append(np.stack([model.process_frame(state, frame).copy() for frame in samples]))
    states = [model.create_state(), model.create_state()]
    outputs, thread_ids = [[], []], [set(), set()]

    def process(state, frame):
        return threading.get_ident(), model.process_frame(state, frame).copy()

    # Keep both one-thread executors alive: alternating submissions necessarily
    # move the same state between distinct threads, with no concurrent use of it.
    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        workers = (first, second)
        for frame_index in range(len(audio[0])):
            for session_id in range(2):
                worker = workers[(frame_index + session_id) % 2]
                ident, result = worker.submit(process, states[session_id], audio[session_id][frame_index]).result()
                outputs[session_id].append(result)
                thread_ids[session_id].add(ident)
    for session_id in range(2):
        assert len(thread_ids[session_id]) == 2
        np.testing.assert_array_equal(np.stack(outputs[session_id]), expected[session_id])


async def _fragmented_pcm(filter_instance, pcm, sizes):
    output, offset, index = bytearray(), 0, 0
    while offset < len(pcm):
        size = sizes[index % len(sizes)]
        output.extend(await filter_instance.process_async(pcm[offset:offset + size], "speaker"))
        offset += size
        index += 1
    output.extend(await filter_instance.flush_async("speaker"))
    assert await filter_instance.flush_async("speaker") == b""
    assert len(output) == len(pcm)
    return np.frombuffer(output, dtype="<i2").astype(np.int32)


@pytest.mark.asyncio
@pytest.mark.parametrize("atten_lim_db", [100, 20])
async def test_fragmented_public_filter_matches_native_pcm(assets, atten_lim_db):
    library_name = os.environ.get("HUSH_TEST_NATIVE_LIBRARY")
    if not library_name:
        pytest.skip("Set HUSH_TEST_NATIVE_LIBRARY to compare local native and Python backends")
    library = Path(library_name).expanduser()
    assert library.is_file(), "HUSH_TEST_NATIVE_LIBRARY must name an existing local file"
    bundle, compiled = assets
    audio = _audio(301, seed=17).ravel()[:48073]  # 3 seconds plus a partial final hop.
    audio[:3200] = audio[12000:22000] = audio[-1600:] = 0
    pcm = np.rint(audio * 32768).astype("<i2").tobytes()
    async with HushAudioFilter(model_path=bundle, lib_path=library, atten_lim_db=atten_lim_db,
                               max_workers=2) as native:
        expected = await _fragmented_pcm(native, pcm, (1, 137, 320, 997, 64, 511))
    async with HushAudioFilter(backend="python", model_path=compiled, atten_lim_db=atten_lim_db,
                               max_workers=2) as python_filter:
        actual = await _fragmented_pcm(python_filter, pcm, (631, 2, 17, 1600, 83))
    # FFT/backend summation order can differ while still rounding to adjacent
    # int16 samples. Subtraction uses int32 to avoid wraparound in this check.
    assert int(np.max(np.abs(actual - expected))) <= 2


@pytest.mark.asyncio
async def test_python_filter_downloads_once_then_reuses_bundle(assets, tmp_path, monkeypatch):
    """Use the real local model as a mocked download; no live network request."""
    from aiavatar.sts.vad.filters.hush import assets as model_assets

    bundle, compiled = assets
    payload = bundle.read_bytes()
    cache = tmp_path / "cache"
    requests = []
    loop_thread = threading.get_ident()

    @contextmanager
    def download(method, url, **kwargs):
        assert threading.get_ident() != loop_thread
        requests.append(url)
        response = httpx.Response(200, content=payload, request=httpx.Request(method, url))
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(model_assets.httpx, "stream", download)
    pcm = np.rint(_audio(23, seed=38).ravel() * 32768).astype("<i2").tobytes()
    async with HushAudioFilter(backend="python", model_path=compiled) as reference:
        expected = await _fragmented_pcm(reference, pcm, (320, 97))

    for index in range(2):
        hush = HushAudioFilter(backend="python", cache_dir=cache)
        if index == 0:
            assert not cache.exists()
            assert not requests
        async with hush:
            actual = await _fragmented_pcm(hush, pcm, (7, 512, 111))
            np.testing.assert_array_equal(actual, expected)
    assert requests == [model_assets.MODEL_URL]
    assert [p.suffixes[-2:] for p in cache.iterdir()] == [[".tar", ".gz"]]
