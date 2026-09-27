"""Hermetic C-ABI integration tests; no Hush binary, model, or network required."""

import asyncio
import builtins
import ctypes
import importlib
import struct
import threading

import pytest

from aiavatar.sts.vad.filters.hush import HushAudioFilter


def test_public_hush_import_reexports_implementation():
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.hush")
    assert HushAudioFilter is module.HushAudioFilter


class _CFunction:
    """Allow the binding to attach ctypes argtypes and restype to fake symbols."""

    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def _value(value):
    return getattr(value, "value", value)


class _Native:
    """Identity processor with the upstream model's 160-sample algorithmic delay."""

    def __init__(self):
        self.sessions = {}
        self.freed_sessions = []
        self.freed_models = []
        self.threads = []
        self.loaded_paths = []
        self.next_session = 1
        self.load_started = threading.Event()
        self.load_release = threading.Event()
        self.load_release.set()
        self.process_started = threading.Event()
        self.process_release = threading.Event()
        self.process_release.set()
        self.fail_process = False
        self.process_result = 0.0
        self.nonfinite_output = False
        self.fail_create = False
        self.gain = 1.0
        self.lock = threading.RLock()
        self.active = {}
        self.frames = {}
        self.before_process = None
        self.after_write = None
        for name in (
            "model_load_from_path", "model_free", "session_create", "session_free",
            "get_frame_length", "get_sample_rate", "get_input_sample_rate",
            "process_frame", "reset",
        ):
            setattr(self, "weya_nc_" + name, _CFunction(getattr(self, name)))

    def _record(self):
        self.threads.append(threading.get_ident())

    def load(self, *args, **kwargs):
        self._record()
        self.load_started.set()
        assert self.load_release.wait(5), "Test did not release native library loading"
        return self

    def model_load_from_path(self, path):
        self._record()
        self.loaded_paths.append(_value(path))
        return 1000

    def model_free(self, model):
        self._record()
        assert not self.sessions, "Model freed before its sessions"
        assert not any(self.active.values()), "Model freed during inference"
        self.freed_models.append(_value(model))

    def session_create(self, model, sample_rate, atten_lim_db):
        self._record()
        assert _value(model) == 1000
        assert _value(sample_rate) == 16000
        if self.fail_create:
            return None
        with self.lock:
            handle = self.next_session
            self.next_session += 1
            self.sessions[handle] = [0.0] * 160
            self.active[handle] = 0
            self.frames[handle] = []
        return handle

    def session_free(self, session):
        self._record()
        handle = _value(session)
        with self.lock:
            assert handle in self.sessions, "Native session freed more than once"
            assert self.active[handle] == 0, "Native session freed during inference"
            del self.sessions[handle]
            self.freed_sessions.append(handle)

    def get_frame_length(self, session):
        self._record()
        return 160

    def get_sample_rate(self, session):
        self._record()
        return 16000

    def get_input_sample_rate(self, session):
        self._record()
        return 16000

    def process_frame(self, session, source, destination):
        self._record()
        handle = _value(session)
        with self.lock:
            assert self.active[handle] == 0, "Concurrent inference on the same native session"
            self.active[handle] += 1
        try:
            return self._process_frame(handle, source, destination)
        finally:
            with self.lock:
                self.active[handle] -= 1

    def _process_frame(self, handle, source, destination):
        self.process_started.set()
        if not self.process_release.wait(timeout=5):
            raise AssertionError("Test did not release blocked native inference")
        if self.fail_process:
            raise RuntimeError("fake native inference failed")
        source = ctypes.cast(source, ctypes.POINTER(ctypes.c_float))
        destination = ctypes.cast(destination, ctypes.POINTER(ctypes.c_float))
        incoming = [source[i] for i in range(160)]
        self.frames[handle].append(round(incoming[0] * 32768))
        if self.before_process:
            self.before_process(handle, incoming)
        if self.process_result == -100.0:
            for i, sample in enumerate(incoming):
                destination[i] = sample
            return self.process_result
        # The native runtime short-circuits very quiet frames without advancing
        # its delay line. Flush must still recover the preceding speech tail.
        if sum(value * value for value in incoming) / 160 < 1e-7:
            for i in range(160):
                destination[i] = 0.0
            return 0.0
        previous = self.sessions[handle]
        self.sessions[handle] = incoming
        for i, sample in enumerate(previous):
            destination[i] = sample * self.gain
        if self.nonfinite_output:
            destination[0] = float("nan")
        if self.after_write:
            self.after_write(handle, incoming)
        return self.process_result

    def reset(self, session):
        self._record()
        self.sessions[_value(session)] = [0.0] * 160


@pytest.fixture
def native(monkeypatch):
    instance = _Native()
    monkeypatch.setattr(ctypes, "CDLL", instance.load)
    yield instance
    instance.load_release.set()
    instance.process_release.set()


@pytest.fixture
def make_filter(tmp_path, native):
    model = tmp_path / "hush-model.tar.gz"
    library = tmp_path / "libweya_nc.so"
    model.touch()
    library.touch()
    instances = []

    def factory(**kwargs):
        params = {"model_path": model, "lib_path": library}
        params.update(kwargs)
        instance = HushAudioFilter(**params)
        instances.append(instance)
        return instance

    yield factory
    native.load_release.set()
    native.process_release.set()
    for instance in instances:
        instance.close()


def _pcm(values):
    return struct.pack("<" + "h" * len(values), *values)


async def _started(native):
    assert await asyncio.to_thread(native.process_started.wait, 2), "Inference did not start"


def test_rejects_unsupported_pcm_and_invalid_queue_limits(make_filter):
    for kwargs in (
        {"sample_rate": 48000}, {"channels": 2},
        {"max_pending_duration": 0}, {"max_pending_duration": -1},
        {"max_pending_duration": float("inf")},
        {"atten_lim_db": 0}, {"atten_lim_db": 101}, {"atten_lim_db": float("nan")},
        {"max_workers": 0}, {"max_workers": -1}, {"max_workers": 1.5},
    ):
        with pytest.raises(ValueError):
            make_filter(**kwargs)


def test_constructor_only_configures_without_starting_workers(make_filter, native, monkeypatch):
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.hush")

    def reject_executor(*args, **kwargs):
        pytest.fail("Constructing a Hush filter must not start loading or allocate workers")

    monkeypatch.setattr(module, "ThreadPoolExecutor", reject_executor)
    instance = make_filter(max_workers=2, atten_lim_db=36)
    assert instance.get_config()["max_workers"] == 2
    assert instance.get_config()["atten_lim_db"] == 36
    assert native.threads == []


def test_native_model_may_be_omitted_without_eager_io(make_filter, native, monkeypatch):
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.hush")

    def reject_resolution(*args, **kwargs):
        pytest.fail("Model resolution must wait for prepare, including automatic native models")

    monkeypatch.setattr(module, "resolve_model_path", reject_resolution)
    instance = make_filter(model_path=None)
    assert instance.get_config()["backend"] == "native"
    assert native.threads == []


@pytest.mark.parametrize("backend", ["invalid", "onnx", "PYTHON", None])
def test_invalid_backend_is_rejected(make_filter, backend):
    with pytest.raises(ValueError, match="backend"):
        make_filter(backend=backend)


@pytest.mark.parametrize("configuration", [{}, {"backend": "native"}])
def test_native_backend_can_omit_both_assets_without_eager_io(
    make_filter, native, monkeypatch, configuration,
):
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.hush")

    def reject_resolution(*args, **kwargs):
        pytest.fail("Automatic asset resolution must wait for prepare")

    monkeypatch.setattr(module, "resolve_library_path", reject_resolution)
    monkeypatch.setattr(module, "resolve_model_path", reject_resolution)
    instance = make_filter(lib_path=None, model_path=None, **configuration)
    assert instance.get_config()["backend"] == "native"
    assert native.threads == []


def test_python_backend_rejects_native_library(make_filter):
    with pytest.raises(ValueError, match="lib_path"):
        make_filter(backend="python")


@pytest.mark.asyncio
async def test_native_backend_rejects_converted_onnx_before_native_model_parse(
    make_filter, native, tmp_path,
):
    model = tmp_path / "streaming.onnx"
    model.touch()
    instance = make_filter(model_path=model)
    with pytest.raises(ValueError, match="original tar.gz bundle"):
        await instance.prepare()
    assert native.loaded_paths == []
    await instance.aclose()


@pytest.mark.asyncio
async def test_native_auto_model_resolution_uses_original_bundle_off_loop(
    make_filter, native, monkeypatch, tmp_path,
):
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.hush")
    model = tmp_path / "official-model.tar.gz"
    payload = b"original model archive: consumed only by the fake native runtime"
    model.write_bytes(payload)
    cache = tmp_path / "cache"
    resolutions = []
    loop_thread = threading.get_ident()

    def resolve(model_path, cache_dir):
        assert threading.get_ident() != loop_thread
        assert native.load_started.is_set(), "Validate the local native library before downloading"
        resolutions.append((model_path, cache_dir))
        return model

    monkeypatch.setattr(module, "resolve_model_path", resolve)
    instance = make_filter(model_path=None, cache_dir=cache)
    assert not resolutions
    assert native.threads == []
    await instance.prepare()
    await instance.prepare()
    assert resolutions == [(None, cache)]
    assert native.loaded_paths == [bytes(model)]
    assert model.read_bytes() == payload
    pcm = _pcm([1900] * 321)
    result = await instance.process_async(pcm, "native")
    result += await instance.flush_async("native")
    assert result == pcm


@pytest.mark.asyncio
async def test_native_auto_library_resolution_is_off_loop_and_prepare_waits(
    make_filter, native, monkeypatch, tmp_path,
):
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.hush")
    library = tmp_path / "downloaded-library.so"
    library.touch()
    cache = tmp_path / "cache"
    started, release = threading.Event(), threading.Event()
    resolutions = []
    loop_thread = threading.get_ident()

    def resolve(lib_path, cache_dir):
        assert threading.get_ident() != loop_thread
        resolutions.append((lib_path, cache_dir))
        started.set()
        assert release.wait(5), "Test did not release native library resolution"
        return library

    monkeypatch.setattr(module, "resolve_library_path", resolve)
    instance = make_filter(lib_path=None, cache_dir=cache)
    assert not resolutions
    preparing = asyncio.create_task(instance.prepare())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert not preparing.done()
        assert not native.load_started.is_set()
        with pytest.raises(RuntimeError, match="prepare"):
            instance.process(_pcm([1200] * 160), "speaker")
    finally:
        release.set()
        await asyncio.wait_for(preparing, 3)
    await instance.prepare()
    assert resolutions == [(None, cache)]
    pcm = _pcm([1900] * 321)
    result = await instance.process_async(pcm, "speaker")
    result += await instance.flush_async("speaker")
    assert result == pcm


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["platform", "checksum"])
async def test_native_library_resolution_failure_is_not_retried_or_bypassed(
    make_filter, native, monkeypatch, failure,
):
    module = importlib.import_module("aiavatar.sts.vad.filters.hush.hush")
    error = RuntimeError("Unsupported platform") if failure == "platform" else ValueError("SHA-256 mismatch")
    attempts = []

    def fail_resolution(*args):
        attempts.append(threading.get_ident())
        raise error

    monkeypatch.setattr(module, "resolve_library_path", fail_resolution)
    instance = make_filter(lib_path=None)
    for _ in range(2):
        with pytest.raises(type(error)) as caught:
            await instance.prepare()
        assert caught.value is error
    assert len(attempts) == 1
    assert threading.get_ident() not in attempts
    assert native.threads == []
    assert native.loaded_paths == []
    await instance.aclose()
    await instance.aclose()


@pytest.mark.asyncio
async def test_native_backend_never_imports_python_backend_or_onnx_dependencies(
    make_filter, monkeypatch,
):
    import_module = builtins.__import__
    forbidden = {"model_backend", "runtime", "dsp", "graph", "onnxruntime", "onnx"}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        requested = {name.rsplit(".", 1)[-1], *(fromlist or ())}
        assert not requested & forbidden, "Native processing imported the Python backend or ONNX dependencies"
        return import_module(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    async with make_filter() as instance:
        pcm = _pcm([2200] * 321)
        result = await instance.process_async(pcm, "native")
        result += await instance.flush_async("native")
        assert result == pcm


@pytest.mark.asyncio
@pytest.mark.parametrize("asset", ["model_path", "lib_path"])
async def test_asset_validation_is_deferred_until_prepare(make_filter, tmp_path, native, asset):
    instance = make_filter(**{asset: tmp_path / "missing-asset"})
    assert native.threads == []
    with pytest.raises(FileNotFoundError, match="Hush asset"):
        await instance.prepare()
    await instance.aclose()


@pytest.mark.asyncio
async def test_audio_requires_prepare_without_starting_model_load(make_filter, native):
    instance = make_filter()
    for audio in (b"", _pcm([1234] * 160)):
        with pytest.raises(RuntimeError, match="prepare"):
            instance.process(audio, "speaker")
        with pytest.raises(RuntimeError, match="prepare"):
            await instance.process_async(audio, "speaker")
    assert native.threads == []
    assert native.sessions == {}


@pytest.mark.asyncio
async def test_prepare_is_shared_and_waits_for_model_before_audio(make_filter, native):
    native.load_release.clear()
    instance = make_filter()
    preparing = [asyncio.create_task(instance.prepare()) for _ in range(4)]
    try:
        assert await asyncio.to_thread(native.load_started.wait, 2)
        assert all(not task.done() for task in preparing)
        preparing[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await preparing[0]
        with pytest.raises(RuntimeError, match="prepare"):
            instance.process(_pcm([1000] * 160), "speaker")
        with pytest.raises(RuntimeError, match="prepare"):
            await instance.process_async(_pcm([1000] * 160), "speaker")
        assert all(not task.done() for task in preparing[1:])
    finally:
        native.load_release.set()
        await asyncio.gather(*preparing, return_exceptions=True)
    # One cancelled waiter must not cancel model loading for its peers.
    assert all(task.exception() is None for task in preparing[1:])
    await instance.prepare()
    assert len(native.loaded_paths) == 1
    pcm = _pcm([2300] * 321)
    output = await instance.process_async(pcm, "speaker")
    output += await instance.flush_async("speaker")
    assert output == pcm


@pytest.mark.asyncio
async def test_close_before_prepare_does_not_load_and_prevents_reopening(make_filter, native):
    instance = make_filter()
    await instance.aclose()
    await instance.aclose()
    await asyncio.to_thread(instance.close)
    assert native.threads == []
    with pytest.raises(RuntimeError, match="closed"):
        await instance.prepare()


@pytest.mark.asyncio
async def test_cancelled_prepare_can_close_without_leaking_loaded_model(make_filter, native):
    native.load_release.clear()
    instance = make_filter()
    preparing = asyncio.create_task(instance.prepare())
    closing = None
    try:
        assert await asyncio.to_thread(native.load_started.wait, 2)
        preparing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await preparing
        closing = asyncio.create_task(instance.aclose())
        # Starting close must leave the loop responsive while loading finishes.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closing.done()
        assert native.freed_models == []
    finally:
        native.load_release.set()
        await asyncio.gather(preparing, return_exceptions=True)
        if closing is not None:
            await asyncio.wait_for(closing, 3)
    assert len(native.loaded_paths) == 1
    assert native.freed_models == [1000]
    await instance.aclose()
    assert native.freed_models == [1000]
    with pytest.raises(RuntimeError, match="closed"):
        await instance.prepare()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_workers", [1, 4])
async def test_native_calls_stay_off_event_loop_with_configured_workers(make_filter, native, max_workers):
    instance = make_filter(max_workers=max_workers)
    assert instance.get_config()["max_workers"] == max_workers
    await instance.prepare()
    output = instance.process(_pcm([4200] * 321), "speaker")
    output += await instance.flush_async("speaker")
    assert output == _pcm([4200] * 321)
    await instance.aclose()
    assert native.loaded_paths
    assert threading.get_ident() not in native.threads
    assert native.freed_models == [1000]


@pytest.mark.asyncio
async def test_pcm_endianness_extremes_and_odd_chunk_boundaries(make_filter):
    instance = make_filter()
    await instance.prepare()
    pcm = _pcm([-32768, -32767, -1, 0, 1, 32766, 32767] * 61)
    boundaries = [0, 1, 18, 319, 322, 649, 711, len(pcm)]
    output = b"".join(
        instance.process(pcm[start:end], "speaker")
        for start, end in zip(boundaries, boundaries[1:])
    )
    output += await instance.flush_async("speaker")
    assert output == pcm


@pytest.mark.asyncio
async def test_async_processing_conserves_fragmented_offline_audio(make_filter):
    instance = make_filter(max_pending_duration=0.04)
    await instance.prepare()
    pcm = _pcm([-3000 + index % 1000 for index in range(4097)])
    output = bytearray()
    # Longer than the queue limit overall, with sample boundaries split across
    # successive calls. Awaiting each chunk keeps offline processing bounded.
    for offset in range(0, len(pcm), 179):
        output.extend(await instance.process_async(pcm[offset:offset + 179], "offline"))
    assert output, "Async processing should emit available audio before EOF"
    output.extend(await instance.flush_async("offline"))
    assert bytes(output) == pcm


@pytest.mark.asyncio
async def test_flush_preserves_exact_length_for_short_and_partial_frames(make_filter, native):
    instance = make_filter()
    await instance.prepare()
    for count in (0, 1, 159, 160, 161, 319, 320, 321):
        pcm = _pcm([1000 + value for value in range(count)])
        session = str(count)
        output = instance.process(pcm, session)
        output += await instance.flush_async(session)
        assert output == pcm, f"Lost or added samples for length {count}"
        assert await instance.flush_async(session) == b""
    assert native.sessions == {}

    speech = _pcm([2000] * 160)
    output = instance.process(speech + bytes(320) + speech, "silence")
    output += await instance.flush_async("silence")
    assert len(output) == 960
    assert output[:320] == speech
    assert output[-320:] == speech
    assert max(abs(value) for value in struct.unpack("<160h", output[320:640])) <= 14


@pytest.mark.asyncio
async def test_out_of_range_model_output_is_saturated(make_filter, native):
    native.gain = 2.0
    instance = make_filter()
    await instance.prepare()
    pcm = _pcm([-32768, -20000, -1, 0, 1, 20000, 32767] * 50)
    expected = _pcm([-32768, -32768, -2, 0, 2, 32767, 32767] * 50)
    output = instance.process(pcm, "speaker")
    output += await instance.flush_async("speaker")
    assert output == expected


@pytest.mark.asyncio
async def test_interleaved_sessions_do_not_share_recurrent_audio(make_filter):
    instance = make_filter()
    await instance.prepare()
    first = [_pcm([value] * 193) for value in (1200, 1300, 1400)]
    second = [_pcm([value] * 257) for value in (-4200, -4300, -4400)]
    first_output = second_output = b""
    for left, right in zip(first, second):
        first_output += instance.process(left, "left")
        second_output += instance.process(right, "right")
    first_output += await instance.flush_async("left")
    second_output += await instance.flush_async("right")
    assert first_output == b"".join(first)
    assert second_output == b"".join(second)


@pytest.mark.asyncio
async def test_busy_session_does_not_starve_peer_or_share_output_buffer(make_filter, native):
    instance = make_filter(max_workers=2)
    await instance.prepare()
    blocked = threading.Event()
    release = threading.Event()

    def hold_first_session_output(handle, incoming):
        # Leave meaningful output in the native destination while the peer
        # runs. A shared destination buffer would corrupt this exact frame.
        if handle == 1 and len(native.frames[handle]) == 2:
            blocked.set()
            assert release.wait(5), "Test did not release the first session"

    native.after_write = hold_first_session_output
    first = _pcm([1000] * 320)
    second = _pcm([2000] * 320)
    first_output = instance.process(first, "busy")
    try:
        assert await asyncio.to_thread(blocked.wait, 2)
        first_output += instance.process(second, "busy")
        other = _pcm([-6000] * 480)
        other_output = await asyncio.wait_for(instance.process_async(other, "peer"), 2)
        other_output += await asyncio.wait_for(instance.flush_async("peer"), 2)
        assert other_output == other
        assert not release.is_set()
    finally:
        release.set()
    first_output += await instance.flush_async("busy")
    assert first_output == first + second
    assert native.frames[1][:4] == [1000, 1000, 2000, 2000]
    assert len(native.loaded_paths) == 1, "All worker sessions should share one loaded model"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["reset", "flush", "close"])
async def test_active_session_teardown_waits_for_its_queued_work(make_filter, native, operation):
    instance = make_filter(max_workers=2)
    await instance.prepare()
    blocked = threading.Event()
    release = threading.Event()
    finalizer = None

    def hold_first_session(handle, incoming):
        if handle == 1 and len(native.frames[handle]) == 1:
            blocked.set()
            assert release.wait(5), "Test did not release active inference"

    native.before_process = hold_first_session
    first = _pcm([1600] * 320)
    second = _pcm([2400] * 160)
    output = instance.process(first, "speaker")
    try:
        assert await asyncio.to_thread(blocked.wait, 2)
        output += instance.process(second, "speaker")
        if operation == "reset":
            instance.reset_session("speaker")
            # A new incarnation must be able to progress independently while
            # the old native handle is still in flight and awaiting cleanup.
            fresh = _pcm([-3000] * 161)
            fresh_output = await asyncio.wait_for(instance.process_async(fresh, "speaker"), 2)
            fresh_output += await asyncio.wait_for(instance.flush_async("speaker"), 2)
            assert fresh_output == fresh
        elif operation == "flush":
            finalizer = asyncio.create_task(instance.flush_async("speaker"))
        else:
            finalizer = asyncio.create_task(instance.aclose())
        await asyncio.sleep(0)
        assert 1 not in native.freed_sessions
        assert native.active[1] == 1
        assert not native.freed_models
        if finalizer:
            assert not finalizer.done()
    finally:
        release.set()
        if finalizer:
            result = await asyncio.wait_for(finalizer, 3)
            if operation == "flush":
                output += result
    if operation == "flush":
        assert output == first + second
    await instance.aclose()
    assert native.frames[1][:3] == [1600, 1600, 2400]
    assert native.sessions == {}
    assert native.freed_models == [1000]


@pytest.mark.asyncio
async def test_failed_session_does_not_block_peer_or_following_cleanup(make_filter, native):
    instance = make_filter(max_workers=2)
    await instance.prepare()
    blocked = threading.Event()
    release = threading.Event()

    def fail_first_session(handle, incoming):
        if handle == 1:
            blocked.set()
            assert release.wait(5), "Test did not release failing inference"
            raise RuntimeError("fake native inference failed")

    native.before_process = fail_first_session
    instance.process(_pcm([2000] * 160), "failing")
    try:
        assert await asyncio.to_thread(blocked.wait, 2)
        instance.process(_pcm([3000] * 160), "failing")
        peer = _pcm([-4000] * 321)
        peer_output = await asyncio.wait_for(instance.process_async(peer, "peer"), 2)
        peer_output += await asyncio.wait_for(instance.flush_async("peer"), 2)
        assert peer_output == peer
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="inference|failed"):
        await asyncio.wait_for(instance.flush_async("failing"), 2)
    instance.reset_session("failing")
    fresh = _pcm([5000] * 161)
    output = await instance.process_async(fresh, "failing")
    output += await instance.flush_async("failing")
    assert output == fresh
    await instance.aclose()
    assert native.sessions == {}


@pytest.mark.asyncio
async def test_process_and_reset_do_not_wait_for_inference(make_filter, native):
    instance = make_filter()
    await instance.prepare()
    native.process_release.clear()
    assert instance.process(_pcm([100] * 320), "speaker") == b""
    await _started(native)
    watchdog = threading.Timer(2, native.process_release.set)
    watchdog.start()
    try:
        assert instance.process(_pcm([200] * 160), "speaker") == b""
        instance.reset_session("speaker")
        assert not native.process_release.is_set(), "Synchronous filter call blocked on inference"
        fresh = _pcm([-700] * 481)
        output = instance.process(fresh, "speaker")
    finally:
        native.process_release.set()
        watchdog.cancel()
    output += await instance.flush_async("speaker")
    assert output == fresh
    await instance.aclose()
    assert native.sessions == {}
    assert len(native.freed_sessions) == 2


@pytest.mark.asyncio
async def test_backlog_limit_rejects_audio_without_passthrough(make_filter, native):
    instance = make_filter(max_pending_duration=0.02)
    await instance.prepare()
    native.process_release.clear()
    instance.process(_pcm([100] * 320), "speaker")
    await _started(native)
    try:
        with pytest.raises((BufferError, RuntimeError)):
            instance.process(_pcm([200] * 160), "speaker")
    finally:
        native.process_release.set()
    instance.reset_session("speaker")
    fresh = _pcm([-50] * 160)
    output = instance.process(fresh, "speaker")
    output += await instance.flush_async("speaker")
    assert output == fresh


@pytest.mark.asyncio
async def test_odd_final_byte_is_rejected_and_session_can_be_reused(make_filter, native):
    instance = make_filter()
    await instance.prepare()
    instance.process(b"\x01", "speaker")
    with pytest.raises(ValueError):
        await instance.flush_async("speaker")
    assert native.sessions == {}
    instance.reset_session("speaker")
    pcm = _pcm([4100] * 161)
    output = instance.process(pcm, "speaker")
    output += await instance.flush_async("speaker")
    assert output == pcm
    assert native.sessions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["exception", "fallback_sentinel", "nonfinite"])
async def test_native_inference_failure_is_reported_and_frees_session(make_filter, native, failure_mode):
    instance = make_filter()
    await instance.prepare()
    native.fail_process = failure_mode == "exception"
    native.process_result = -100.0 if failure_mode == "fallback_sentinel" else 0.0
    native.nonfinite_output = failure_mode == "nonfinite"
    instance.process(_pcm([400] * 160), "speaker")
    with pytest.raises(RuntimeError, match="inference|non-finite"):
        await instance.flush_async("speaker")
    await instance.aclose()
    assert native.sessions == {}
    assert native.freed_models == [1000]


@pytest.mark.asyncio
async def test_native_session_creation_failure_is_reported(make_filter, native):
    instance = make_filter()
    await instance.prepare()
    native.fail_create = True
    instance.process(_pcm([400] * 160), "speaker")
    with pytest.raises(RuntimeError):
        await instance.flush_async("speaker")
    await instance.aclose()
    assert native.sessions == {}
    assert native.freed_models == [1000]


@pytest.mark.asyncio
async def test_library_load_failure_does_not_silently_disable_filter(make_filter, monkeypatch):
    attempts = []
    load_error = OSError("unavailable native library")

    def fail_load(*args, **kwargs):
        attempts.append(threading.get_ident())
        raise load_error

    monkeypatch.setattr(ctypes, "CDLL", fail_load)
    instance = make_filter()
    for _ in range(2):
        with pytest.raises(OSError, match="lib_path") as error:
            await instance.prepare()
        assert error.value.__cause__ is load_error
        assert "python" in str(error.value)
    assert len(attempts) == 1, "Failed preparation must not silently retry downloads or loading"
    assert threading.get_ident() not in attempts
    await instance.aclose()
    await instance.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await instance.prepare()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_rejects_further_audio(make_filter, native):
    instance = make_filter()
    await instance.prepare()
    instance.process(_pcm([400] * 320), "left")
    instance.process(_pcm([600] * 320), "right")
    await instance.aclose()
    await instance.aclose()
    await asyncio.to_thread(instance.close)
    assert native.sessions == {}
    assert native.freed_models == [1000]
    assert len(native.freed_sessions) == native.next_session - 1
    with pytest.raises(RuntimeError):
        instance.process(_pcm([123]), "left")


@pytest.mark.asyncio
async def test_context_managers_release_native_resources(make_filter, native):
    async with make_filter() as instance:
        instance.process(_pcm([40] * 160), "async")
    assert native.sessions == {}
    assert len(native.freed_models) == 1

    def use_sync_context():
        with make_filter() as instance:
            output = instance.process(_pcm([2000] * 161), "sync")
            output += instance.flush("sync")
            assert output == _pcm([2000] * 161)

    await asyncio.to_thread(use_sync_context)
    assert native.sessions == {}
    assert len(native.freed_models) == 2


def test_runtime_configuration_cannot_be_mutated(make_filter):
    instance = make_filter()
    original = instance.get_config()
    assert original["backend"] == "native"
    assert instance.set_config({
        "sample_rate": 48000, "atten_lim_db": 0, "max_workers": 1,
        "backend": "python", "unknown": True,
    }) == {}
    assert instance.get_config() == original
