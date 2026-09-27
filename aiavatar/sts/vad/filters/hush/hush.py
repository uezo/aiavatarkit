"""Streaming Hush background-speech suppression with session-owned state.

Use the original native library, or explicitly select backend="python".
Importing this module never downloads models or loads either runtime. See
``documents/vad-filter-hush.md`` for setup and lifecycle examples.
"""

import asyncio
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from dataclasses import dataclass, field
import math
import os
import threading

import numpy as np

from ..base import AudioFilter
from .assets import resolve_library_path, resolve_model_path


_SAMPLE_RATE = 16000
_FRAME_SAMPLES = 160
_FRAME_BYTES = _FRAME_SAMPLES * 2
_DELAY_BYTES = _FRAME_BYTES  # FFT (320) - hop (160), zero-lookahead model.


@dataclass
class _AudioState:
    handle: object = None
    output: np.ndarray = field(default_factory=lambda: np.zeros(_FRAME_SAMPLES, dtype=np.float32))
    pending: bytearray = field(default_factory=bytearray)
    received: int = 0
    emitted: int = 0
    skip: int = _DELAY_BYTES
    error: BaseException | None = None


class _HushModel:
    """Shared model and PCM framing; mutable audio state belongs to a session."""

    def __init__(self, lib_path, model_path, atten_lim_db, cache_dir=None, backend="native"):
        self._python = None
        self._model = None
        self._clock_signal = np.tile(np.array([0.0004, -0.0004], dtype=np.float32), 80)
        if backend == "python":
            from .model_backend.runtime import PythonModel
            self._python = PythonModel(resolve_model_path(model_path, cache_dir), atten_lim_db)
            return
        library = resolve_library_path(lib_path, cache_dir)
        try:
            self._lib = ctypes.CDLL(str(library))
        except OSError as exc:
            raise OSError(
                f"Could not load Hush native library {library}: {exc}. "
                "Check OS dependencies, provide a compatible lib_path, or select backend='python'."
            ) from exc
        model = resolve_model_path(model_path, cache_dir)
        if model.suffix.lower() == ".onnx":
            raise ValueError("Native Hush requires the original tar.gz bundle; use backend='python' for streaming ONNX")
        pointer = ctypes.c_void_p
        float_pointer = ctypes.POINTER(ctypes.c_float)
        signatures = {
            "weya_nc_model_load_from_path": ([ctypes.c_char_p], pointer),
            "weya_nc_model_free": ([pointer], None),
            "weya_nc_session_create": ([pointer, ctypes.c_size_t, ctypes.c_float], pointer),
            "weya_nc_session_free": ([pointer], None),
            "weya_nc_get_frame_length": ([pointer], ctypes.c_size_t),
            "weya_nc_get_sample_rate": ([pointer], ctypes.c_size_t),
            "weya_nc_process_frame": ([pointer, float_pointer, float_pointer], ctypes.c_float),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes = argtypes
            function.restype = restype
        self._model = self._lib.weya_nc_model_load_from_path(os.fsencode(model))
        if not self._model:
            raise RuntimeError(f"Hush could not load the ONNX model bundle: {model}")
        self._atten_lim_db = atten_lim_db

    def _get_session(self, state):
        if state.error is not None:
            raise RuntimeError("Hush session failed; reset the session before reuse") from state.error
        if state.handle is None:
            if self._python is not None:
                state.handle = self._python.create_state()
                return state
            handle = self._lib.weya_nc_session_create(
                self._model, _SAMPLE_RATE, ctypes.c_float(self._atten_lim_db)
            )
            if not handle:
                raise RuntimeError("Hush could not create a native session")
            try:
                if (self._lib.weya_nc_get_frame_length(handle) != _FRAME_SAMPLES
                        or self._lib.weya_nc_get_sample_rate(handle) != _SAMPLE_RATE):
                    raise ValueError("Hush requires the 16 kHz / 160-sample-hop model")
            except BaseException:
                self._lib.weya_nc_session_free(handle)
                raise
            state.handle = handle
        return state

    def _frame(self, state, pcm):
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        # Hush's native runtime skips *all state updates* below this energy.
        # Advance STFT/OLA and recurrent state even during digital silence, or
        # a preceding word's tail can be lost or replayed after a long pause.
        # A tiny Nyquist signal (~-68 dBFS) clocks those frames. Align its sign
        # with the input to avoid cancellation; energy is then >= 1.6e-7.
        # Keep the same input in the Python backend for native-output parity.
        if float(np.mean(audio * audio)) < 1e-7:
            sign = 1.0 if float(np.dot(audio, self._clock_signal)) >= 0.0 else -1.0
            audio += sign * self._clock_signal
        state.output.fill(0.0)
        if self._python is not None:
            state.output[:] = self._python.process_frame(state.handle, audio)
        else:
            snr = self._lib.weya_nc_process_frame(
                state.handle,
                audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                state.output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            # The native ABI returns the original frame and this sentinel on
            # inference errors. Never silently pass through unfiltered audio.
            if snr == -100.0:
                raise RuntimeError("Hush native inference failed")
        if not np.isfinite(state.output).all():
            raise RuntimeError("Hush produced non-finite audio")
        result = np.clip(np.rint(state.output * 32768.0), -32768, 32767).astype("<i2").tobytes()
        skip = min(state.skip, len(result))
        state.skip -= skip
        return result[skip:]

    def process(self, state, samples):
        try:
            self._get_session(state)
            state.received += len(samples)
            state.pending.extend(samples)
            output = bytearray()
            while len(state.pending) >= _FRAME_BYTES:
                frame = bytes(state.pending[:_FRAME_BYTES])
                del state.pending[:_FRAME_BYTES]
                output.extend(self._frame(state, frame))
            state.emitted += len(output)
            return bytes(output)
        except BaseException as exc:
            self.close_session(state)
            state.error = exc
            raise

    def flush(self, state):
        try:
            if state.error is not None:
                raise RuntimeError("Hush session failed") from state.error
            if state.handle is None:
                return b""
            if state.received % 2:
                raise ValueError("Hush stream ends with an incomplete 16-bit PCM sample")
            remaining = state.received - state.emitted
            output = bytearray()
            while len(output) < remaining:
                frame = bytes(state.pending).ljust(_FRAME_BYTES, b"\x00")
                state.pending.clear()
                output.extend(self._frame(state, frame))
            return bytes(output[:remaining])
        finally:
            self.close_session(state)

    def close_session(self, state):
        if state.handle is not None:
            if self._python is None:
                self._lib.weya_nc_session_free(state.handle)
            state.handle = None

    def close(self):
        if self._model:
            self._lib.weya_nc_model_free(self._model)
            self._model = None
        self._python = None


@dataclass
class _Session:
    audio: _AudioState = field(default_factory=_AudioState)
    # The filter lock protects scheduling and output collection. Only this
    # session's running worker touches `audio`, in submission order.
    work: deque = field(default_factory=deque)
    running: bool = False
    jobs: deque = field(default_factory=deque)
    ready: bytearray = field(default_factory=bytearray)
    outstanding: int = 0
    error: Exception | None = None


class HushAudioFilter(AudioFilter):
    """Suppress background speech before VAD, using 16-bit LE mono PCM.

    ``process`` never waits for inference: output is collected on subsequent
    calls, including ``process(b"", session_id)``. Feed the microphone's silence
    frames continuously. Besides Hush's own latency, this adds scheduling and
    typically one input chunk of buffering. Call ``await prepare()`` before
    accepting audio, or enter the filter's sync/async context manager.

    Sessions share a worker pool, without fixed thread assignments. Each owns
    its inference state and FIFO; only one worker processes a session at a time.

    Only the official Hush 16 kHz, FFT=320, hop=160, zero-lookahead architecture
    is supported. The default ``backend="native"`` uses the original model
    bundle without extra Python dependencies. Without ``lib_path``, prepare
    downloads a pinned native library for macOS ARM64, Linux x86_64 or Windows
    x86_64. OS dependencies must be supplied by the caller if needed.
    Select ``backend="python"`` for the alternative Python/ONNX Runtime engine.
    Converting a bundle in that backend requires ``aiavatar[hush-python]``;
    a preconverted streaming ONNX needs only the standard dependencies.
    Without ``model_path``, prepare downloads the pinned official bundle into
    ``cache_dir`` (default: ~/.cache/aiavatar/hush) for either backend. Only the
    Python backend converts it in memory. Downloaded libraries use the same cache.
    Construction only records configuration; preparation does all model I/O.

    At the end of a finite stream, ``flush`` / ``flush_async`` returns its tail
    and frees its audio/model state. ``reset_session`` discards pending audio.
    Close the caller-owned filter with ``close`` / ``aclose`` after its detector
    stops accepting audio. Settings are immutable through ``set_config``.
    """

    def __init__(
        self, *, model_path=None, lib_path=None, backend: str = "native", cache_dir=None,
        sample_rate: int = 16000,
        channels: int = 1, atten_lim_db: float = 100.0,
        max_pending_duration: float = 1.0,
        max_workers: int = 4,
    ):
        if sample_rate != _SAMPLE_RATE or channels != 1:
            raise ValueError("HushAudioFilter requires 16000 Hz mono PCM")
        # Native 0 dB bypasses the STFT entirely, changing the stream delay.
        if not math.isfinite(atten_lim_db) or not 0.01 <= atten_lim_db <= 100:
            raise ValueError("atten_lim_db must be between 0.01 and 100")
        if not math.isfinite(max_pending_duration) or max_pending_duration < 0.02:
            raise ValueError("max_pending_duration must be at least 0.02 seconds")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if backend not in ("native", "python"):
            raise ValueError("backend must be 'native' or 'python'")
        if backend == "python" and lib_path is not None:
            raise ValueError("lib_path is only supported with backend='native'")
        self._atten_lim_db = float(atten_lim_db)
        self._max_pending_duration = float(max_pending_duration)
        self._max_pending_bytes = int(max_pending_duration * _SAMPLE_RATE) * 2
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._sessions = {}
        self._closed = False
        self._max_workers = max_workers
        self._backend = backend
        self._model_args = (lib_path, model_path, self._atten_lim_db, cache_dir, backend)
        self._executor = None
        self._runtime = None

    def get_config(self) -> dict:
        return {"backend": self._backend, "sample_rate": _SAMPLE_RATE, "channels": 1,
                "atten_lim_db": self._atten_lim_db,
                "max_pending_duration": self._max_pending_duration,
                "max_workers": self._max_workers}

    def set_config(self, config: dict) -> dict:
        # Changing native geometry or suppression mid-stream would leave live
        # sessions inconsistent. Construct a new filter for different settings.
        return {}

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("HushAudioFilter is closed")

    def _start_prepare(self):
        with self._lock:
            self._ensure_open()
            if self._runtime is None:
                executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="hush")
                try:
                    runtime = executor.submit(_HushModel, *self._model_args)
                except BaseException:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                self._executor, self._runtime = executor, runtime
            return self._runtime

    async def prepare(self):
        """Download, convert and load once, off the event loop; await before serving audio.

        Concurrent callers share the same preparation, including any failure.
        Cancelling a waiter leaves preparation running; aclose waits for it.
        """
        pending = asyncio.wrap_future(self._start_prepare())
        # Observe failures even if this waiter is cancelled during preparation.
        pending.add_done_callback(lambda future: None if future.cancelled() else future.exception())
        await asyncio.shield(pending)
        with self._lock:
            self._ensure_open()

    def _call(self, method, audio, *args):
        return getattr(self._runtime.result(), method)(audio, *args)

    def _submit(self, state, operation, *args):
        """Enqueue under the filter lock; at most one pool job per session."""
        result = Future()
        state.work.append((result, operation, args))
        if not state.running:
            state.running = True
            self._executor.submit(self._drain, state)
        return result

    def _drain(self, state):
        while True:
            with self._lock:
                if not state.work:
                    state.running = False
                    return
                result, operation, args = state.work.popleft()
            if result.set_running_or_notify_cancel():
                try:
                    result.set_result(operation(*args))
                except BaseException as exc:
                    # Keep draining: reset/flush/close must run after failures.
                    result.set_exception(exc)

    @staticmethod
    def _collect(state):
        if state.error is not None:
            raise RuntimeError("Hush session failed; call reset_session before reuse") from state.error
        while state.jobs and state.jobs[0].done():
            job = state.jobs.popleft()
            try:
                state.ready.extend(job.result())
            except Exception as exc:
                state.error = exc
                raise RuntimeError("Hush inference failed; audio was not passed through") from exc

    def process(self, samples: bytes, session_id: str) -> bytes:
        if not isinstance(samples, (bytes, bytearray, memoryview)):
            raise TypeError("samples must contain raw 16-bit PCM bytes")
        samples = bytes(samples)
        with self._lock:
            self._ensure_open()
            if self._runtime is None or not self._runtime.done():
                raise RuntimeError("Hush is not ready; await prepare() before processing audio")
            self._runtime.result()
            state = self._sessions.get(session_id)
            if state is None:
                if not samples:
                    return b""
                state = self._sessions[session_id] = _Session()
            self._collect(state)
            # Limit catch-up output: Silero evaluates only the last VAD frame
            # of each call, so avoid releasing a long backlog in one burst.
            limit = max(_FRAME_BYTES, len(samples) // 2 * 2) if samples else len(state.ready)
            available = min(limit, len(state.ready))
            if state.outstanding + len(samples) - available > self._max_pending_bytes:
                raise BufferError("Hush audio backlog exceeded max_pending_duration; reset the session")
            if samples:
                state.jobs.append(self._submit(state, self._call, "process", state.audio, samples))
                state.outstanding += len(samples)
            result = bytes(state.ready[:available])
            del state.ready[:available]
            state.outstanding -= len(result)
            return result

    async def process_async(self, samples: bytes, session_id: str) -> bytes:
        """Await this chunk's inference, for offline or async preprocessing.

        Calls for one session must be sequential. The synchronous VAD filter
        chain should keep using ``process``; do not mix the two entry points
        concurrently for the same stream.
        """
        with self._lock:
            result = self.process(samples, session_id)
            state = self._sessions.get(session_id)
            last_job = state.jobs[-1] if state is not None and state.jobs else None
        if last_job is not None:
            await asyncio.shield(asyncio.wrap_future(last_job))
        with self._lock:
            self._ensure_open()
            if self._sessions.get(session_id) is not state:
                raise RuntimeError("Hush session was reset while awaiting inference")
            if state is not None:
                self._collect(state)
                result += bytes(state.ready)
                state.outstanding -= len(state.ready)
                state.ready.clear()
            return result

    def reset_session(self, session_id: str):
        """Discard a stream; state cleanup is ordered after its queued work."""
        with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is not None and not self._closed:
                self._submit(state, self._call, "close_session", state.audio)

    def _finish(self, state):
        runtime = self._runtime.result()
        try:
            if state.error is not None:
                raise RuntimeError("Hush session failed") from state.error
            return bytes(state.ready) + b"".join(job.result() for job in state.jobs) + runtime.flush(state.audio)
        finally:
            runtime.close_session(state.audio)

    def _start_flush(self, session_id):
        with self._lock:
            self._ensure_open()
            state = self._sessions.pop(session_id, None)
            if state is not None:
                return self._submit(state, self._finish, state)
            result = Future()
            result.set_result(b"")
            return result

    @staticmethod
    def _require_sync(operation):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(f"Use the async Hush {operation} method inside an event loop")

    def flush(self, session_id: str) -> bytes:
        """Drain an offline stream, preserving its exact sample count."""
        self._require_sync("flush_async")
        return self._start_flush(session_id).result()

    async def flush_async(self, session_id: str) -> bytes:
        """Drain a stream without blocking. Cancellation still frees its state."""
        return await asyncio.shield(asyncio.wrap_future(self._start_flush(session_id)))

    def close(self):
        """Discard pending output and release all resources (offline only)."""
        self._require_sync("aclose")
        with self._close_lock:
            if self._closed:
                return
            with self._lock:
                self._closed = True
                for state in self._sessions.values():
                    self._submit(state, self._call, "close_session", state.audio)
                self._sessions.clear()
            # Drain all accepted work, including reset/flush of old sessions,
            # before freeing their shared model. Never wait under _lock.
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            if self._runtime is not None and self._runtime.exception() is None:
                self._runtime.result().close()

    async def aclose(self):
        """Release resources off the event loop; safe to call repeatedly."""
        await asyncio.shield(asyncio.to_thread(self.close))

    def __enter__(self):
        self._require_sync("prepare")
        try:
            self._start_prepare().result()
            with self._lock:
                self._ensure_open()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        try:
            await self.prepare()
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *args):
        await self.aclose()
