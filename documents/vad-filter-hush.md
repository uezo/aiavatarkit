# Hush background-speech filter

`HushAudioFilter` suppresses background speech and ambient noise before passing the main speaker's voice to VAD, recording, and speech-to-text (STT).
No voice enrollment is required. It does not identify a specific person, so background voices may remain when they are about as loud as the main speaker.

## Quick start

Use the `native` backend with the official library and automatic downloads of the required files.
Supported platforms are **macOS on Apple Silicon, Linux x86_64, and Windows x86_64**.
Audio input must be **16 kHz / mono / signed 16-bit little-endian PCM**.

### 1. Add the filter to VAD

Create a `HushAudioFilter` and pass it to the VAD's `audio_filters` argument.

```python
from aiavatar.sts.vad.filters.hush import HushAudioFilter
from aiavatar.sts.vad.silero import SileroSpeechDetector

hush = HushAudioFilter()
vad = SileroSpeechDetector(sample_rate=16000, audio_filters=[hush])
```

You can also pass `audio_filters=[hush]` to `SileroStreamSpeechDetector` in the same way.

### 2. Prepare during server startup

Await `prepare()` before accepting audio, and release resources with `aclose()` when the server shuts down.
For FastAPI, use a lifespan handler as shown below. If your app already has one, add these calls to its startup and shutdown logic.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await hush.prepare()
        yield
    finally:
        await hush.aclose()

app = FastAPI(lifespan=lifespan)
```

On the first run, `prepare()` downloads and loads the official library and model.
The server does not start accepting requests until preparation is complete. An internet connection is required for the first startup.
Downloaded files are stored in `~/.cache/aiavatar/hush` and reused on subsequent runs.
To change the cache location, use `HushAudioFilter(cache_dir="/path/to/cache")`.

If a required system library is missing, loading may fail after the download completes.
Check the error details and install the required system dependencies, or use the Python backend described below.

## Use manually downloaded files

To use a library and model you have downloaded yourself, specify `lib_path` and `model_path`.
The following example is for Linux. Use the same preparation and shutdown steps as in the quick start.

```python
hush = HushAudioFilter(
    lib_path="/opt/hush/libweya_nc.so",
    model_path="/opt/hush/advanced_dfnet16k_model_best_onnx.tar.gz",
)
```

Download the official library for your OS and CPU architecture.
These links point to the same version used by automatic downloads.

| Platform | Library |
| --- | --- |
| macOS Apple Silicon | [libweya_nc.dylib](https://github.com/pulp-vision/Hush/blob/9f6414e91461a8f4bdf9840c0cdcdcb7da986339/deployment/lib/libweya_nc.dylib) |
| Linux x86_64 | [libweya_nc.so](https://github.com/pulp-vision/Hush/blob/9f6414e91461a8f4bdf9840c0cdcdcb7da986339/deployment/lib/libweya_nc.so) |
| Windows x86_64 | [weya_nc.dll](https://github.com/pulp-vision/Hush/blob/9f6414e91461a8f4bdf9840c0cdcdcb7da986339/deployment/lib/weya_nc.dll) |

Download [advanced_dfnet16k_model_best_onnx.tar.gz](https://huggingface.co/weya-ai/hush/resolve/40812c28145510d8a4b14641bb58c879a7a7b4fe/onnx/advanced_dfnet16k_model_best_onnx.tar.gz)
and pass its path as `model_path` without extracting the archive.

No download is performed for a file you specify. If you omit either path, that file is downloaded automatically.
Specify both paths to avoid downloads at startup.
On an OS or CPU architecture that does not support automatic downloads, you can still use `lib_path` if you have a compatible library.

## Use the Python backend

If the official library is unavailable for your environment, you can use the alternative implementation based on Python, NumPy, and ONNX Runtime.
Install it with the `onnx` package required for model conversion:

```sh
python -m pip install "aiavatar[hush-python]"
```

Specify `backend="python"` when creating the filter.

```python
hush = HushAudioFilter(backend="python")
```

Adding the filter to VAD and calling `prepare()` and `aclose()` work the same way as in the quick start.
On the first run, `prepare()` downloads the official model, converts it for execution, and loads it.
To use a local copy of the official model, specify `model_path`.
Do not specify `lib_path`.

### Convert the model in advance

To skip conversion at startup, convert a downloaded official model in an environment with
`aiavatar[hush-python]` installed.

```sh
python -m aiavatar.sts.vad.filters.hush.model_backend.graph \
  --input /path/to/advanced_dfnet16k_model_best_onnx.tar.gz \
  --output /path/to/hush-streaming.onnx
```

Place the generated file in your runtime environment and specify it with `model_path`.

```python
hush = HushAudioFilter(
    backend="python",
    model_path="/path/to/hush-streaming.onnx",
)
```

In this case, the runtime environment only needs `pip install aiavatar`.
Use a model generated by the command above. Individual ONNX files from the official archive cannot be used directly.

## Combine with NearFieldAudioGate

If background speech still triggers VAD when using Hush alone, try adding `NearFieldAudioGate`.
Use the order **NearFieldAudioGate → Hush → VAD**.
The gate evaluates the original audio level relative to ambient noise, then Hush suppresses background speech.

```python
from aiavatar.sts.vad.filters import NearFieldAudioGate
from aiavatar.sts.vad.filters.hush import HushAudioFilter
from aiavatar.sts.vad.silero import SileroSpeechDetector

near_field_gate = NearFieldAudioGate()
hush = HushAudioFilter()
vad = SileroSpeechDetector(
    sample_rate=16000,
    audio_filters=[near_field_gate, hush],
)
```

Use the same Hush preparation and shutdown steps as in the quick start.
The gate adds about 120 ms of latency by default and may also attenuate quiet speech or brief acknowledgments.
Hush cannot recover speech suppressed by the gate, so compare the results using real audio.
See [Audio Filters](vad-filters.md) for gate settings.

## Try a WAV file

The repository includes a [WAV conversion example](../examples/hush/filter_wav.py).
Get the source code and run the example from the repository root in an environment with AIAvatarKit installed.
The input must be a **16 kHz / mono / 16-bit PCM WAV file**. Choose an output path that does not already exist.

```sh
python -m examples.hush.filter_wav \
  --input /path/to/noisy-16k-mono.wav \
  --output /path/to/hush-output.wav
```

The required library and model are downloaded automatically. Use `--lib` and `--model` for local files,
and `--cache-dir` to change the cache location.
To try the Python backend, install `aiavatar[hush-python]` and specify `--backend python`.

Compare clips containing only background speech, overlapping speakers, quiet speech, and brief acknowledgments.
Check both how much background sound is suppressed and whether any of the main speaker's words are lost.

## Key settings

All of these are arguments to `HushAudioFilter(...)`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `backend` | `"native"` | `"native"` uses the official library; `"python"` uses the alternative implementation. |
| `lib_path` | `None` | Path to the native library. Downloaded automatically if omitted. |
| `model_path` | `None` | Path to the model. Downloaded automatically if omitted. |
| `cache_dir` | `None` | Location for downloaded files. Defaults to `~/.cache/aiavatar/hush`. |
| `atten_lim_db` | `100.0` | Maximum attenuation (0.01–100 dB). Lower values preserve more of the original audio. |
| `max_workers` | `4` | Maximum number of workers running inference concurrently. |
| `max_pending_duration` | `1.0` | Maximum duration of audio not yet returned per conversation, in seconds. Minimum: 0.02 seconds. |

Create a new filter to change its settings.

## How it works

### Backends and models

Both backends use the same pretrained Hush models in ONNX format.
With `native`, the official binary handles audio preprocessing, inference, and postprocessing.
With `python`, NumPy handles audio preprocessing and postprocessing, and ONNX Runtime runs inference.

The Python backend combines the three models in the official bundle into a single ONNX model
with conversation state exposed as inputs and outputs. By default, this conversion happens in memory during `prepare()`,
and the result is not saved. Creating a new filter therefore requires another conversion.
Use the conversion command described above to skip this step at startup.

The supported model is the official 16 kHz model (FFT=320, hop=160, lookahead=0).
Models with different frame sizes or lookahead settings are not supported.

### Preparation and shutdown

Creating a `HushAudioFilter` only stores its configuration. `prepare()` downloads and loads the assets and starts the workers.
Passing audio before or during preparation raises `RuntimeError`.
If preparation fails, close the filter with `aclose()`, resolve the cause, and create a new filter.

Automatic downloads use fixed versions of the library and model, with SHA256 verification before saving.
Cached files are also verified and reused without a network connection if valid.

You can use `async with hush:` to prepare the filter on entry and release resources on exit.
For synchronous programs, use `with hush:`.
`aclose()` shuts down the entire filter; a closed instance cannot be reused.

### Multiple conversations and workers

The model is shared, while audio processing history and buffers are kept separately for each conversation.
Audio within a conversation is processed in input order. Different conversations are processed in parallel, up to `max_workers`.
A conversation does not need a dedicated worker thread: its state is carried over when another worker processes it.

Downloads, model loading, and inference run in worker threads.
Awaiting `prepare()` waits for completion while allowing the asyncio event loop to continue running.

### Streaming and latency

The `process()` method called by VAD queues the input and returns audio that has already been processed.
It does not wait for inference to finish, so it may return empty audio, particularly on the first call.
Processed audio is returned on a subsequent call, so keep supplying input, including silence.
In addition to model processing, worker wait times and the handoff of roughly one input chunk add latency.

If the duration of audio not yet returned exceeds `max_pending_duration`, the filter raises `BufferError`.
Stop and reset the stream, then review the number of concurrent conversations and workers.

### Processing files and other finite audio streams

For WAV conversion and similar tasks, use `await hush.process_async(samples, session_id)`
to wait for a chunk to be processed and receive its output.
Await calls for the same conversation sequentially, and do not use this method concurrently with `process()`.

At the end, call `await hush.flush_async(session_id)` to receive the remaining audio and release the conversation state.
Concatenating all returned audio produces the same number of samples as the input.
Synchronous programs can use `flush(session_id)`.

VAD does not flush automatically. For live input, continue sending silence until VAD detects the end of speech,
then use `vad.delete_session(session_id)` on disconnect to discard any remaining audio and state.
When using the filter directly, `hush.reset_session(session_id)` discards them in the same way.

## License

Automatic downloads fetch the official native library and model from their respective official sources into the user's environment.
The native library is covered by [Hush's Apache-2.0 license](https://github.com/pulp-vision/Hush/blob/9f6414e91461a8f4bdf9840c0cdcdcb7da986339/LICENSE),
and the model is covered by [Apache-2.0 as stated in its model card](https://huggingface.co/weya-ai/hush#license).

The Python DSP implementation is adapted from Hendrik Schröter's DeepFilterNet as bundled with Hush,
using Apache-2.0 from the upstream license options.
The original copyright notice is `Copyright 2021 Hendrik Schröter`. See the
[dsp.py header](../aiavatar/sts/vad/filters/hush/model_backend/dsp.py) for attribution
and [LICENSE](../LICENSE) for the full license text.

## Related links

- [Hush repository](https://github.com/pulp-vision/Hush)
- [Hush model card](https://huggingface.co/weya-ai/hush)
