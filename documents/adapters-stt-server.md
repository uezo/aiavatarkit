# Speech recognition server

`StreamSpeechRecognitionServer` exposes voice activity detection and speech recognition over
a WebSocket, with no LLM and no speech synthesis. It is the pipeline's front half on its
own.

Reach for it when you want AIAvatarKit's recognition behaviour — streaming partial results,
pre-roll buffering, speaker gating, per-session engine switching — inside an application
that already has its own idea of what to do with the text.

## Setup

```python
import os
from fastapi import FastAPI
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.stt.azure import AzureSpeechRecognizer
from aiavatar.adapter.stt import StreamSpeechRecognitionServer

speech_recognizer = AzureSpeechRecognizer(
    azure_api_key=os.environ["AZURE_API_KEY"],
    azure_region=os.environ["AZURE_REGION"],
    language="ja-JP",
    alternative_languages=["en-US", "zh-CN"],
)

vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    silence_duration_threshold=0.5,
    segment_silence_threshold=0.05,   # Emit partial results after 0.05s of silence
    max_duration=30.0,
    preroll_buffer_count=10,
)

stt_server = StreamSpeechRecognitionServer(
    vad=vad,
    api_key=os.environ.get("STT_API_KEY"),   # Optional
)

app = FastAPI()
app.include_router(stt_server.get_websocket_router("/ws/stt"))
```

The router path defaults to `/ws/stt`. When `api_key` is set, clients must authenticate; the
key is passed as a WebSocket subprotocol rather than a query parameter, so it stays out of
access logs.

A runnable version of this, with a browser client, is in
[`examples/stt/`](../examples/stt/).

## Streaming versus batch detectors

The server adapts to whichever detector you give it.

- **With `SileroStreamSpeechDetector`** it recognises in segments while the user is still
  speaking, sending `partial` messages as it goes and a `final` message when the turn ends.
- **With a non-streaming detector** it buffers the utterance, recognises once the turn ends,
  and sends a single `final` message.

`segment_silence_threshold` controls how eagerly partial results are emitted. Lower values
give more responsive captions at the cost of more recognition calls.

## Protocol

Client messages (`STTRequest`):

| Field | Notes |
| --- | --- |
| `type` | `start`, `data`, or `stop` |
| `session_id` | Identifies the recognition session |
| `audio_data` | Base64-encoded raw signed 16-bit mono PCM, on `data` messages. The sample rate must match the VAD's `sample_rate`; do not include a WAV header. |
| `metadata` | Optional session metadata read from the `start` message and stored in `session_data.data["metadata"]`. It is not copied to responses. |

Server messages (`STTResponse`):

| Field | Notes |
| --- | --- |
| `type` | `connected`, `partial`, `final`, `voiced`, or `error` |
| `session_id` | Echoes the session |
| `text` | Recognised text |
| `is_final` | Whether this is the settled result for the turn |
| `metadata` | Optional server-generated details, such as `duration` on `final` messages. Request metadata is not copied here. |

`voiced` messages report that speech is present without carrying text, which is enough to
drive a level meter or a "listening" indicator.

## Callbacks

```python
@stt_server.on_connect
async def on_connect(request, session_data):
    print(f"Client connected: {request.session_id}")

@stt_server.on_disconnect
async def on_disconnect(session_data):
    print(f"Client disconnected: {session_data.id}")
```

`session_data.data` is a plain dict you can use to hang application state off a session for
its lifetime.

## Recording

The server accepts the same voice recorder as the pipeline, so you can build a dataset from
real traffic:

```python
stt_server = StreamSpeechRecognitionServer(
    vad=vad,
    voice_recorder_enabled=True,
    voice_recorder_dir="recorded_voices",
)
```

Pass a `voice_recorder` instance directly to store somewhere other than the local
filesystem. See [Pipeline](pipeline.md) for the recorder implementations.

## See also

- [Speech detector](vad.md) — detectors, thresholds, and pre-roll buffering
- [Speech-to-Text](stt.md) — recognizers, hooks, and speaker handling
- [Adapters](adapters.md) — the full conversational channels

---

[← Documentation index](../README.md#-documentation)
