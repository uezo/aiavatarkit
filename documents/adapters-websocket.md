# WebSocket adapter

`AIAvatarWebSocketServer` accepts streaming microphone audio and runs voice activity
detection on the server. It is the lowest-latency channel and the one the built-in
application and browser examples use.

## Setup

Below is the simplest example of a server program:

```python
from fastapi import FastAPI
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer

# Create AIAvatar
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=OPENAI_API_KEY,
    volume_db_threshold=-30,  # <- Adjust for your audio env
    debug=True
)

# Set router to FastAPI app
app = FastAPI()
router = aiavatar_app.get_websocket_router()
app.include_router(router)
```

Save the above code as `server.py` and run it using:

```sh
uvicorn server:app
```

Next is the simplest example of a Python client program. This client uses local microphone and speaker devices, so install the optional local audio dependencies first:

```sh
pip install "aiavatar[local-audio]"
```

```python
import asyncio
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

client = AIAvatarWebSocketClient()
asyncio.run(client.start_listening(session_id="ws_session", user_id="ws_user"))
```

Save the above code as `client.py` and run it using:

```sh
python client.py
```

You can now perform voice interactions just like when running locally.

**NOTE:** When using the WebSocket API, voice activity detection (VAD) is performed on the server side, so clients can simply stream microphone input directly to the server.


## Audio delivery

`response_audio_chunk_size=0` (the default) sends each complete WAV as base64
`audio_data`. A positive value sends raw PCM in chunks of that many **bytes**:

```python
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=OPENAI_API_KEY,
    response_audio_chunk_size=4096,
)
```

Choose a size divisible by `channels * sample_width`, where `sample_width` is in
bytes. For 16-bit mono PCM, use a multiple of 2; for 16-bit stereo, a multiple of
4. This is a configuration requirement, without runtime validation. The last
chunk may be shorter but still contains whole PCM frames.

For each original WAV, the server first sends a `type: "chunk"` descriptor with
`audio_data: null`. It preserves the response's text, `voice_text`, controls, and
metadata, and adds:

```json
{
  "audio_id": "unique-id-for-this-audio",
  "audio_frame_count": 48000,
  "pcm_format": {
    "sample_rate": 24000,
    "channels": 1,
    "sample_width": 2
  }
}
```

These fields are inside `metadata`. One PCM frame contains one sample per
channel, so this example represents two seconds. Each following audio
`chunk` carries base64 raw PCM in `audio_data` and the same `metadata.audio_id`
and `metadata.pcm_format`. The client associates the PCM with its descriptor by
ID and determines receipt completion from the total frame count. WebSocket
preserves delivery order; no sequence numbers, acknowledgments, or retransmission
messages are needed.

The standard `NodPipelineBridge` uses this same delivery mode. In PCM mode, its
descriptor carries `metadata.nod=true` and the Nod metadata; its audio messages
contain PCM, allowing PCM-only clients to play acknowledgments too.

The maintained `index.html` and `3d.html` clients support signed 16-bit
little-endian PCM with the declared sample rate and channel count. Use the
updated server and browser together for this descriptor protocol. They schedule
each arriving PCM chunk for playback without waiting for the complete audio to
arrive. The server still synthesizes a complete WAV before splitting it, so the
total length is known before transmission. See the
[browser playback details](../examples/websocket/README.md#pcm-audio-playback).

With `debug=True`, the server logs one INFO summary per original audio, including
Nod, prefixed `WebSocket audio:` with its format, IDs, PCM format, byte/chunk sizes, and `nod` flag.
For PCM, `audio_id` matches the browser's `playback_id`; `planned_chunks` excludes
the descriptor and counts planned chunks, so interruption can reduce the number actually sent.
WAV reports `audio_id=None` and `planned_chunks=1`; its browser playback ID is generated later.
`debug=False` emits no audio summaries.


## Response transaction IDs

WebSocket responses expose the pipeline's `transaction_id` as an optional
top-level field. Events for one response, including `start`, audio `chunk`, and
`final`, carry the same ID. Both complete-WAV and split-PCM audio preserve it.
When a new transaction interrupts the previous one, the synthetic interrupted
`final` carries the old transaction's ID and `accepted` carries the new one.

Session controls such as `connected`, `voiced`, and the session-wide `stop`
notification may have no transaction ID. Clients also accept older responses
without this field. This ID identifies an AI response; it is distinct from the
browser's per-audio `playback_id` and PCM `audio_id`. The shared browser returns
the transaction ID in playback
start notifications for [turn-taking context](../examples/websocket/README.md#playback-context-for-turn-taking).

## Connection and disconnection handling

You can register callbacks to handle WebSocket connection and disconnection events. This is useful for logging, session management, or custom initialization/cleanup logic.

```python
import time

@aiavatar_app.on_connect
async def on_connect(request, session_data):
    # The identifiers live on the request, not on session_data
    print(f"Client connected: session={request.session_id} user={request.user_id}")

    # session_data.data is yours; use it to carry state into on_disconnect
    session_data.data["connected_at"] = time.time()

    # Custom initialization logic
    # e.g., load user preferences, initialize resources, etc.

@aiavatar_app.on_disconnect
async def on_disconnect(session_data):
    print(f"Client disconnected: {session_data.id}")

    # Custom cleanup logic
    # e.g., save session data, release resources, etc.
```

`WebSocketSessionData` carries:

| Attribute | Contents |
| --- | --- |
| `id` | The session id, assigned when the session opens |
| `data` | An empty dict, yours to use |
| `active_transaction_id` | The turn currently in flight, or `None` |

There is no `user_id` or `session_id` attribute on it. `on_connect` receives the
`AIAvatarRequest` that opened the session, so read them from there — and copy anything
`on_disconnect` will need into `session_data.data` while the session is opening, since that
callback receives only `session_data`.

## Attaching to an existing pipeline

The examples above let the adapter build its own pipeline. Pass `sts=` instead to attach to
one that already exists, which is how the WebSocket channel shares a conversation with the
phone or LINE. The default channel name is `websocket`.

```python
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer

websocket_adapter = AIAvatarWebSocketServer(
    sts=sts,
    channel="websocket",
    api_key="YOUR_WEBSOCKET_API_KEY",  # Optional
)
app.include_router(websocket_adapter.get_websocket_router(path="/ws"))
```

See [Adapters](adapters.md) for connecting several channels to one pipeline.

## See also

- [Adapters](adapters.md) — choosing a channel and sharing a pipeline
- [Speech detector](vad.md) — server-side detection
- [Avatar control](avatar.md) — the control tags the viewers render

---

[← Documentation index](../README.md#-documentation)
