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

**NOTE:** When you specify `response_audio_chunk_size` in the `AIAvatarWebSocketServer` instance, the audio response will be streamed as PCM data chunks of the specified byte size. In this case, no WAVE header will be included in the response - you'll receive raw PCM audio data only.


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
