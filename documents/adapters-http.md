# HTTP (SSE) adapter

`AIAvatarHttpServer` exposes the pipeline over HTTP with Server-Sent Events. It also serves
a Dify-compatible endpoint and standalone speech recognition and synthesis endpoints, which
are useful even when you are not using the conversational API at all.

This adapter needs two packages that the base install does not pull in:

```sh
pip install sse-starlette python-multipart
```

`sse-starlette` provides the SSE streaming responses — without it, importing the server
module fails. `python-multipart` is what lets FastAPI accept the file upload on
`/transcribe`.

Below is the simplest example of a server program:

```python
from fastapi import FastAPI
from aiavatar.adapter.http.server import AIAvatarHttpServer

# AIAvatar
aiavatar_app = AIAvatarHttpServer(
    openai_api_key=OPENAI_API_KEY,
    debug=True
)

# Setup FastAPI app with AIAvatar components 
app = FastAPI()
router = aiavatar_app.get_api_router()
app.include_router(router)
```

Save the above code as `server.py` and run it using:

```sh
uvicorn server:app
```


Next is the simplest example of a client program. It captures from the microphone and plays
through the speaker, so it needs the local audio extra:

```sh
pip install "aiavatar[local-audio]"
```


```python
import asyncio
from aiavatar.adapter.http.client import AIAvatarHttpClient

client = AIAvatarHttpClient(
    debug=True
)
asyncio.run(client.start_listening(session_id="http_session", user_id="http_user"))
```

Save the above code as `client.py` and run it using:

```sh
python client.py
```

You can now perform voice interactions just like when running locally.


When using the streaming API via HTTP, clients communicate with the server using JSON-formatted requests.

Below is the format for initiating a session:

```json
{
    "type": "start",          // Always `start`
    "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d",
    "user_id": "user_id",
    "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7",   // Set null or provided id in `start` response
    "text": "こんにちは",       // If set, audio_data will be ignored         
    "audio_data": "XXXX",     // Base64 encoded audio data
    "files": [
        {
            "type": "image",        // Only `image` is supported for now
            "url": "https://xxx",
        }
    ],
    "metadata": {}
}
```

The server returns responses as a stream of JSON objects in the following structure.

The communication flow typically consists of:

```json
{
    "type": "chunk",    // start -> chunk -> final
    "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d",
    "user_id": "user01",
    "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7",
    "text": "<face name=\"joy\" />こんにちは！",   // Response text with info
    "voice_text": "こんにちは！",       // Response text for voice synthesis
    "avatar_control_request": {
        "animation_name": null,       // Parsed animation name
        "animation_duration": null,   // Parsed duration for animation
        "face_name": "joy",           // Parsed facial expression name
        "face_duration": 4.0          // Parsed duration for the facial expression
    },
    "audio_data": "XXXX",   // Base64 encoded. Playback this as the character's voice.
    "metadata": {
        "is_first_chunk": true
    }
}
```


You can test the streaming API using a simple curl command:

```sh
curl -N -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
        "type": "start",
        "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d",
        "user_id": "user01",
        "text": "こんにちは"
    }'

```

Sample response (streamed from the server):

```sh
data: {"type": "start", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": null, "voice_text": null, "avatar_control_request": null, "audio_data": "XXXX", "metadata": {"request_text": "こんにちは"}}

data: {"type": "chunk", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "<face name=\"joy\" />こんにちは！", "voice_text": "こんにちは！", "avatar_control_request": {"animation_name": null, "animation_duration": null, "face_name": "joy", "face_duration": 4.0}, "audio_data": "XXXX", "metadata": {"is_first_chunk": true}}

data: {"type": "chunk", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "今日はどんなことをお手伝いしましょうか？", "voice_text": "今日はどんなことをお手伝いしましょうか？", "avatar_control_request": {"animation_name": null, "animation_duration": null, "face_name": null, "face_duration": null}, "audio_data": "XXXX", "metadata": {"is_first_chunk": false}}

data: {"type": "final", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "<face name=\"joy\" />こんにちは！今日はどんなことをお手伝いしましょうか？", "voice_text": "こんにちは！今日はどんなことをお手伝いしましょうか？", "avatar_control_request": null, "audio_data": "XXXX", "metadata": {}}
```

To continue the conversation, include the `context_id` provided in the `start` response in your next request.

**NOTE:** When using the RESTful API, voice activity detection (VAD) must be performed client-side.

**NOTE:** To protect API with API Key, set `api_key=API_KEY_YOU_MAKE` to AIAvatarHttpServer and send `Authorization: Bearer {API_KEY_YOU_MAKE}` as HTTP header from client.

## Dify-compatible API

`AIAvatarHttpServer` provides a Dify-compatible `/chat-messages` endpoint (SSE streaming only).
This allows you to connect frontend applications that use Dify as their backend.

For more details, refer to the [Dify API Guide](https://docs.dify.ai/en/guides/application-publishing/developing-with-apis)
or the API documentation of your self-hosted Dify application.

## Speech recognition and synthesis endpoints

Alongside the conversational API, the router exposes the pipeline's STT and TTS as standalone
endpoints. They are useful on their own — a transcription service, a voice service — without
any conversation involved.

Both honour `api_key` when it is set: send `Authorization: Bearer {api_key}`.

### POST /transcribe

Audio in, text out. The request is **multipart/form-data**, not a raw body.

| Field | Type | Notes |
| --- | --- | --- |
| `audio` | file | Required — **raw PCM**, see below |
| `session_id` | form field | Optional |

**Send raw PCM, not a container.** The bytes you upload are handed to the recognizer
unchanged, and the recognizer wraps them in a WAV container itself. Uploading a `.wav` file
therefore feeds its 44-byte RIFF header in as audio samples and produces garbage.

| Requirement | Value |
| --- | --- |
| Container | None — headerless PCM |
| Encoding | Signed 16-bit little-endian |
| Channels | Mono |
| Sample rate | Whatever the recognizer is configured for — `16000` by default |

```python
import requests

# Raw PCM: signed 16-bit LE, mono, matching the recognizer's sample rate
with open("audio.pcm", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"audio": ("audio.pcm", f, "application/octet-stream")},
        data={"session_id": "session_001"},          # Optional
        headers={"Authorization": f"Bearer {API_KEY}"},   # When api_key is set
    )

print(response.json())
# {"text": "recognized speech", "preprocess_metadata": null,
#  "postprocess_metadata": null, "speakers": null}
```

To send an existing WAV file, strip the container first and confirm the format matches:

```python
import wave

with wave.open("audio.wav", "rb") as w:
    assert w.getnchannels() == 1 and w.getsampwidth() == 2
    assert w.getframerate() == 16000        # Must match the recognizer
    pcm = w.readframes(w.getnframes())
```

The same applies to audio sent to the conversational endpoints: `audio_data` is raw PCM in
the same format, which is what the WebSocket and HTTP clients stream.

`speakers` is populated only when a speaker registry is configured on the server — see
[Speech-to-Text](stt.md). An empty audio field returns `400`.

`POST /transcribe/speaker` registers a name for a speaker in that registry.

### POST /synthesize

Text in, WAV out. The request is JSON; the response is the audio itself, not a wrapper.

| Field | Notes |
| --- | --- |
| `text` | Required |
| `style_info` | Optional, passed through to the synthesizer |
| `language` | Optional |

```python
import requests

response = requests.post(
    "http://localhost:8000/synthesize",
    json={"text": "Hello, this is AI Avatar speaking"},
    headers={"Authorization": f"Bearer {API_KEY}"},   # When api_key is set
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

The response carries `Content-Type: audio/wav` and a `Content-Disposition` filename. Missing
`text` returns `400`.

Pass `stt=` or `tts=` to `get_api_router()` to serve these endpoints from a different
recognizer or synthesizer than the conversational pipeline uses.

## REST API Adapter

`AIAvatarHttpServer` exposes the streaming HTTP/SSE API. Its default channel name is `"http"`.

```python
from aiavatar.adapter.http.server import AIAvatarHttpServer

http_adapter = AIAvatarHttpServer(
    sts=sts,
    channel="http",
    api_key="YOUR_HTTP_API_KEY",  # Optional
)
app.include_router(http_adapter.get_api_router(path="/chat"))
```

The request and response formats are documented above.

## See also

- [Adapters](adapters.md) — choosing a channel and sharing a pipeline
- [WebSocket adapter](adapters-websocket.md) — the lower-latency alternative

---

[← Documentation index](../README.md#-documentation)
