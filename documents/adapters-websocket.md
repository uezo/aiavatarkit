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

The Silero detectors handle [recoverable audio-filter errors](vad-filters.md#recoverable-errors)
by dropping the affected chunk before VAD processing. The adapter can then
continue receiving on the same connection, without filter-specific handling.


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
`audio_data: null`. It preserves the response's text, `voice_text`,
`voice_text_tag`, controls, and metadata, and adds:

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

`send_response()` frames raw audio supplied as `bytes` in `AIAvatarResponse.audio_data`.
When passing already encoded audio directly, supply the Base64 as a `str`.

The maintained `index.html` and `3d.html` clients support signed 16-bit
little-endian PCM with the declared sample rate and channel count. Use the
updated server and browser together for this descriptor protocol. They schedule
each arriving PCM chunk for playback without waiting for the complete audio to
arrive. The server still synthesizes a complete WAV before splitting it, so the
total length is known before transmission. See the
[browser playback details](../examples/websocket/README.md#pcm-audio-playback).

With `debug=True`, the server logs one INFO summary per original audio, prefixed
`WebSocket audio:` with its format, IDs, PCM format, byte/chunk sizes, and `voice_text_tag`.
For PCM, `audio_id` matches the browser's `playback_id`; `planned_chunks` excludes
the descriptor and counts planned chunks, so interruption can reduce the number actually sent.
WAV reports `audio_id=None` and `planned_chunks=1`; its browser playback ID is generated later.
`debug=False` emits no audio summaries.

### Protect an ACK from barge-in

Set `barge_in_protected_tags` to keep selected voice clips playing during barge-in
in the maintained browser clients.

```python
server = AIAvatarWebSocketServer(
    vad=vad, stt=speech_recognizer, llm=llm, tts=tts,
    mute_on_barge_in=True,
    barge_in_protected_tags=["ack"],
)
```

WebSocket chunks expose the pipeline's `voice_text_tag` as a top-level field.
For audio, a tag is carried on the complete WAV message or PCM descriptor.
Untagged chunks and the following raw PCM payloads have `voice_text_tag: null`.
An ACK has `voice_text_tag: "ack"` and uses the same audio framing as the answer.
The adapter matches `barge_in_protected_tags` against this tag and defaults to
protecting no tags. Include `"ack"` in the LLM's
[`voice_text_tag`](llm.md#voice-text-tag-think-before-answering) configuration.

The browser keeps protected WAV and PCM clips on an ordinary `stop`, including
when `mute_on_barge_in` stops playback or the pipeline starts a new response.
Once the server begins sending protected PCM audio, it finishes the clip using
the configured chunk size. Muting before a clip begins sending still drops it.
In this example, the answer remains interruptible, and the microphone stays
active when BARGE-IN is enabled.

A `stop` with `metadata.force_stop=true` stops even protected audio. Guardrail
stops use this flag, and applications can set it through
`server.stop_response(session_id, context_id, metadata={"force_stop": True})`.
The browser Stop button and disconnect also stop all audio.

## Response transaction IDs

WebSocket responses expose the pipeline's `transaction_id` as an optional
top-level field. Events for one response, including `start`, audio `chunk`, and
`final`, carry the same ID. Both complete-WAV and split-PCM audio preserve it.
When a new transaction interrupts the previous one, the synthetic interrupted
`final` carries the old transaction's ID and `accepted` carries the new one.
Each transaction ends with one `final`: a transaction that already sent its own
`final` gets no synthetic one, and no output of the old transaction follows it.

Session controls such as `connected`, `voiced`, and the session-wide `stop`
notification may have no transaction ID. Clients also accept older responses
without this field. This ID identifies an AI response; it is distinct from the
browser's per-audio `playback_id` and PCM `audio_id`. The shared browser returns
the transaction ID in [playback start notifications](../examples/websocket/README.md#playback-context-for-turn-taking).

## Context for interruptions

The LLM can use the part of the assistant response that had already played when
the user started speaking. This option uses playback reports from the client and
requires a VAD that calls `on_speech_started` callbacks with `(session_id, metadata)`,
such as `SileroStreamSpeechDetector`. The adapter updates the recording's metadata
in place. Enable it when constructing the adapter.

```python
aiavatar_app = AIAvatarWebSocketServer(
    vad=vad, stt=speech_recognizer, llm=llm, tts=tts,
    use_invoke_queue=True,
    merge_request_threshold=0.0,
    mute_on_barge_in=True,
    include_interruption_context=True,
)
```

The [invoke queue](pipeline.md#invoke-queue) lets a follow-up wait for the
previous request to finish and save its history. In direct mode
(`use_invoke_queue=False`), a follow-up can run before the first reply saves
any history. If the pipeline creates a new conversation ID in that case, it
processes the follow-up as an ordinary request without the interruption
supplement.

The adapter reads the browser's playback reports directly for interruption
context, with or without a turn-taking gate. If you use
[playback-aware turn taking](vad-turn-taking.md#quick-start-with-jev), keep
forwarding the same reports to the gate in your `on_request` hook.

A recording captures the estimated spoken text at its first detected speech.
It uses the current clip as soon as some text has played. While a new clip
has no spoken text yet, the previous audible clip can supply the context.
The spoken text remains available for follow-up utterances until the browser
reports that the reply's last audio clip has played in full.
Successive chunks in the same streamed reply retain the earlier spoken text.

The adapter matches reported playback to the response's audio chunks by
transaction ID and `voice_text`. Unplayed chunks do not contribute to the
spoken text. ACKs use the same playback start and end reports as answers.
The browser selects the final non-ACK audio clip for its playback final
report. An ACK-only response still has a response `final`, and its spoken audio
remains available as interruption context after playback ends.

When there is no spoken prefix and the active response has sent no audio, the
adapter can capture that response's transaction ID and the last transaction
that sent audio or an ordinary `final`.
The pipeline uses those IDs to recover the unanswered text of earlier requests
that reached processing. A request replaced while waiting in the queue is
discarded. A normal new turn starts a fresh chain.

The supplement applies to the accepted recording in the same connection and
conversation. Use a fresh `session_id` for every WebSocket session, including
reconnects. Pass an existing `context_id` to resume a conversation. When processing
begins, the adapter finds the session by `session_id` and passes the captured
facts to the pipeline. The pipeline creates tracking state at session start
and removes it when the session closes. Requests can only update existing state.
`start.metadata.request_text`,
the QuickResponder when used, and the main LLM receive the augmented text.
`start.metadata.recognized_text` still contains the original transcript. The
system prompt stays the same, and the LLM's conversation history can retain the
augmented user message.

Override the wording with the [pipeline's interruption templates](pipeline.md#interruption-context).

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
