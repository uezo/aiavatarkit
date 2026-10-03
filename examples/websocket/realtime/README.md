# Realtime voice comparison

Swap the pipeline behind the existing WebSocket avatar pages to compare voice
conversation with the usual VAD → STT → LLM → TTS cascade. This is a small
comparison example, not a replacement for the main AIAvatarKit pipeline.

| Pipeline | Input and output |
| --- | --- |
| `OpenAIRealtimePipeline` | Microphone → Realtime text response → external TTS (VOICEVOX by default) |
| `OpenAILivePipeline` | Microphone ↔ GPT-Live native, continuous audio |

Realtime preserves the archived implementation's external TTS so you can use
the same voice as the cascade. It does **not** use Realtime's native voice
output. GPT-Live uses its configured voice and Responses delegation.

Pipeline details: [Realtime with external TTS](sts-realtime.md) and
[GPT-Live continuous audio](sts-live.md).

## Run

Run from the repository root with AIAvatarKit installed (`pip install -e .`).
Supply your existing OpenAI key through `OPENAI_API_KEY` in the process
environment. The example does not read local experiments or `pytest.ini`.

Realtime additionally needs a running VOICEVOX service, or an injected
`SpeechSynthesizer` in `create_app()`:

```sh
python -m examples.websocket.realtime.server --pipeline realtime
```

GPT-Live does not need VOICEVOX:

```sh
python -m examples.websocket.realtime.server --pipeline live
```

Open [the image avatar](http://127.0.0.1:8000/html/index.html) or
[the existing 3D viewer](http://127.0.0.1:8000/html/3d.html), enable **BARGE-IN**,
then press **Start**.
Both modes serve the existing `../html/` directory directly. Microphone access
requires localhost or HTTPS. The default server binds to localhost; if you
expose it elsewhere, configure your usual TLS and authentication.

Available options include `--port`, `--model`, `--instructions`, `--voice`
(Live), `--voicevox-url`, and `--voicevox-speaker` (Realtime). For example:

```sh
python -m examples.websocket.realtime.server --pipeline live --port 8001 \
  --instructions 'Speak briefly and naturally in Japanese.'
```

For application-specific TTS or authentication, call
`create_app("realtime", tts=your_synthesizer, api_key=your_websocket_key)` and
run the returned FastAPI app. Set the matching client `apiKey` in the shared
page's client configuration. An injected TTS remains caller-owned; the example
closes its own default TTS and all provider sessions on shutdown.

## Comparison scope

- Microphone input, audio output, the existing avatar/lip sync, captions, and
  Start/Stop use the same pages as the cascade.
- Enable BARGE-IN manually to keep microphone input active during playback.
  The page keeps its existing toggle and mute controls.
  Realtime uses provider VAD; GPT-Live manages its continuous conversation.
- Text/image `invoke`, Nod, application function tools, custom turn-taking,
  Admin integration, recording, and persistent context are outside this demo.
  Start a new connection for a fresh conversation.
- The comparison app disables Realtime history storage with
  `context_db_path=None` and does not create a conversation database. The
  migrated pipeline retains its optional history/metrics code for callers.
- GPT-Live captions are display snapshots. They do not create turns or trigger
  TTS. Live has no per-utterance `final`, so its audio is excluded from the
  3D backlog. Backlog compatibility is not a requirement of this example.

## Integration boundary

`adapter.py` extends the existing `AIAvatarWebSocketServer`. It prepares the
provider session before `connected`, then releases provider output; disconnect
uses the existing adapter's finalization path. The core `aiavatar/` package is
unchanged. Source under `pipeline/` was migrated from the archived experiments,
with core imports pointing to the installed `aiavatar` package.

Ordinary WAV and finite PCM responses keep their existing framing. Only Live
uses a descriptor and chunks carrying `metadata.continuous_audio=true`, `audio_id`,
and `pcm_format`, without an `audio_frame_count`. The shared player schedules
these continuously and ends them on stop/disconnect rather than inventing
voice-turn boundaries. Input and output use mono PCM16; the supplied pages
send microphone input at 16 kHz.

`continuous_audio` identifies session-long audio without per-turn completion;
it is not set for ordinary WebSocket PCM chunks, which retain backlog support.

Live playback uses the existing PCM queue without dropping audio to catch up.
If playback falls behind, use Stop/Start to clear the queue and begin a new session.

## Local verification

Provider tests use fake WebSockets and TTS; they do not require keys or services:

```sh
python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  tests/examples/websocket/realtime -q
node --test tests/examples/websocket/realtime/*.mjs \
  tests/examples/websocket/test_pcm_playback.mjs \
  tests/examples/websocket/test_partial_transcript.mjs \
  tests/examples/websocket/test_3d_backlog.mjs
```

Real microphone/provider checks are separate and use the API account configured
in the process environment.
