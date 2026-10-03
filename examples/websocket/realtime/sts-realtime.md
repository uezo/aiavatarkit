# Realtime with an external TTS

This comparison pipeline sends microphone audio to Realtime, requests text
responses, and synthesizes them with an AIAvatarKit `SpeechSynthesizer`.
It uses the same browser pages as the cascade. Realtime's native output audio
is not used; for native voice, see the [GPT-Live example](sts-live.md).

## Run

Follow the installation and credential setup in the [README](README.md#run),
start VOICEVOX, and run from the repository root:

```sh
python -m examples.websocket.realtime.server --pipeline realtime
```

Open `/html/index.html` or `/html/3d.html` on the example server, enable
**BARGE-IN**, and press **Start**. Use `--voicevox-url` and
`--voicevox-speaker` to select the engine and voice. The shared page's microphone
mute and BARGE-IN controls keep their existing behavior; the example's server
always permits interruption when microphone audio reaches it.

For another synthesizer, pass an existing `SpeechSynthesizer` to the app factory:

```python
from examples.websocket.realtime.server import create_app

app = create_app("realtime", tts=your_synthesizer)
```

The app shuts down the pipeline and closes its own default TTS. An injected TTS
remains caller-owned: close it after the app's shutdown completes.

## Pipeline configuration

The implementation is local to this example, under
[`pipeline/realtime/`](pipeline/realtime/). Its public imports are:

```python
from examples.websocket.realtime.pipeline.realtime import (
    OpenAIRealtimePipeline,
    InputSilenceEvent,
    RealtimeInputEvent,
)
```

For settings beyond the CLI, adapt the pipeline construction in
[`server.py`](server.py). For example, selecting spoken XML sections is a
pipeline option:

```python
pipeline = OpenAIRealtimePipeline(
    tts=your_synthesizer,
    instructions="Put spoken responses in <answer>...</answer>.",
    voice_text_tags=["answer"],
    input_sample_rate=16000,
    context_db_path=None,
    restore_context=False,
)
```

Use the example's [`RealtimeWebSocketServer`](adapter.py) for the transport.
It prepares the provider session before sending `connected`, then activates
provider output. Browser disconnects use the adapter's normal finalization path.
The app's lifespan calls `shutdown()` to close any remaining sessions.

## Audio, captions, and response events

- Input is mono PCM16. The supplied browser sends 16 kHz audio; the pipeline
  resamples it to 24 kHz for the upstream connection. Match `input_sample_rate`
  to the client when constructing a different client.
- Generated text is split into sentences and sent through the injected TTS in
  order. `sentence_end` controls delimiters; `tts_queue_size` bounds the waiting
  synthesis queue. The final text suffix is flushed on response completion.
- `voice_text_tags` selects sections for speech. Control tags are removed from
  TTS input but remain in response text for the existing adapter parser.
  Face and animation tags therefore use the existing avatar controls.
- TTS output uses the ordinary WAV or finite PCM playback path. It does not
  carry `continuous_audio`.
- Input transcription is optional and disabled by the app factory. Set
  `transcription_model` on the pipeline to enable it. Input captions arrive as
  complete `info.metadata.partial_request_text` snapshots; they are separate
  from the text used to generate the response.

Accepted responses use the existing `accepted`, `start`, synthesized `chunk`,
and `final` events. Interruptions stop previous output; failure and cancellation
use the existing error/cancellation paths. Server completion does not prove
that the browser has played all of the audio.

## Scope of the comparison app

The WebSocket example accepts microphone audio, configuration, and session
controls. Text/image `invoke`, Nod, Admin, application tools, and custom local
turn-taking are outside its scope.

The migrated pipeline retains optional input/silence callbacks, direct text
`invoke()` on an active session, history restoration, and performance recording.
These are not enabled or exposed by the comparison app. In particular,
`context_db_path=None` with no injected context manager disables local history;
`restore_context=False` keeps each connection independent. If constructing the
pipeline directly, keep those explicit settings for the same behavior.
An injected performance recorder remains caller-owned.

See the constructor and methods in [`pipeline.py`](pipeline/realtime/pipeline.py)
for those optional interfaces, and the [README](README.md#local-verification)
for local verification commands. Provider tests use fake connections and TTS;
they do not measure actual API or playback latency.
