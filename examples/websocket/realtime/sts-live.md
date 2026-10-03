# GPT-Live speech sessions

`OpenAILivePipeline` connects microphone audio to GPT-Live and plays its native
continuous audio. There is no local VAD/STT/LLM/TTS turn loop and no synthetic
`accepted`, `start`, or `final` event per utterance. For text generation with a
chosen external TTS voice, see the [Realtime example](sts-realtime.md).

## Run

Follow the installation and credential setup in the [README](README.md#run),
then run from the repository root:

```sh
python -m examples.websocket.realtime.server --pipeline live
```

Open `/html/index.html` or `/html/3d.html`, enable **BARGE-IN**, and press
**Start**. No VOICEVOX service is needed. BARGE-IN is enabled manually so the
existing browser microphone policy allows listening during playback.

The app factory accepts `model`, `voice`, and `instructions`:

```python
from examples.websocket.realtime.server import create_app

app = create_app("live", instructions="日本語で短く自然に会話してください。")
```

For lower-level configuration, the pipeline import is:

```python
from examples.websocket.realtime.pipeline.live import OpenAILivePipeline
```

Adapt the construction in [`server.py`](server.py) to set options such as
`delegation` or `debug`. Use [`RealtimeWebSocketServer`](adapter.py), which
prepares the session before `connected` and activates output after that write.
The app finalizes each disconnected session and calls `shutdown()` on exit.

## Continuous audio and playback

The supplied app uses mono signed 16-bit little-endian PCM at 16 kHz for input
and output. The pipeline also accepts 24 kHz when constructing a matching client;
it does not resample. Audio format stays fixed for the stream.

The adapter sends an audio-free descriptor carrying `audio_id`, `pcm_format`,
and `continuous_audio: true`, followed by base64 PCM chunks with the same ID.
There is no `audio_frame_count` because the stream has no predetermined length.
Audio is not wrapped in WAV or decoded separately for every provider packet.
A `stop` clears playback; audio sent afterward gets a new descriptor and ID.

The shared PCM player schedules incoming buffers consecutively. When all
received buffers have finished, it clears the playing flag and closes the
avatar's mouth while keeping the stream available. New audio resumes that
stream. This detects an empty playback queue, not acoustic silence: continuous
PCM containing zero-valued samples is still considered playing.

Live playback has no special audio-dropping or catch-up policy. If playback
falls behind, use Stop/Start to clear it and begin a new session. The 3D backlog
excludes continuous audio because Live does not supply per-utterance finals.

## Captions and events

| Event | Meaning in this example |
| --- | --- |
| `connected` | Provider preparation has completed. |
| `chunk` | Continuous PCM descriptor or audio data, as described above. |
| `info` | User caption in `metadata.partial_request_text` or AI caption in `metadata.partial_response_text`. |
| `backend_event` | Delegated Responses event, separate from the spoken stream. |
| `usage` | Cumulative usage snapshot rather than an increment to sum. |
| `stop` | Clear local playback. |
| `session_closed` | End of the session, with a close reason and finalization status. |
| `error` | Session, provider, transport, or unsupported-operation failure. |

Caption deltas are accumulated into complete display snapshots, resetting when
the speaker changes. The shared UI replaces the displayed text; captions do
not define turn boundaries, trigger TTS, or identify words already heard.
Caption text stays in metadata so it is not parsed as avatar control tags.

Responses delegation is configured on the pipeline. Completion of a delegated
response does not end the voice stream. Application function tools, text/image
requests, recording, Admin integration, and persistent conversation history
are outside this example. A `context_id` does not restore a Live conversation
after reconnecting.

## Optional diagnostics

When constructing the pipeline directly, enable its event logger with:

```python
import logging
from examples.websocket.realtime.pipeline.live import OpenAILivePipeline

logging.getLogger("examples.websocket.realtime.pipeline.live.events").setLevel(logging.DEBUG)
pipeline = OpenAILivePipeline(debug=True)
```

Configure a logging handler in the application to display these records.
Debug events include transcript text and audio receipt timing, so enable them
only when needed. Audio payloads and credentials are not included.
`audio_total_ms` is cumulative received PCM duration; `audio_elapsed_ms` is
elapsed time since the first audio block. Differences across several seconds
can show delivery pace, but they do not measure browser playback latency.

The old external voice-changer bridge and its environment options are not part
of this app. For local fake-provider and browser checks, see
[Local verification](README.md#local-verification).
