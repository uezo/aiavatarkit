# Getting started

AIAvatarKit runs as a FastAPI application. You can start it with the bundled `aiavatar`
command and never write Python, or assemble the pipeline yourself when you need control
over components, routes, and application lifecycle.

## Requirements

- Python 3.11 or newer
- An OpenAI API key (the default stack uses OpenAI for STT and LLM)
- A reachable VOICEVOX-compatible server for Japanese speech synthesis

## Install

```sh
pip install aiavatar
```

Provider SDKs are not installed by default. Add them only for the components you use:

| Component | Extra install |
| --- | --- |
| Anthropic Claude | `pip install anthropic` |
| Google Gemini | `pip install google-genai` |
| LiteLLM | `pip install litellm` |
| Azure Stream speech detector | `pip install azure-cognitiveservices-speech` |
| Amazon Transcribe stream speech detector | `pip install amazon-transcribe` |
| Amazon Polly recipe | `pip install boto3` |
| Smart Turn gate | `pip install "aiavatar[smart-turn]"` |
| Namo Turn gate | `pip install "aiavatar[namo-turn]"` (the `aiavatar` command offers to do this for you — see [Your first application](#your-first-application)) |
| Local microphone and speaker | `pip install "aiavatar[local-audio]"` |
| HTTP (SSE) adapter | `pip install sse-starlette python-multipart` |
| HTTP STT/TTS client examples | `pip install requests` |
| OpenAI-compatible endpoint adapter | `pip install sse-starlette` |
| LINE Bot adapter | `pip install line-bot-sdk` |
| Twilio adapter | `pip install twilio` |
| Speaker diarization and registry | `pip install resemblyzer` |
| VRChat face and animation control | `pip install python-osc` |
| Vision from a camera | `pip install opencv-python` |
| PostgreSQL for any persistence layer | `pip install asyncpg` |
| Azure Blob voice recorder | `pip install azure-storage-blob` |
| SpeechGateway local gateway mode | `pip install speech-gateway` |
| Web scraper tool | `pip install playwright` then `playwright install chromium` |
| MCP | `pip install fastmcp` |

Azure and Google speech recognition and synthesis call their REST APIs directly, so they
need no vendor SDK.

Two of these need something beyond `pip`:

- **`AudioConverter.convert` runs `ffmpeg`**, which is not a Python package. Install it with
  your system package manager if you use a TTS provider that returns encoded audio. A
  missing binary only shows up at synthesis time. See
  [Text-to-Speech](tts.md#audio-format-conversion).
- **The web scraper launches Chromium.** `pip install playwright` installs the library only;
  `playwright install chromium` downloads the browser it drives. Without it the tool fails
  the first time it runs.

## Your first application

```sh
export OPENAI_API_KEY=sk-xxx
aiavatar
```

The built-in application uses the Namo Turn semantic VAD gate, whose dependencies are not
part of the base install. When they are missing, the command asks before doing anything else:

```text
Additional dependencies are required to enable Semantic VAD. Install them now? [y/N]:
```

Answer `y` and it runs `pip install "aiavatar[namo-turn]"` in the current Python environment,
then continues the same launch — no second command, no restart. Answer `n` (the default) and
the application starts without the Namo Turn gate; Silero VAD and the filler gate still run,
so turn-end detection falls back to silence plus filler handling.

The first run *with* the gate enabled also downloads the turn-end model and its tokenizer
from Hugging Face, so that run needs network access and takes a moment before the server is
ready. Later runs read from the cache.

Open http://127.0.0.1:8000/ to talk to the avatar, and http://127.0.0.1:8000/admin/ for the
Admin Panel.

To assemble a WebSocket application yourself, save this as `run.py`:

```python
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.admin import setup_admin_panel
from aiavatar.util import download_example

# Download example UI if not exists
html_dir = download_example("websocket/html")

# Build Speech-to-Speech pipeline with WebSocket adapter
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

app = FastAPI()
app.include_router(aiavatar_app.get_websocket_router())

# Admin panel (optional)
setup_admin_panel(app, adapter=aiavatar_app)

# Serve the UI at "/". This catch-all mount must come after the routes above.
app.mount("/", StaticFiles(directory=html_dir, html=True), name="ui")
```

It serves the same URLs as the command above.

```sh
python -m uvicorn run:app
```

**NOTE:** If the steps in a technical blog do not work as expected, the post may predate
v0.6. Try `pip install aiavatar==0.5.8` to match the environment it describes.

The `aiavatar` command starts a ready-to-use WebSocket application when no script is supplied, or runs a custom Python ASGI application in script mode. Use `aiavatar --help` to list command-line options such as `--host` and `--port`.

## Built-in Application

The built-in application uses `SileroStreamSpeechDetector` with the filler and Namo Turn gates, `OpenAISpeechRecognizer`, `OpenAIResponsesWebSocketService`, and the WebSocket Adapter. Japanese speech routes to `VoicevoxSpeechSynthesizer` with `AlphabetToKanaPreprocessor`; other languages route to `OpenAISpeechSynthesizer`.

### Semantic VAD dependencies

Namo Turn needs `onnxruntime`, `transformers`, and `huggingface-hub`, none of which ship with the base package. The command checks for them before it builds anything, and what happens next depends on where it is running:

| Situation | Behaviour |
| --- | --- |
| All three are importable | Namo Turn is enabled, silently |
| Missing, interactive terminal | Prompts to install `aiavatar[namo-turn]`, then continues the same launch |
| Missing, prompt declined | Starts without the Namo Turn gate |
| Missing, non-interactive (containers, systemd, CI) | Logs a warning and starts without the Namo Turn gate — it never installs unattended |
| Script mode | Not checked at all; the script owns its component graph |

Because a declined or non-interactive start still runs, a container image that was never built with the extra will keep working — just with silence-based turn-end detection instead of semantic. Add `aiavatar[namo-turn]` to the image when you want the gate, rather than relying on the prompt that a container will never see.

The same choice is available in Python. `create_app(use_namo_turn=False)` and `build_components(use_namo_turn=False)` build the default stack without the gate, and without importing its dependencies.

The command downloads the WebSocket example UI into `html/` only when that directory does not already exist. The application is then available at http://127.0.0.1:8000/, with the Admin Panel at http://127.0.0.1:8000/admin/.

See [`.env.example`](../.env.example) for all built-in application settings. The command automatically loads `.env` from the current working directory without overriding variables already present in the process environment:

```sh
cp .env.example .env
# Edit OPENAI_API_KEY in .env
aiavatar
```

The Admin Config view can update safe members of the running Pipeline, components, and Adapter. These changes are intentionally volatile and are discarded when the process exits. Component composition remains owned by Python application code.

## OpenAI and LLM Configuration

`OPENAI_API_KEY` and `OPENAI_BASE_URL` are the shared defaults for STT, LLM, and OpenAI TTS. A component-specific value takes precedence when set:

| Component | API key | Base URL |
| --- | --- | --- |
| STT | `AIAVATAR_STT_OPENAI_API_KEY` | `AIAVATAR_STT_OPENAI_BASE_URL` |
| LLM | `AIAVATAR_LLM_OPENAI_API_KEY` | `AIAVATAR_LLM_OPENAI_BASE_URL` |
| TTS and its preprocessors | `AIAVATAR_TTS_OPENAI_API_KEY` | `AIAVATAR_TTS_OPENAI_BASE_URL` |

Command options take precedence over their corresponding shared environment variables, so `--openai-api-key` and `--openai-base-url` affect the launched process. Component-specific environment variables still take precedence over those shared values. Prefer environment variables or the hidden API-key prompt because command arguments may be recorded in shell history and process listings.

Set `AIAVATAR_LLM_MODEL` or `AIAVATAR_LLM_SYSTEM_PROMPT` before startup to override the default application's LLM model or system prompt:

```sh
AIAVATAR_LLM_MODEL=gpt-5.6-terra \
AIAVATAR_LLM_SYSTEM_PROMPT="You are a helpful voice assistant." \
OPENAI_API_KEY=sk-... aiavatar
```

The default LLM API is the OpenAI Responses WebSocket API. Set `AIAVATAR_LLM_API=chat-completions` or pass `--llm-api chat-completions` for an OpenAI-compatible service that only implements Chat Completions. Those two values are the only ones accepted; any other raises at startup.

For an OpenAI-compatible endpoint that uses `extra_body` instead of OpenAI's `reasoning_effort`, pass a JSON object through `AIAVATAR_LLM_EXTRA_BODY` or `--llm-extra-body`. Supplying a non-empty object disables the default `reasoning_effort="none"`; the command option takes precedence. Set `AIAVATAR_LLM_REASONING_EFFORT` when the provider needs an explicit value, or set it to `omit` to suppress the field independently of `extra_body`.

```sh
aiavatar --llm-extra-body '{"thinking":{"type":"disabled"}}'
```

For example, STT and TTS can continue using OpenAI while only the LLM uses an OpenAI-compatible Chat Completions endpoint:

```sh
OPENAI_API_KEY=sk-openai-... \
AIAVATAR_LLM_OPENAI_API_KEY=provider-key \
AIAVATAR_LLM_OPENAI_BASE_URL=https://provider.example/v1 \
AIAVATAR_LLM_MODEL=provider/model \
aiavatar --llm-api chat-completions
```

## Built-in TTS Routing

Japanese and non-Japanese TTS are independent routes. `AIAVATAR_JA_TTS` defaults to `voicevox`, while `AIAVATAR_MULTI_TTS` defaults to `openai`; either route can select `voicevox`, `openai`, or `instant`. The corresponding `AIAVATAR_JA_TTS_CONFIG` and `AIAVATAR_MULTI_TTS_CONFIG` JSON objects override that route's provider options. `--ja-tts` and `--multi-tts` override only the provider selection.

Shared VOICEVOX and OpenAI defaults remain available through `AIAVATAR_VOICEVOX_*` and `AIAVATAR_OPENAI_TTS_*`. Route config values take precedence. Japanese TTS enables `AlphabetToKanaPreprocessor` by default and the multi route disables it; set `"alphabet_to_kana": false` or `true` in the applicable route config to override that behavior.

`instant` maps the route config to `create_instant_synthesizer()`. It is intentionally limited to a single HTTP request whose raw response body is uncompressed PCM WAV audio. Authentication headers, request parameters, and JSON bodies are supplied directly in the config. More complex response parsing, encoded audio extraction, conversion, or authentication logic belongs in a Python application script.

For example, Aivis Cloud can be configured as an instant Japanese TTS. The API key is part of the private process environment and must not be committed:

```sh
AIAVATAR_JA_TTS=instant \
AIAVATAR_JA_TTS_CONFIG='{"method":"POST","url":"https://api.aivis-project.com/v1/tts/synthesize","headers":{"Authorization":"Bearer YOUR_AIVIS_API_KEY","Content-Type":"application/json"},"json":{"model_uuid":"YOUR_MODEL_UUID","text":"{text}","output_format":"wav","output_sampling_rate":16000,"output_audio_channels":"mono","use_ssml":false},"cache_dir":"ttscache/aivis"}' \
aiavatar
```

## Script Mode

The command can also run a Python ASGI application that exports `app`:

```sh
aiavatar run.py
```

In script mode, the script owns the complete component graph, Adapter, Admin setup, routes, and lifespan. The command only loads the application, supplies process-level options such as `--host` and `--port`, and runs Uvicorn. `python -m uvicorn run:app` remains fully supported.

Starter applications can reuse only the built-in component defaults while keeping the Pipeline and Adapter explicit and easy to customize:

```python
from aiavatar.cli import build_components

components = build_components()
vad, stt, llm, tts = components

# Assemble STSPipeline, the Adapter, hooks, and FastAPI here.
# Await components.close() from the application's shutdown path.
```

Pass any custom component to `build_components(vad=..., stt=..., llm=..., tts=...)`; only omitted components are created. If STT is supplied while VAD is omitted, the custom STT is used by the default streaming VAD. The helper deliberately does not create a Pipeline, Adapter, routes, or application lifespan.

`build_components(use_namo_turn=False)` builds the default streaming VAD without the Namo Turn gate, leaving the filler gate in place. Use it when the optional dependencies are not installed — script mode does not check for them, so the import error would otherwise surface as a startup crash. Supplying your own `vad` makes the argument irrelevant.

## See also

- [Pipeline](pipeline.md) — how a turn flows through the system
- [Adapters](adapters.md) — serving other channels than WebSocket
- [Administration](admin.md) — the Admin Panel and its REST API

---

[← Documentation index](../README.md#-documentation)
