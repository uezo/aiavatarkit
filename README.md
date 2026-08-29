# AIAvatarKit

🥰 Building AI-based conversational avatars lightning fast ⚡️💬

![AIAvatarKit Architecture Overview](documents/images/aiavatarkit_overview.png) 

## ✨ Features

- **🌏 Live anywhere**: AIAvatarKit is a general-purpose Speech-to-Speech framework with multimodal input/output support. It can serve as the backend for a wide range of conversational AI systems.
    - Metaverse Platforms: Compatible with VRChat, cluster, Vket Cloud, and other platforms
    - Standalone Apps: Enables ultra-low latency real-time interaction via WebSocket or HTTP (SSE), with a unified interface that abstracts differences between LLMs
    - Channels and Devices: Supports edge devices like Raspberry Pi and telephony services like Twilio
- **🧩 Modular architecture**: Components such as VAD, STT, LLM, and TTS are modular and easy to integrate via lightweight interfaces. Supported modules include:
    - VAD: Built-in standard VAD (silence-based end-of-turn detection), SileroVAD
    - STT: Google, Azure, OpenAI, AmiVoice
    - LLM: ChatGPT, OpenAI Responses API (REST / WebSocket), Gemini, Claude, and any model supported by LiteLLM or Dify
    - TTS: VOICEVOX / AivisSpeech, OpenAI, SpeechGateway (including Style-Bert-VITS2 and Aivis Cloud API)
- **⚡️ AI Agent native**: Designed to support agentic systems. In addition to standard tool calls, it offers Dynamic Tool Calls for extensibility and supports progress feedback for high-latency operations.


## 🚀 Quick start

**Requirements**: Python 3.11+, an OpenAI API key, and a reachable VOICEVOX-compatible server at its default URL.

Install AIAvatarKit.

```sh
pip install aiavatar
```

Start the built-in default application.

```sh
export OPENAI_API_KEY=sk-xxx
aiavatar
```

The built-in application uses Namo Turn semantic VAD. When its optional
dependencies are missing, the command asks whether to install them:

```text
Additional dependencies are required to enable Semantic VAD. Install them now? [y/N]:
```

Answer `y` to install `aiavatar[namo-turn]` into the current Python environment
and continue startup automatically. Answer `n` to start without the Namo Turn
gate; Silero VAD and the filler gate remain enabled.

Open http://127.0.0.1:8000/ and enjoy the conversation! The Admin Panel is available at http://127.0.0.1:8000/admin/.

See [Command Line Interface](#-command-line-interface) for configuration options and details about the built-in application.

Or, write your own application script when you need full, fine-grained control over components, routes, and application lifecycle. Save the following as `run.py`.

```python
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.util import download_example

# Download example UI if not exists
download_example("websocket/html")

# Build Speech-to-Speech pipeline with WebSocket adapter
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Build websocket server
app = FastAPI()
router = aiavatar_app.get_websocket_router()
app.include_router(router)
app.mount("/static", StaticFiles(directory="html"), name="static")

# Setup admin panel (Optional)
from aiavatar.admin import setup_admin_panel
setup_admin_panel(app, adapter=aiavatar_app)
```

Start server. Also, don't forget to launch VOICEVOX beforehand.

```bash
$ python -m uvicorn run:app
```

**NOTE:** If the steps in technical blogs don’t work as expected, the blog may be based on a version prior to v0.6. Some features may be limited, but you can try downgrading with `pip install aiavatar==0.5.8` to match the environment described in the blog.

For a Python console client that uses local microphone and speaker devices, see [WebSocket](#-websocket).


## 🔖 Contents

- [🎓 Generative AI](#-generative-ai)
    - [ChatGPT](#chatgpt)
    - [OpenAI Responses API](#openai-responses-api)
    - [Claude](#claude)
    - [Gemini](#gemini)
    - [Dify](#dify)
    - [OpenAI-compatible APIs](#openai-compatible-apis)
    - [Other LLMs](#other-llms)

- [🗣️ Speech Synthesizer](#%EF%B8%8F-speech-synthesizer)
    - [VOICEVOX](#voicevox)
    - [Azure](#azure)
    - [Google](#google)
    - [OpenAI](#openai)
    - [SpeechGateway](#speechgateway)
    - [Instant TTS Synthesizer](#instant-tts-synthesizer)
    - [TTS Routing](#tts-routing)
    - [TTS Caching](#tts-caching)
    - [Preprocessing](#preprocessing)
    - [Adjusting Speech Speed](#adjusting-speech-speed)

- [👂 Speech Listener](#-speech-listener)
    - [Preprocessing and Postprocessing](#preprocessing-and-postprocessing)
    - [Speaker Diarization](#speaker-diarization)
    - [Per-session STT Switching](#per-session-stt-switching)

- [🎙️ Speech Detector](#%EF%B8%8F-speech-detector)
    - [Silero VAD Speech Detector](#silero-speech-detector)
    - [Silero Stream Speech Detector](#silero-stream-speech-detector)
    - [Semantic VAD](#semantic-vad)
    - [Audio Filters](#audio-filters)
    - [Azure Stream Speech Detector](#azure-stream-speech-detector)
    - [AWS Stream Speech Detector](#aws-stream-speech-detector)
    - [Customization](#customization)
    - [Standard Speech Detector (Legacy)](#standard-speech-detector-legacy)

- [🥰 Face Expression](#-face-expression)

- [💃 Animation](#-animation)

- [🖼️ Artifacts](#%EF%B8%8F-artifacts)

- [🥳 Character Management](#-character-management)
    - [Get started](#get-started)
    - [Updating Diaries](#updating-diaries)
    - [Updating Schedules](#updating-schedules)
    - [Automated Daily Updates](#automated-daily-updates)
    - [Batch Generation](#batch-generation)
    - [Long-term Memory](#long-term-memory)
    - [Binding to Adapter](#binding-to-adapter)

- [🧩 API](#-api)
    - [💫 RESTful API (SSE)](#-restful-api-sse)
    - [🔵 Dify-compatible API](#-dify-compatible-api)
    - [🔌 WebSocket](#-websocket)
    - [🟩 LINE Bot](#-line-bot)

- [🦜 AI Agent](#-ai-agent)
    - [⚡️ Tool Call](#️-tool-call)
    - [⌛️ Tool Call with Streaming Progress](#%EF%B8%8F-tool-call-with-streaming-progress)
    - [🔄 Background Tool Execution](#-background-tool-execution)
    - [📋 Tool Response Formatter (Direct Response)](#-tool-response-formatter-direct-response)
    - [📦 Structured Content (Client-side Data)](#-structured-content-client-side-data)
    - [🪄 Dynamic Tool Call](#-dynamic-tool-call)
    - [🔌 MCP](#-mcp)
    - [🛠️ Built-in Tools](#️-built-in-tools)
    - [🦞 OpenClaw / Hermes](#-openclaw--hermes)

- [📡 Channel Adapter](#-channel-adapter)
    - [Adapters](#adapters)
    - [Connecting Multiple Channels](#connecting-multiple-channels)
    - [Sharing Context Across Channels](#sharing-context-across-channels)
    - [Channel-aware Processing](#channel-aware-processing)

- [💻 Command Line Interface](#-command-line-interface)
    - [Built-in Application](#built-in-application)
    - [OpenAI and LLM Configuration](#openai-and-llm-configuration)
    - [Built-in TTS Routing](#built-in-tts-routing)
    - [Script Mode](#script-mode)

- [🛡️ Guardrail](#%EF%B8%8F-guardrail)

- [🌎 Platform Guide](#-platform-guide)
    - [🐈 VRChat](#-vrchat)
    - [🍓 Raspberry Pi](#-raspberry-pi)

- [⚙️ Administration](#️-administration)
    - [Admin Panel](#admin-panel)
    - [REST API](#rest-api)
    - [📈 Observability](#-observability)

- [🧪 Evaluation](#-evaluation)

- [🤿 Deep Dive](#-deep-dive)
    - [🐘 PostgreSQL](#-postgresql)
    - [👀 Vision](#-vision)
    - [💾 Long-term Memory](#-long-term-memory)
    - [🐓 Wakeword](#-wakeword)
    - [📋 System Prompt Parameters](#-system-prompt-parameters)
    - [🎛️ Inline LLM Parameters](#️-inline-llm-parameters)
    - [⏰ Timestamp Insertion](#-timestamp-insertion)
    - [🧵 Request merging](#-request-merging)
    - [📥 Invoke Queue](#-invoke-queue)
    - [🧺 Shared Context](#-shared-context)
    - [🔗 Channel Session Manager](#-channel-session-manager)
    - [📡 Channel-aware Processing](#-channel-aware-processing)
    - [🔈 Audio Device](#-audio-device)
    - [🐆 Quick Response](#-quick-response)
    - [🎭 Custom Behavior](#-custom-behavior)
    - [✅ Request Validation](#-request-validation)
    - [🎚️ Noise Filter](#%EF%B8%8F-noise-filter)
    - [🔄 Migration Guide: From v0.6.x to v0.7.0](#-migration-guide-from-v06x-to-v070)


## 🎓 Generative AI

You can set model and system prompt when instantiate `AIAvatar`.

```python
aiavatar_app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    llm_reasoning_effort="none",
    system_prompt="You are my cat."
)
```

### ChatGPT

If you want to configure in detail, create instance of `ChatGPTService` with custom parameters and set it to `AIAvatar`.

```python
# Create ChatGPTService
from aiavatar.sts.llm.chatgpt import ChatGPTService
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    reasoning_effort="none",
    system_prompt="You are my cat."
)

# Create AIAvatar with ChatGPTService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

For Azure OpenAI and other custom configurations, construct the official client
yourself and pass it through `openai_client`. `model` is always the actual model
or Azure deployment name; it no longer needs to double as a provider switch.

```python
import os
from openai import AsyncAzureOpenAI

azure_client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
llm = ChatGPTService(
    openai_client=azure_client,
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    system_prompt="You are my cat.",
)
```

The previous `model="azure"` detection remains available for compatibility but
emits `DeprecationWarning`. `custom_openai_module` is deprecated in the same way;
pass a client instance instead. Injected clients are caller-owned and are not
closed by `ChatGPTService.close()`.

### OpenAI Responses API

Use `OpenAIResponsesService` to leverage the OpenAI Responses API. Conversation history is managed server-side via `previous_response_id`, eliminating the need for client-side context management.

```python
from aiavatar.sts.llm.openai_responses import OpenAIResponsesService
llm = OpenAIResponsesService(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-5.6-terra",
    system_prompt="You are my cat."
)

aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

`OpenAIResponsesService` also accepts `openai_client`, including a regular
`AsyncOpenAI` configured for Azure's `/openai/v1/` endpoint. As with
`ChatGPTService`, the injected client is caller-owned.

For lower latency, use the WebSocket variant. This maintains persistent connections via a connection pool, which can reduce latency by up to 40%, especially in tool-call-heavy workflows.

```python
# pip install websockets
from aiavatar.sts.llm.openai_responses_websocket import OpenAIResponsesWebSocketService
llm = OpenAIResponsesWebSocketService(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-5.6-terra",
    reasoning_effort="low",
    system_prompt="You are my cat."
)
```

NOTE: The WebSocket variant does not support the `temperature` parameter. Use `reasoning_effort` ("none", "low", "medium", "high", "xhigh", "max") instead to control response behavior. Dynamic Tool Calls are not supported in either variant, as the server-side history management via `previous_response_id` is incompatible with the pre-flight tool filtering calls.

The WebSocket variant uses the event protocol directly rather than an OpenAI
HTTP client. For Azure, set `ws_url` to the resource's `/openai` WebSocket base
and set `model` to the deployment name; no `"azure"` model sentinel is needed.


### Claude

Create instance of `ClaudeService` with custom parameters and set it to `AIAvatar`. The default model is `claude-sonnet-4-5`.

```python
# Create ClaudeService
from aiavatar.sts.llm.claude import ClaudeService
llm = ClaudeService(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model="claude-sonnet-4-5",
    temperature=0.0,
    system_prompt="You are my cat."
)

# Create AIAvatar with ClaudeService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

NOTE: We support Claude on Anthropic API, not Amazon Bedrock for now. Use LiteLLM or other API Proxies.


### Gemini

Create instance of `GeminiService` with custom parameters and set it to `AIAvatar`. The default model is `gemini-2.0-flash-exp`.

```python
# Create GeminiService
# pip install google-genai
from aiavatar.sts.llm.gemini import GeminiService
llm = GeminiService(
    gemini_api_key=GEMINI_API_KEY,
    model="gemini-2.0-pro-latest",
    temperature=0.0,
    system_prompt="You are my cat."
)

# Create AIAvatar with GeminiService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

NOTE: We support Gemini on Google AI Studio, not Vertex AI for now. Use LiteLLM or other API Proxies.


### Dify

You can use the Dify API instead of a specific LLM's API. This eliminates the need to manage code for tools or RAG locally.

```python
# Create DifyService
from aiavatar.sts.llm.dify import DifyService
llm = DifyService(
    api_key=DIFY_API_KEY,
    base_url=DIFY_URL,
    user="aiavatarkit_user",
    is_agent_mode=True
)

# Create AIAvatar with DifyService
aiavatar_app = AIAvatar(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```


### OpenAI-compatible APIs

`ChatGPTService` supports OpenAI-compatible APIs, such as Grok, Gemini, and Claude.

By specifying the `model`, `openai_api_key`, and `base_url`, these models can now be used with a non-reasoning configuration out of the box.

```python
# Grok
MODEL = "grok-4-1-fast-non-reasoning"
OPENAI_API_KEY = "YOUR_XAI_API_KEY"
BASE_URL = "https://api.x.ai/v1"

# Gemini on Google AI Studio
MODEL = "gemini-2.5-flash"
OPENAI_API_KEY = "YOUR_GEMINI_API_KEY"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Claude on Anthropic
LLM_MODEL = "claude-haiku-4-5"
OPENAI_API_KEY = "YOUR_ANTHROPIC_API_KEY"
BASE_URL = "https://api.anthropic.com/v1/"

# Configure ChatGPTService
from aiavatar.sts.llm.chatgpt import ChatGPTService
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    # extra_body={"thinking": { "type": "disabled"}},   # Claude
)
```


### Other LLMs

You can use other LLMs by using `LiteLLMService` or implementing `LLMService` interface.

See the details of LiteLLM here: https://github.com/BerriAI/litellm


### Voice Text Tag (Think Before Answering)

By setting `voice_text_tag`, you can have the LLM "think before answering" (Chain-of-Thought) while vocalizing only the answer portion. You can specify a single tag or a list of tags.

```python
# Single tag: vocalize only <answer> content
llm = ChatGPTService(
    system_prompt="Think within <think> tags. Write your answer within <answer> tags.",
    voice_text_tag="answer"
)

# Multiple tags: vocalize both <ack> and <answer>, skip <think>
llm = ChatGPTService(
    system_prompt="Output <ack>first reaction</ack><think>reasoning</think><answer>full response</answer>",
    voice_text_tag=["ack", "answer"]
)
```


## 🗣️　Speech Synthesizer

You can use a variety of text-to-speech (TTS) services as `SpeechSynthesizer` components.

### VOICEVOX

[VOICEVOX](https://voicevox.hiroshiba.jp) is a free text-to-speech engine popular for its wide variety of Japanese character voices. AIAvatarKit uses `VoicevoxSpeechSynthesizer` by default, and you can configure the speaker ID and VOICEVOX server URL directly when initializing an adapter.

```python
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key="YOUR_OPENAI_API_KEY",
    # 46 is Sayo. See http://127.0.0.1:50021/speakers to get all ids for characters
    voicevox_speaker=46,
    voicevox_url="http://127.0.0.1:50021",
)
```

For more control, create a `VoicevoxSpeechSynthesizer` instance with custom parameters.

```python
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer

tts = VoicevoxSpeechSynthesizer(
    base_url="http://127.0.0.1:50021",
    speaker=46,  # Sayo; see /speakers on your VOICEVOX server for other IDs
    timeout=10.0,
    cache_dir="./tts_cache/voicevox",
    debug=True,
)
```

[AivisSpeech](https://aivis-project.com) can also be used with `VoicevoxSpeechSynthesizer` because it provides a VOICEVOX-compatible API.

```python
tts = VoicevoxSpeechSynthesizer(
    base_url="http://127.0.0.1:10101",  # Your AivisSpeech API server
    speaker="888753761",  # Anneli
)
```

### Azure

```python
from aiavatar.sts.tts.azure import AzureSpeechSynthesizer

tts = AzureSpeechSynthesizer(
    azure_api_key="YOUR_AZURE_API_KEY",
    azure_region="YOUR_AZURE_REGION",
    speaker="ja-JP-NanamiNeural",
)
```

### Google

```python
from aiavatar.sts.tts.google import GoogleSpeechSynthesizer

tts = GoogleSpeechSynthesizer(
    google_api_key="YOUR_GOOGLE_API_KEY",
    speaker="ja-JP-Neural2-B",
)
```

### OpenAI

```python
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer

tts = OpenAISpeechSynthesizer(
    openai_api_key="YOUR_OPENAI_API_KEY",
    speaker="sage",
    audio_format="wav",
    sample_rate=16000,
)
```

`sample_rate` selects the final sample rate for synthesized audio. The built-in postprocessor resamples PCM WAV output in-process without ffmpeg. For other formats, add a compatible `TTSPostprocessor`.

Irodori-TTS can also be used through `OpenAISpeechSynthesizer` with [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server/), an OpenAI Text-to-Speech API-compatible server.

### SpeechGateway

[SpeechGateway](https://github.com/uezo/speech-gateway) provides a unified API for speech synthesis, along with features such as response caching and language-based routing across TTS services.

```python
from aiavatar.sts.tts.speech_gateway import SpeechGatewaySpeechSynthesizer

tts = SpeechGatewaySpeechSynthesizer(
    service_name="sbv2",
    speaker="0-0",
    tts_url="http://127.0.0.1:8000/tts",
)
```

### Instant TTS Synthesizer

For quick setup of custom TTS services with HTTP API endpoints, use `create_instant_synthesizer`. This allows you to create a TTS synthesizer with just HTTP request parameters.

Examples:

```python
from aiavatar.sts.tts import create_instant_synthesizer

# Style-Bert-VITS2 API
sbv2_tts = create_instant_synthesizer(
    method="POST",
    url="http://127.0.0.1:5000/voice",
    json={
        "model_id": "0",
        "speaker_id": "0",
        "text": "{text}"  # Placeholder for processed text
    }
)

# ElevenLabs
elevenlabs_tts = create_instant_synthesizer(
    method="POST",
    url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
    headers={
        "xi-api-key": ELEVENLABS_API_KEY
    },
    json={
        "text": "{text}",
        "model_id": "eleven_v3",
        "output_format": "pcm_16000"
    }
)

# Aivis Cloud API
from aiavatar.sts.tts import AudioConverter
aivis_tts = create_instant_synthesizer(
    method="POST",
    url="https://api.aivis-project.com/v1/tts/synthesize",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIVIS_API_KEY}"
    },
    json={
        "model_uuid": "22e8ed77-94fe-4ef2-871f-a86f94e9a579",   # Kohaku
        "text": "{text}"
    },
    response_parser=AudioConverter(debug=True).convert
)

# Kotodama API (Implement `make_request` to apply style or language.)
import base64
async def base64_to_bytes(http_response) -> bytes:
    response_json = http_response.json()
    b64audio = response_json["audios"][0]
    return base64.b64decode(b64audio)

kotodama_tts = create_instant_synthesizer(
    method="POST",
    url=f"https://tts3.spiral-ai-app.com/api/tts_generate",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": KOTODAMA_API_KEY
    },
    json={
        "text": "{text}",
        "speaker_id": "Marlo",
        "decoration_id": "neutral",
        "audio_format": "wav"
    },
    response_parser=base64_to_bytes
)

# Coefont
import hmac
import hashlib

def make_coefont_request(text: str, style_info: dict, language: str):
    date = str(int(datetime.now(tz=timezone.utc).timestamp()))

    data = json.dumps({
        "coefont": "33e0a2ff-5050-434c-9506-defe97e52f15",  # Yuko Goto
        "text": text
    })

    signature = hmac.new(
        key=bytes(COEFONT_ACCESS_SECRET, "utf-8"),
        msg=(date+data).encode("utf-8"),
        digestmod=hashlib.sha256
    ).hexdigest()

    return httpx.Request(
        method="post",
        url="https://api.coefont.cloud/v2/text2speech",
        headers={
            "Content-Type": "application/json",
            "Authorization": COEFONT_ACCESS_KEY,
            "X-Coefont-Date": date,
            "X-Coefont-Content": signature
        },
        data=data
    )

tts = create_instant_synthesizer(
    request_maker=make_coefont_request,
    follow_redirects=True
)

# Amazon Polly (AWS)
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

region = "ap-northeast-1"
voice_id = "Mizuki"

session = boto3.Session()
# Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as environment variables
credentials = session.get_credentials().get_frozen_credentials()

convert_pcm_to_wave = AudioConverter(input_sample_rate=16000).pcm_to_wave

def aws_polly_request_maker(text, style_info=None, language=None):
    url = f"https://polly.{region}.amazonaws.com/v1/speech"
    body = json.dumps({
        "OutputFormat": "pcm",
        "SampleRate": "16000",
        "Text": text,
        "VoiceId": voice_id,
    })
    aws_request = AWSRequest(method="POST", url=url, data=body, headers={"Content-Type": "application/json"})
    SigV4Auth(credentials, "polly", region).add_auth(aws_request)
    return httpx.Request(method="POST", url=url, headers=dict(aws_request.headers), content=body)

tts = create_instant_synthesizer(
    request_maker=aws_polly_request_maker,
    response_parser=convert_pcm_to_wave,
)

# COEIROINK
tts = create_instant_synthesizer(
    method="POST",
    url="http://127.0.0.1:50032/v1/synthesis",
    headers={"Content-Type": "application/json"},
    json={
        "speakerUuid": "3c37646f-3881-5374-2a83-149267990abc",  # Tsukuyomi-chan
        "styleId": 0,
        "text": "{text}",
        "volumeScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "prePhonemeLength": 0.0,
        "postPhonemeLength": 0.0,
        "outputSamplingRate": 16000,
        "speedScale": 1.0,
    },
    cache_dir="ttscache/coeiroink/tsukuyomi-chan",
)
```

The `{text}` and `{language}` placeholders in params, headers, and json will be automatically replaced with the processed text and language values during synthesis.


You can also make custom TTS components by implementing the `SpeechSynthesizer` interface. The base `synthesize()` method handles empty text, preprocessing, caching, and postprocessing, so a minimal implementation only needs to provide `generate()`. The default cache key uses the synthesizer class, processed text, style information, and language. Override `make_synthesis_cache_key()` when synthesis also depends on provider-specific request settings such as the model, speaker, or speed. The `text` passed to both methods has already been preprocessed.

### TTS Routing

Use `SpeechSynthesizerRouter` to select one of several TTS synthesizers for each request. Register a synchronous routing function with `@tts.route`; it receives the text, style information, and language, and returns a key from the `synthesizers` mapping.

```python
import os

from aiavatar.sts.tts import SpeechSynthesizerRouter
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer

japanese_tts = VoicevoxSpeechSynthesizer(
    sample_rate=16000,
    cache_dir="./tts_cache/voicevox",
)
multilingual_tts = OpenAISpeechSynthesizer(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    sample_rate=16000,
    cache_dir="./tts_cache/openai",
)

tts = SpeechSynthesizerRouter(
    synthesizers={
        "ja": japanese_tts,
        "multi": multilingual_tts,
    },
    default="ja",
)

@tts.route
def select_tts(text, style_info, language):
    if not language:
        return None  # Use the default route
    if language.lower().split("-", 1)[0] == "ja":
        return "ja"
    return "multi"
```

The router only selects and delegates. Preprocessors, postprocessors, sample-rate conversion, and caching belong to each registered synthesizer and run after routing. This allows language-specific processing, such as applying `AlphabetToKanaPreprocessor` only to the Japanese synthesizer. Cached audio is also stored and read by the selected synthesizer rather than by the router.

### TTS Caching

All TTS synthesizers support optional response caching. When `cache_dir` is set, synthesized audio is saved to disk and reused for identical requests, avoiding redundant API calls.

```python
tts = AzureSpeechSynthesizer(
    azure_api_key=AZURE_API_KEY,
    azure_region=AZURE_REGION,
    speaker="ja-JP-MayuNeural",
    cache_dir="./tts_cache/azure",  # Enable caching
    cache_ext="wav",                # File extension (default: "wav")
)
```

- Cache files are stored as `{sha256_hash}.{cache_ext}` in the specified directory
- The hash is computed from all request parameters (URL, headers, body, etc.)
- Set `cache_dir=None` (default) to disable caching
- Works with all built-in TTS classes, including SpeechGateway and InstantSynthesizer

### Preprocessing

AIAvatarKit provides text preprocessing functionality that transforms text before Text-to-Speech processing. This enables improved speech quality and conversion of specific text patterns.

#### Alphabet to Katakana Conversion

A preprocessor that converts alphabet text to katakana using LLM. Supports kana_map for storing word-to-reading mappings to reduce latency on repeated words.

```python
from aiavatar.sts.tts.preprocessor.alphabet2kana import AlphabetToKanaPreprocessor

# Create preprocessor with kana_map for pre-registered word-reading mappings
alphabet2kana_preproc = AlphabetToKanaPreprocessor(
    openai_api_key=OPENAI_API_KEY,
    alphabet_length=3,                        # Minimum alphabet length to convert (default: 3)
    special_chars=".'-'−–",                   # Characters that connect words (default: ".'-'−–")
    use_kana_map=True,                        # Enable kana_map mode (default: True)
    kana_map={"GitHub": "ギットハブ"},         # Pre-registered word-reading mappings (optional)
    debug=True,                               # Enable debug logging (default: False)
)

# Add to TTS
tts.preprocessors.append(alphabet2kana_preproc)

# Words converted by LLM are automatically added to kana_map
# You can persist and restore kana_map for future sessions:
import json
# Save
with open("kana_map.json", "w") as f:
    json.dump(alphabet2kana_preproc.kana_map, f, ensure_ascii=False)
# Load
with open("kana_map.json") as f:
    kana_map = json.load(f)
```

Preprocessors may optionally accept a keyword-only `synthesizer` argument to access shared TTS configuration such as `sample_rate`. Existing preprocessors with the original `process(text, style_info, language)` signature remain supported.

Key features:
- **kana_map**: Pre-register known word-reading mappings and automatically add LLM results to avoid repeated API calls
- **special_chars**: Words containing these characters (e.g., `Mr.`, `You're`, `Wi-Fi`) are always processed regardless of `alphabet_length`
- **Case-insensitive**: Matches `API`, `api`, and `Api` with a single kana_map entry
- **debug mode**: Logs `[KanaMap]` for cached hits and `[LLM]` for new readings with elapsed time

#### Pattern Match Conversion

You can also use regular expressions and string patterns for conversion:

```python
from aiavatar.sts.tts.preprocessor.patternmatch import PatternMatchPreprocessor

# Create pattern match preprocessor
pattern_preproc = PatternMatchPreprocessor(patterns=[
    ("API", "エーピーアイ"),               # Fixed string replacement
    ("URL", "ユーアールエル"),
    (r"\d+", lambda m: "number"),          # Regex replacement with function
])

# Add common patterns
pattern_preproc.add_number_dash_pattern()  # Number-dash patterns (e.g., 12-34 → イチニの サンヨン)
pattern_preproc.add_phonenumber_pattern()  # Phone number patterns

# Add to TTS
tts.preprocessors.append(pattern_preproc)
```

#### Creating Custom Preprocessors

You can create your own preprocessors by implementing the `TTSPreprocessor` interface:

```python
from aiavatar.sts.tts.preprocessor import TTSPreprocessor

class CustomPreprocessor(TTSPreprocessor):
    def __init__(self, custom_dict: dict = None):
        self.custom_dict = custom_dict or {}
    
    async def process(self, text: str, style_info: dict = None, language: str = None) -> str:
        # Custom conversion logic
        processed_text = text
        
        # Dictionary-based replacement
        for original, replacement in self.custom_dict.items():
            processed_text = processed_text.replace(original, replacement)
        
        # Language-specific conversions
        if language == "ja-JP":
            processed_text = processed_text.replace("OK", "オーケー")
        
        return processed_text

# Use custom preprocessor
custom_preproc = CustomPreprocessor(custom_dict={
    "GitHub": "ギットハブ",
    "Python": "パイソン",
    "Docker": "ドッカー"
})

tts.preprocessors.append(custom_preproc)
```

#### Combining Preprocessors

Multiple preprocessors can be used together. They are executed in the order they were registered:

```python
# Combine multiple preprocessors
tts.preprocessors.extend([
    pattern_preproc,        # 1. Pattern match conversion
    alphabet2kana_preproc,  # 2. Alphabet to katakana conversion
    custom_preproc          # 3. Custom conversion
])
```


### Adjusting Speech Speed

With `SpeechGatewaySpeechSynthesizer`, you can change the speech speed per session by setting the speed either on the entire instance or in `style_info`.

Here is an example of storing the speech speed as `tts_speed` in session data when using WebSocketAdapter.

```python
# Apply speech speed per session
from aiavatar.sts.llm import LLMResponse
@aiavatar_app.sts.process_llm_chunk
async def process_llm_chunk(llm_stream_chunk: LLMResponse, session_id: str, user_id: str) -> dict:
    if session_data := aiavatar_app.sessions.get(session_id):
        if speed := session_data.data.get("tts_speed"):
            return {"speed": float(speed)}
```

NOTE: To configure `tts_speed`, you can either set up a REST API endpoint to update it directly, or use control tags included in responses to update it.


## 👂 Speech listener

If you want to configure in detail, create instance of `SpeechRecognizer` with custom parameters and set it to `AIAvatar`. We support Azure, Google and OpenAI Speech-to-Text services.

NOTE: **`AzureSpeechRecognizer` is much faster** than Google and OpenAI(default).

```python
# Create AzureSpeechRecognizer
from aiavatar.sts.stt.azure import AzureSpeechRecognizer
stt = AzureSpeechRecognizer(
    azure_api_key=AZURE_API_KEY,
    azure_region=AZURE_REGION
)

# Create AIAvatar with AzureSpeechRecognizer
aiavatar_app = AIAvatar(
    stt=stt,
    openai_api_key=OPENAI_API_KEY   # API Key for LLM
)
```

You can also make custom STT components by implementing `SpeechRecognizer` interface.

### Preprocessing and Postprocessing

You can add custom preprocessing and postprocessing to any `SpeechRecognizer` implementation. This is useful for tasks like speaker verification, audio filtering, or text normalization.

```python
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

# Create recognizer
recognizer = OpenAISpeechRecognizer(openai_api_key="your-api-key")

# Add preprocessing - e.g., speaker verification
@recognizer.preprocess
async def verify_speaker(session_id: str, audio_data: bytes):
    # Perform speaker verification
    is_valid_speaker = await check_speaker_identity(audio_data)
    
    if not is_valid_speaker:
        # Return empty bytes to skip transcription
        return b"", {"rejected": True, "reason": "speaker_mismatch"}
    
    # Return processed audio and metadata
    filtered_audio = apply_noise_filter(audio_data)
    return filtered_audio, {"speaker_verified": True, "session_id": session_id}

# Add postprocessing - e.g., text formatting
@recognizer.postprocess
async def format_text(session_id: str, text: str, audio_data: bytes, preprocess_metadata: dict):
    # Format transcribed text
    formatted_text = text.strip().capitalize()
    
    # Add punctuation if missing
    if formatted_text and formatted_text[-1] not in '.!?':
        formatted_text += '.'
    
    # Return formatted text and metadata
    return formatted_text, {
        "original_text": text,
        "formatting_applied": True,
        "preprocess_info": preprocess_metadata
    }

# Use the recognizer with preprocessing and postprocessing
result = await recognizer.recognize(
    session_id="user-123",
    data=audio_bytes
)

print(f"Text: {result.text}")
print(f"Preprocess metadata: {result.preprocess_metadata}")
print(f"Postprocess metadata: {result.postprocess_metadata}")
```

The preprocessing and postprocessing functions can return either:
- Just the processed data (bytes for preprocess, string for postprocess)
- A tuple of (processed_data, metadata_dict) for additional information

If preprocessing returns empty bytes, the transcription is skipped and the result will have `text=None`.


### Speaker Diarization

AIAvatarKit provides speaker diarization functionality to suppress responses to voices other than the main speaker. This prevents interruptions from surrounding conversations or venue announcements at events.

The `MainSpeakerGate` provides the following features:

- Calculates voice embeddings from request audio
- Registers a voice as the main speaker when similarity exceeds threshold for 2 consecutive requests (per session)
- Returns `accepted=True` when request audio similarity exceeds threshold after main speaker registration
- Returns `accepted=True` when no main speaker is registered yet

**NOTE:** While mechanically ignoring non-main speaker voices (Example 1) is simplest, it risks stopping conversation due to misidentification and cannot handle speaker changes. Consider context-aware handling (Example 2) as well.

```python
from aiavatar.sts.stt.speaker_gate import MainSpeakerGate
speaker_gate = MainSpeakerGate()

# Example 1: Drop request when the voice is not from main speaker
@aiavatar_app.sts.stt.preprocess
async def stt_preprocess(session_id: str, audio_bytes: bytes):
    # Compare with main speaker's voice embedding
    gate_response = await speaker_gate.evaluate(session_id, audio_bytes, aiavatar_app.sts.vad.sample_rate)
    # Branch processing based on similarity with main speaker's voice
    if not gate_response.accepted:
        logger.info(f"Ignore other speaker's voice: confidence={gate_response.confidence}")
        return None, gate_response.to_dict()
    else:
        return audio_bytes, gate_response.to_dict()

# Example 2: Add annotation for LLM that the voice is not from main speaker
@aiavatar_app.sts.stt.postprocess
async def stt_postprocess(session_id: str, text: str, audio_bytes: bytes, preprocess_metadata: dict):
    # Compare with main speaker's voice embedding
    gate_response = await speaker_gate.evaluate(session_id, audio_bytes, aiavatar_app.sts.vad.sample_rate)
    # Branch processing based on similarity with main speaker's voice
    if not gate_response.accepted:
        logger.info(f"Adding note that this may be from a different speaker: confidence={gate_response.confidence}")
        return f"$The following request may not be from the main speaker (similarity: {gate_response.confidence}). Determine from the content whether to respond. If you should not respond, output just[wait:user] as the answer:\n\n{text}", gate_response.to_dict()
    else:
        return text, gate_response.to_dict()
```


### Per-session STT Switching

You can override the STT engine for a specific session without changing the default engine used by other sessions. Batch VAD and stream VAD perform recognition in different places, so use the API belonging to the component that performs STT:

- For batch VAD, use `STSPipeline.set/get/clear_speech_recognizer()`.
- For `SileroStreamSpeechDetector` and its echo-cancelling variant, use the detector's `set/get/clear_speech_recognizer()` methods.

The examples below ask the LLM to prefix questions whose answers may contain proper nouns with `<require_noun />`. The tag is detected before TTS, so the specialized STT is already active if the user barges in while the question is being spoken.

```python
from aiavatar.sts.llm import LLMResponse
from aiavatar.sts.stt.amivoice import AmiVoiceSpeechRecognizer

REQUIRE_NOUN_TAG = "<require_noun />"

# Create this once and share it across sessions. Do not mutate the engine on a
# shared recognizer because concurrent sessions may require different engines.
proper_noun_stt = AmiVoiceSpeechRecognizer(
    amivoice_api_key=AMIVOICE_API_KEY,
    engine="YOUR_PROPER_NOUN_ENGINE",
)

# Prompt convention for the LLM:
# When the next user response may contain a place or other proper noun, prefix
# the response with <require_noun />.
# Example: <require_noun />Where are you traveling for summer vacation?
```

#### Batch VAD

Batch recognition is performed by `STSPipeline`, which owns the per-session override:

```python
# aiavatar_app.sts.get_speech_recognizer(session_id) returns the override,
# or the pipeline's default STT when no override is set.

@aiavatar_app.sts.process_llm_chunk
async def switch_stt_for_travel_question(
    llm_chunk: LLMResponse,
    session_id: str,
    user_id: str,
) -> dict:
    if REQUIRE_NOUN_TAG in (llm_chunk.text or ""):
        aiavatar_app.sts.set_speech_recognizer(
            session_id,
            proper_noun_stt,
        )
    return {}


@aiavatar_app.sts.on_finish
async def restore_default_stt(request, response):
    # Keep the specialized STT after the tagged travel question. The response
    # generated after the user's answer normally has no tag, so it restores the
    # default STT for the following turn.
    if REQUIRE_NOUN_TAG not in (response.text or ""):
        aiavatar_app.sts.clear_speech_recognizer(request.session_id)
```

#### Stream VAD

Stream recognition is performed inside the VAD. Call the `SileroStreamSpeechDetector` instance that was passed to the app. `EchoCancellingSileroStreamSpeechDetector` uses the same API:

```python
# stream_vad is the SileroStreamSpeechDetector instance passed to AIAvatar.
# stream_vad.get_speech_recognizer(session_id) returns the override,
# or the stream detector's default STT when no override is set.

@aiavatar_app.sts.process_llm_chunk
async def switch_stream_stt_for_travel_question(
    llm_chunk: LLMResponse,
    session_id: str,
    user_id: str,
) -> dict:
    if REQUIRE_NOUN_TAG in (llm_chunk.text or ""):
        stream_vad.set_speech_recognizer(
            session_id,
            proper_noun_stt,
        )
    return {}


@aiavatar_app.sts.on_finish
async def restore_default_stream_stt(request, response):
    if REQUIRE_NOUN_TAG not in (response.text or ""):
        stream_vad.clear_speech_recognizer(request.session_id)
```

Switching back is intentionally controlled by user code. Batch overrides remain in the pipeline until `aiavatar_app.sts.clear_speech_recognizer()` is called, so also clear them from a `finally` block or an adapter disconnect callback if a session can end before `on_finish` runs. Stream overrides are stored in the VAD recording session and are naturally discarded when that session is deleted, but explicit cleanup is still recommended for consistent application behavior.


## 🎙️ Speech Detector

AIAvatarKit includes Voice Activity Detection (VAD) components to automatically detect when speech starts and ends in audio streams. This enables seamless conversation flow without manual input controls.

### Silero Speech Detector

The default Speech Detector is `SileroSpeechDetector`, which employs AI-based voice activity detection using the Silero VAD model:

```python
from aiavatar.sts.vad.silero import SileroSpeechDetector

vad = SileroSpeechDetector(
    speech_probability_threshold=0.5,    # AI model confidence threshold (0.0-1.0)
    silence_duration_threshold=0.5,      # Seconds of silence to end recording
    volume_db_threshold=None,            # Optional: filter by volume in dB (e.g., -30.0)
    max_duration=10.0,                   # Maximum recording duration
    min_duration=0.2,                    # Minimum recording duration
    sample_rate=16000,                   # Audio sample rate
    channels=1,                          # Audio channels
    chunk_size=512,                      # Audio processing chunk size
    model_pool_size=1,                   # Number of parallel AI models
    debug=True
)

aiavatar_app = AIAvatar(vad=vad, openai_api_key=OPENAI_API_KEY)
```

For high-concurrency applications:

```python
vad = SileroSpeechDetector(
    speech_probability_threshold=0.6,    # Stricter threshold for noisy environments
    model_pool_size=4,                   # 4 parallel AI models for load balancing
    debug=False
)
```

To use a local Silero VAD hub repository/cache instead of downloading from the network, set `hub_cache_path`.
The path must point to a Silero VAD repository/cache directory that contains `hubconf.py`:

```python
vad = SileroSpeechDetector(
    hub_cache_path="/models/silero-vad"
)
```

When `hub_cache_path` is set, the detector loads the Silero model and utility functions from that local path only.
If the path does not exist, initialization fails instead of downloading from the network.

For custom JIT model files, `model_path` is still supported. In that case, the detector first loads the Silero utilities from `hub_cache_path` or the default Torch Hub source, then replaces the model with the local JIT file:

```python
vad = SileroSpeechDetector(
    hub_cache_path="/models/silero-vad",
    model_path="path/to/silero_vad.jit"
)
```


### Silero Stream Speech Detector

`SileroStreamSpeechDetector` extends `SileroSpeechDetector` with segment-based speech recognition. It performs partial transcription during recording, allowing you to receive intermediate results before the final transcription.

```python
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.stt.google import GoogleSpeechRecognizer

vad = SileroStreamSpeechDetector(
    speech_recognizer=GoogleSpeechRecognizer(...),
    segment_silence_threshold=0.2,       # Silence duration to trigger segment recognition
    silence_duration_threshold=0.5,      # Silence duration to finalize recording
    # Inherits all SileroSpeechDetector parameters
)
```

#### Segment Recognition Callback

The `on_speech_detecting` callback is triggered when a speech segment is recognized:

```python
@vad.on_speech_detecting
async def on_speech_detecting(text, session):
    print(f"Partial text: {text}")

    # For WebSocket apps, send partial text to client via info message
    # resp = STSResponse(
    #     type="info",
    #     session_id=session.session_id,
    #     metadata={"partial_request_text": text}
    # )
    # await ws_app.handle_response(resp)
```

#### Text Validation

Use `validate_recognized_text` to filter out invalid recognition results:

```python
@vad.validate_recognized_text
def validate(text):
    if len(text) < 2:
        return "Text too short"  # Return error message to reject
    return None  # Return None to accept
```


### Semantic VAD

AIAvatarKit supports semantic VAD by combining acoustic VAD with optional turn-end gates. The acoustic VAD first detects a turn-end candidate from silence, then turn-end gates inspect audio, recognized text, or model-specific signals to decide whether the user's utterance is semantically complete.

`SileroSpeechDetector` and `SileroStreamSpeechDetector` can use built-in gates such as Smart Turn, Filler-only, Namo Turn, and LLM-based gates. You can also implement your own `TurnEndGate` when you need domain-specific turn-end logic. Gates are called only after `silence_duration_threshold` has already been reached. All gates must pass to end the turn. If any gate returns "wait", the detector keeps the current recording open until the user resumes speaking or the waiting gate's timeout forces the turn to end.

This is useful for utterances that contain a short pause but are likely to continue, such as trailing conjunctions, filler phrases, or incomplete requests.

```python
vad = SileroSpeechDetector(
    silence_duration_threshold=0.5,
    turn_end_gates=[my_gate],
)
```

Gate timeouts are measured after `silence_duration_threshold` has been reached. For example, with `silence_duration_threshold=0.5` and a gate timeout of `2.0`, the longest silence wait is approximately 2.5 seconds.

Gate coordination is handled by `TurnEndGateManager`. VAD detectors only ask the manager whether the current turn-end candidate should end or keep recording. The manager keeps per-session wait state, passes previous gate decisions through `TurnEndGateContext`, and uses the longest timeout among gates that returned wait. If the detector reaches `max_duration`, it still ends the turn even when a gate is holding it, so gate waits cannot keep the recording buffer open forever.

Gates can opt into background execution by setting `run_in_background=True`. Background gates do not block audio processing while they are pending. While pending, their `timeout` is used as a provisional wait timeout, so a detector can be configured with only background gates. When a background gate finishes, its result replaces the provisional pending decision. If the result is still pending when the timeout expires, it is ignored and the turn ends.

`SmartTurnEndGate` and `NamoTurnEndGate` use one ONNX Runtime session per gate instance and serialize inference with an internal lock. This is fine for typical usage because gates run only after a VAD turn-end candidate, not for every audio chunk. For very high concurrency, create separate gate instances per detector or worker process, or add a small gate/session pool if turn-end gate latency becomes visible.

#### Smart Turn Gate

`SmartTurnEndGate` uses [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) to classify the current recorded audio as complete or incomplete.

```sh
pip install "aiavatar[smart-turn]"
```

```python
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.turn_end_gates.smart_turn import SmartTurnEndGate

turn_end_gate = SmartTurnEndGate(
    threshold=0.5,
    timeout=1.5,
    debug=True,
)

vad = SileroSpeechDetector(
    silence_duration_threshold=0.5,
    turn_end_gates=[turn_end_gate],
    debug=True,
)
```

To use a local Smart Turn ONNX model instead of downloading from Hugging Face, set `model_path`:

```python
turn_end_gate = SmartTurnEndGate(
    model_path="/models/smart-turn-v3.2-cpu.onnx",
)
```

#### Filler-Only Gate

`FillerOnlyTurnEndGate` waits longer when the recognized text is only a filler phrase, or ends with a trailing filler phrase, such as "えっと", "あの", "um", or "uh". It normalizes text before matching, so spaces, punctuation, and symbols are ignored; for example, "えっと。" matches "えっと". One-character fillers such as "あ" are not used for trailing-filler matching, and short replies that can be meaningful answers, such as "うん", are not included in the default filler list.

This gate is most useful with `SileroStreamSpeechDetector`, because it needs recognized text.

```python
from aiavatar.sts.vad.turn_end_gates import FillerOnlyTurnEndGate, FillerPhrase

filler_gate = FillerOnlyTurnEndGate(
    name="filler",
    fillers=[
        FillerPhrase("あの", match="suffix", timeout=6.0),
        FillerPhrase("えっと", match="suffix"),
        "um",  # str means exact match
    ],
    timeout=5.0,
    debug=True,
)
```

#### Namo Turn Gate

`NamoTurnEndGate` uses [videosdk-live/NAMO-Turn-Detector-v1](https://github.com/videosdk-live/NAMO-Turn-Detector-v1) to classify recognized text as end-of-turn or not-end-of-turn. It is most useful with `SileroStreamSpeechDetector`, because the stream detector can pass accumulated partial recognition text to the gate.

```sh
pip install "aiavatar[namo-turn]"
```

```python
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_end_gates.filler import FillerOnlyTurnEndGate
from aiavatar.sts.vad.turn_end_gates.namo_turn import NamoTurnEndGate

filler_gate = FillerOnlyTurnEndGate(
    name="filler",
    timeout=5.0,
)

turn_end_gate = NamoTurnEndGate(
    name="namo",
    language="ja",   # Japanese model. Use language=None for the multilingual model.
    threshold=0.5,
    force_end_phrases=["こんにちは"],
    timeout=1.5,
    debug=True,
)

vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    segment_silence_threshold=0.05,
    silence_duration_threshold=0.5,
    turn_end_gates=[
        filler_gate,
        turn_end_gate,
    ],
    debug=True,
)
```

`threshold` is the minimum predicted probability of class 1 ("End of Turn"). Higher values require stronger evidence before ending the turn, so they hold the turn more often.

`force_end_phrases` is an optional list of exact utterances that should end the turn without running the model. Matching ignores case, whitespace, full-width variants, punctuation, and symbols, so `"こんにちは。"` matches `"こんにちは"`. Longer utterances such as `"こんにちは、今日は相談があります"` do not match.

For long recordings, Namo keeps the end of the recognized text when tokenized text exceeds the model limit, because turn-end detection depends most on the final words. If no text is available, `NamoTurnEndGate` defaults to ending the turn. You can change this with `no_text_should_end=False`.

To run Namo from local files without downloading from Hugging Face, set both `model_path` and `tokenizer_path`:

```python
turn_end_gate = NamoTurnEndGate(
    language="ja",
    model_path="/models/namo/model_quant.onnx",
    tokenizer_path="/models/namo/tokenizer",
)
```

#### LLM Turn Gate

`LLMTurnEndGate` uses an OpenAI-compatible Chat Completions client to make a slower but more flexible text-based decision. It is useful as a second-stage gate after a cheaper gate has already decided to wait. It runs in the background by default, so the current audio receive loop is not blocked while waiting for the LLM response.

Pass a long-lived client instance to the constructor. The gate reuses that client instead of creating one per decision, so the underlying HTTP connection pool can be reused.

```python
from openai import AsyncOpenAI

from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_end_gates import FillerOnlyTurnEndGate
from aiavatar.sts.vad.turn_end_gates.llm import LLMTurnEndGate
from aiavatar.sts.vad.turn_end_gates.namo_turn import NamoTurnEndGate

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

filler_gate = FillerOnlyTurnEndGate(
    name="filler",
    timeout=5.0,
)

namo_gate = NamoTurnEndGate(
    name="namo",
    language="ja",
    threshold=0.5,
    timeout=1.5,
)

llm_gate = LLMTurnEndGate(
    openai_client=openai_client,
    model="gpt-4.1-mini",
    depends_on=["filler", "namo"],
    timeout=10.0,
    request_timeout=2.0,
    debug=True,
)

vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    silence_duration_threshold=0.5,
    turn_end_gates=[
        filler_gate,
        namo_gate,
        llm_gate,
    ],
)
```

When `depends_on` is set, the LLM gate runs only if one of the named previous gates returned wait. The value can be a string or a list of gate names. In the example above, normal utterances do not call the LLM; the LLM is only called when the filler or Namo gate waits. The primary gate first holds the turn for its own timeout. If the LLM finishes during that wait and returns wait, the manager extends the wait to the LLM timeout, so the `10.0` second LLM timeout takes priority. If the LLM is still pending at the primary gate timeout, the pending LLM result is ignored and the turn ends.

`LLMTurnEndGate` accepts `temperature` and `reasoning_effort`, but only passes them to the API when they are explicitly set. Use the option supported by the model you choose.

#### Session Hold Gate

`SessionHoldTurnEndGate` lets application logic hold the next turn-end candidate for a specific session. Unlike gates that inspect the current audio or recognized text, this gate is armed in advance when the application knows that the next answer may need more thinking time.

For example, a restaurant-search assistant may ask the user to choose a cuisine, describe their preferences, or recall an area or budget. These answers often begin with hesitation and contain pauses, such as "Well... maybe Italian." The assistant can prefix questions like these with a control tag such as `<require_restaurant_preferences />`. When the tag is detected, the session hold gate allows a longer pause for the next user answer instead of ending the turn at the normal silence threshold.

```python
from aiavatar.sts.llm import LLMResponse
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.turn_end_gates.session_hold import SessionHoldTurnEndGate

REQUIRE_PREFERENCES_TAG = "<require_restaurant_preferences />"

session_hold_gate = SessionHoldTurnEndGate(debug=True)

vad = SileroSpeechDetector(
    silence_duration_threshold=0.5,
    turn_end_gates=[session_hold_gate],
    debug=True,
)

# Pass vad to AIAvatar when creating aiavatar_app.

# Prompt convention for the LLM:
# Prefix a question with <require_restaurant_preferences /> when its answer may
# require the user to recall details, compare options, or think aloud.
# Example:
# <require_restaurant_preferences />What kind of food are you in the mood for?

@aiavatar_app.sts.process_llm_chunk
async def hold_restaurant_preference_answer(
    llm_chunk: LLMResponse,
    session_id: str,
    user_id: str,
) -> dict:
    if REQUIRE_PREFERENCES_TAG in (llm_chunk.text or ""):
        session_hold_gate.hold(
            session_id,
            timeout=3.0,
            reason="restaurant_preferences",
        )
    return {}
```

The tag is detected while the LLM response is streaming, before the next user turn, and AIAvatarKit removes control tags such as this one from the synthesized voice text. With the settings above, the normal turn-end candidate is detected after 0.5 seconds of silence, and the armed gate can keep the recording open for up to 3.0 additional seconds. The hold is consumed by that candidate; subsequent turns use the normal silence threshold unless another tagged response arms the gate again.

#### Custom Gate

Implement `TurnEndGate` to plug in your own decision logic. Gates receive the current recorded audio, timing information, the session id, recognized text when available, and a context containing previous gate decisions.

```python
from aiavatar.sts.vad.turn_end_gates import TurnEndDecision, TurnEndGate, TurnEndGateContext

class MyTurnEndGate(TurnEndGate):
    async def should_end_turn(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        channels: int,
        recorded_duration: float,
        silence_duration: float,
        session_id: str,
        text: str | None = None,
        session=None,
        context: TurnEndGateContext | None = None,
    ) -> TurnEndDecision:
        if text and text.endswith("and"):
            return TurnEndDecision(should_end=False, confidence=0.9, reason="continues", timeout=3.0)
        return TurnEndDecision(should_end=True, confidence=0.9, reason="complete")
```


### Audio Filters

`SileroSpeechDetector` and `SileroStreamSpeechDetector` can run audio through `audio_filters` before VAD, recording, and speech recognition. Filters are applied in order, and downstream processing sees the filtered audio.

This is useful for acoustic preprocessing such as near-field gating, EQ, gain normalization, and debug recording.

```python
from aiavatar.sts.vad.filters import (
    AGCFilter,
    HighShelfFilter,
    NearFieldAudioGate,
    SessionAudioRecorder,
)
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector

audio_recorder = SessionAudioRecorder("debug_audio")

vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    audio_filters=[
        audio_recorder.tap("raw"),
        NearFieldAudioGate(
            min_rms_db=-42.0,
            open_snr_db_threshold=12.0,
            close_snr_db_threshold=6.0,
        ),
        HighShelfFilter(gain_db=6.0, cutoff_hz=2000.0),
        AGCFilter(target_rms_db=-20.0),
        audio_recorder.tap("processed"),
    ],
)
```

Built-in filters:

- `NearFieldAudioGate`: attenuates far-field or low-SNR audio before it reaches VAD. It uses a short lookahead buffer so speech onsets are not clipped.
- `HighShelfFilter`: boosts or cuts high frequencies above a cutoff. This can help intelligibility on band-limited telephony audio.
- `AGCFilter`: automatic gain control that raises quiet speech toward a target RMS level while avoiding clipping.
- `SessionAudioRecorder`: debug tap that writes audio at selected points in the filter chain to WAV files.

Filter order matters. Put `NearFieldAudioGate` before `AGCFilter`; otherwise AGC may amplify far-field audio and make the gate less useful. `SessionAudioRecorder.tap()` can be placed before and after filters to compare raw and processed audio.

You can implement a custom filter by subclassing `AudioFilter`:

```python
from aiavatar.sts.vad.filters import AudioFilter

class MyAudioFilter(AudioFilter):
    def process(self, samples: bytes, session_id: str) -> bytes:
        # samples are 16-bit linear PCM bytes
        return samples

    def reset_session(self, session_id: str):
        # Optional: release per-session state
        pass
```

Filters may keep short internal buffers and return `b""` while warming up. The detector treats this as "no output yet" and keeps the current recording state unchanged.


### Azure Stream Speech Detector

`AzureStreamSpeechDetector` uses Azure's streaming speech recognition service for both speech detection and transcription. Audio is continuously streamed to Azure, and speech boundaries are determined by Azure's recognition events.

```sh
pip install azure-cognitiveservices-speech
```

```python
from aiavatar.sts.vad.azure_stream import AzureStreamSpeechDetector

vad = AzureStreamSpeechDetector(
    azure_subscription_key=AZURE_API_KEY,
    azure_region=AZURE_REGION
)
```

This detector also supports the `on_speech_detecting` callback for partial transcription results:

```python
@vad.on_speech_detecting
async def on_speech_detecting(text, session):
    print(f"Partial text: {text}")

    # For WebSocket apps, send partial text to client via info message
    # resp = STSResponse(
    #     type="info",
    #     session_id=session.session_id,
    #     metadata={"partial_request_text": text}
    # )
    # await ws_app.handle_response(resp)
```

### AWS Stream Speech Detector

`AmazonTranscribeStreamSpeechDetector` uses Amazon Transcribe's streaming speech recognition service for both speech detection and transcription. Audio is continuously streamed to Amazon Transcribe, and speech boundaries are determined by the recognition results combined with a configurable silence duration threshold.

```sh
pip install amazon-transcribe
```

```python
from aiavatar.sts.vad.amazon_transcribe_stream import AmazonTranscribeStreamSpeechDetector

vad = AmazonTranscribeStreamSpeechDetector(
    aws_region="ap-northeast-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,         # Optional: uses default credential chain if omitted
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,  # Optional: uses default credential chain if omitted
    aws_language="ja-JP",
    silence_duration_threshold=0.5,  # Seconds of silence after last recognition to finalize
    max_duration=20.0,               # Maximum recording duration in seconds
)
```

When `silence_duration_threshold > 0`, multiple recognition results from Amazon Transcribe are accumulated into a single speech detection event. A silence timer starts after each final result, and if new speech arrives before the timer expires, the timer is cancelled and transcription continues. This allows natural pauses within a sentence without splitting the utterance.

> **Note:** The `silence_duration_threshold` timer starts from when Amazon Transcribe returns a final recognition result, not from when the user actually stops speaking. Since Amazon Transcribe takes some time to process audio and return a final result, the actual delay from the user's perspective is: **Transcribe processing delay + `silence_duration_threshold`**. For example, if Transcribe takes ~0.5s to return a final result and `silence_duration_threshold=0.5`, the total delay from the end of speech to firing `on_speech_detected` will be approximately 1.0s.

When `max_duration` is reached during recording, if there are accumulated recognition results, speech detection is triggered immediately with the combined text.

This detector also supports the `on_speech_detecting` callback for partial transcription results. When texts have been accumulated from previous final results, they are prepended to the current partial text:

```python
@vad.on_speech_detecting
async def on_speech_detecting(text, session):
    print(f"Partial text: {text}")

    # For WebSocket apps, send partial text to client via info message
    # resp = STSResponse(
    #     type="info",
    #     session_id=session.session_id,
    #     metadata={"partial_request_text": text}
    # )
    # await ws_app.handle_response(resp)
```

Use `validate_recognized_text` to filter out invalid recognition results:

```python
@vad.validate_recognized_text
def validate(text):
    if len(text) < 2:
        return "Text too short"  # Return error message to reject
    return None  # Return None to accept
```


### Customization

#### on_recording_started Callback

The `on_recording_started` callback is triggered when recording has been active long enough to be considered meaningful speech. This is useful for stopping AI speech when the user starts talking.

```python
# Option 1: Pass callback in constructor
async def my_recording_started_handler(session_id: str):
    print(f"Recording started for session: {session_id}")
    await stop_ai_speech()

vad = SileroSpeechDetector(
    on_recording_started=my_recording_started_handler,
    on_recording_started_min_duration=1.5,    # Trigger after 1.5 sec of speech (default)
    # other parameters...
)

# Option 2: Use decorator
@vad.on_recording_started
async def on_recording_started(session_id):
    await stop_ai_speech()
```

For stream-based detectors (`SileroStreamSpeechDetector`, `AzureStreamSpeechDetector`), the callback can also be triggered by recognized text length:

```python
vad = SileroStreamSpeechDetector(
    speech_recognizer=speech_recognizer,
    on_recording_started_min_duration=1.5,    # Trigger after 1.5 sec of speech
    on_recording_started_min_text_length=2,   # OR trigger when text >= 2 chars
)
```

#### Custom Trigger Condition

You can customize when `on_recording_started` fires using the `should_trigger_recording_started` decorator:

```python
@vad.should_trigger_recording_started
def custom_trigger(text, session):
    # text: Recognized text (None for non-stream detectors)
    # session: Recording session object
    # Return True to trigger the callback
    return text and len(text) >= 5
```


### Standard Speech Detector (Legacy)

`StandardSpeechDetector` uses simple volume-based detection. Consider using `SileroSpeechDetector` for better accuracy. This detector is suitable for environments with limited computing resources:

```python
from aiavatar.sts.vad.standard import StandardSpeechDetector

vad = StandardSpeechDetector(
    volume_db_threshold=-30.0,           # Voice detection threshold in dB
    silence_duration_threshold=0.5,      # Seconds of silence to end recording
    max_duration=10.0,                   # Maximum recording duration
    min_duration=0.2,                    # Minimum recording duration
    sample_rate=16000,                   # Audio sample rate
    channels=1,                          # Audio channels
    preroll_buffer_count=5,              # Pre-recording buffer size
    debug=True
)
```


## 🥰 Face expression

To control facial expressions within conversations, set the facial expression names and values in `FaceController.faces` as shown below, and then include these expression keys in the response message by adding instructions to the prompt.

```python
aiavatar_app.face_controller.faces = {
    "neutral": "🙂",
    "joy": "😀",
    "angry": "😠",
    "sorrow": "😞",
    "fun": "🥳"
}

aiavatar_app.sts.llm.system_prompt = """# Face Expression

* You have the following expressions:

- joy
- angry
- sorrow
- fun

* If you want to express a particular emotion, please insert it at the beginning of the sentence like [face:joy].

Example
[face:joy]Hey, you can see the ocean! [face:fun]Let's go swimming.
"""
```

> **Note:** XML-style tags are also supported: `<face name="joy" />`, `<animation name="wave_hands" />`. Both bracket and XML formats can be used interchangeably.

This allows emojis like 🥳 to be autonomously displayed in the terminal during conversations. To actually control the avatar's facial expressions in a metaverse platform, instead of displaying emojis like 🥳, you will need to use custom implementations tailored to the integration mechanisms of each platform. Please refer to our `VRChatFaceController` as an example.


## 💃 Animation

Now writing... ✍️


## 🖼️ Artifacts

Artifacts let an LLM display an image, chart, presentation, video, web app, or map alongside its spoken response. The Adapter parses registered `<artifact />` control tags, resolves their attributes, and exposes them through `AIAvatarResponse.control_tags`; rendering is implemented by the client. The bundled [WebSocket viewers](examples/websocket/README.md#artifacts-in-the-web-viewers) consume only these structured control tags and do not parse tags from response `text`. Both the standard and 3D viewers support images, charts, presentations, YouTube videos, sandboxed web apps, Google maps, and directions. Artifact tags are excluded from `voice_text`, so they are not read aloud.

Register application-owned artifacts on the Adapter so the LLM only needs to select a short, stable ID:

```python
ARTIFACTS = {
    "about_company": {
        "type": "presentation",
        "src": "https://speakerdeck.com/player/DECK_ID",
        "slide": 1,
        "aspect": "16:9",
        "title": "About the company",
    },
}

aiavatar_app.set_artifacts(ARTIFACTS)
```

The LLM can display the configured artifact with `<artifact id="about_company" />`. Display and navigation attributes supplied by the LLM override configured defaults, so `<artifact id="about_company" slide="5" />` starts at page 5. When an `id` is present, configured `type` and `src` values are protected and cannot be replaced by tag attributes. When the LLM knows a browser-accessible HTTPS URL, it can also display an image with `<artifact type="image" src="https://example.com/image.png" />` or a YouTube video with `<artifact type="video" src="https://www.youtube.com/watch?v=VIDEO_ID" autoplay-delay="3" />`. Use `<artifact action="clear" />` to hide the currently displayed artifact.

YouTube watch, shortened `youtu.be`, and embed URLs are supported. `autoplay-delay` specifies a fixed delay of 0 to 3600 seconds from displaying the video until its first autoplay attempt and defaults to `0`. The `t` and `start` URL parameters select the initial position within the video; they do not affect the autoplay delay. Browsers may block autoplay with sound, in which case the embedded controls remain available for manual playback.

Web apps use `<artifact type="webapp" src="https://example.com/app" />` and run in a sandboxed iframe. Maps use `<artifact type="map" location="Tokyo Station" />`; directions use `<artifact type="map" origin="Tokyo Station" destination="Tokyo Tower" travel-mode="walking" />`. Maps require a Maps Embed API key in the viewer HTML; restrict it to approved websites and the Maps Embed API. See the [WebSocket example documentation](examples/websocket/README.md#artifacts-in-the-web-viewers) for supported attributes and web-app invocation messages.

The catalog can be changed at runtime:

- `set_artifacts(configs)` replaces the complete catalog.
- `update_artifacts(configs)` adds or replaces multiple entries while retaining other IDs.
- `add_artifact(id, config)` adds or replaces one entry.

The catalog belongs to the Adapter and is shared by all sessions. Use an application-level or session-level store instead when generated artifacts must remain private to one user.

Direct artifact URLs cause the user's browser to request the resolved location. The server application is responsible for allowing only URLs that are safe for its users and network environment. `on_response` runs after artifact ID resolution, so it can validate, rewrite, or remove the resolved `control_tags` before delivery:

```python
from urllib.parse import urlsplit

TRUSTED_ARTIFACT_HOSTS = {"cdn.example.com"}

def is_trusted_artifact_url(source):
    try:
        url = urlsplit(source)
        return (
            url.scheme == "https"
            and url.hostname in TRUSTED_ARTIFACT_HOSTS
            and url.username is None
            and url.password is None
        )
    except ValueError:
        return False

@aiavatar_app.on_response
async def validate_artifact_urls(response, _):
    if not response.control_tags:
        return

    validated = []
    for tag in response.control_tags:
        if tag.name != "artifact":
            validated.append(tag)
            continue

        source = tag.attributes.get("src") or tag.attributes.get("href")
        if source and not is_trusted_artifact_url(source):
            continue
        validated.append(tag)

    response.control_tags = validated
```

A compact system-prompt section is usually sufficient:

```markdown
## Artifacts
To display an image, chart, slide, video, map, or route, insert an `<artifact />` tag immediately before the relevant sentence in the response body. Do not read the tag aloud or explain it.

- Registered artifact: `<artifact id="{ARTIFACT_ID}" />`
- HTTPS URL: `<artifact type="{TYPE}" src="{HTTPS_URL}" />`
  - Use `image` for an image, `chart` for a chart, `presentation` for slides, or `video` for a YouTube video.
  - A Docswell viewing URL (`https://www.docswell.com/s/...`) can be used directly as a presentation `src`.
  - Speaker Deck requires an embed URL (`https://speakerdeck.com/player/...`).
  - YouTube videos accept `autoplay-delay` from `0` to `3600` as the number of seconds before the first autoplay attempt: `<artifact type="video" src="https://www.youtube.com/watch?v=VIDEO_ID" autoplay-delay="3" />`
- Map: `<artifact type="map" location="{PLACE_NAME_OR_ADDRESS}" />`
  - Use this when the user asks to display a map for a place name or address.
  - Optionally set `zoom` to an integer from `0` to `21`.
- Directions: `<artifact type="map" origin="{ORIGIN}" destination="{DESTINATION}" travel-mode="{TRAVEL_MODE}" />`
  - Use `driving` for driving, `walking` for walking, `bicycling` for bicycling, or `transit` for public transit. Omit `travel-mode` when the user does not specify a mode of travel.
- Presentation controls are available only with `type="presentation"`, not with images, charts, or videos.
  - Move the displayed presentation to a numbered page: `<artifact type="presentation" slide="3" />`
  - To set the starting page of a new presentation, specify `id` or `src` together with `slide` in the same tag.
  - Move relative to the current page: `<artifact type="presentation" offset="+1" />`, `offset="-1"`, `offset="+2"`, and so on.
  - For relative navigation, use a signed `offset` instead of `slide` and specify it in a single tag.
  - Never use numbered page navigation when the request is relative to the current page.
- Never invent unknown IDs or URLs.
- Only the most recently specified artifact is displayed. A new artifact replaces the previous one.
- To hide the current artifact, output `<artifact action="clear" />`.

### Available Artifacts
- `about_company`: Company overview presentation
```


## 🥳 Character Management

`CharacterService` provides functionality for managing AI character settings and generating dynamic content such as schedules and diaries based on character personalities.

Schedules and diaries are generated as if by the character's own will. By updating these daily and incorporating them into prompts, you can make the character feel like they are actually living in real-world time.

**Note:** This feature requires PostgreSQL as the database backend.


### Get started

Register a new character using a character setting prompt. At this time, both the weekly schedule and today's schedule are also generated.

```python
from datetime import date
from aiavatar.character import CharacterService

# Initialize service
character_service = CharacterService(
    openai_api_key="YOUR_API_KEY"
)

# Initialize a new character with weekly and daily schedules
character, weekly, daily = await character_service.initialize_character(
    name="Alice",
    character_prompt="You are Alice, a cheerful high school student who loves reading..."
)

print(f"Character ID: {character.id}")
```

To use the registered and generated content as a system prompt, implement `LLMService.get_system_prompt` as follows:

```python
@llm.get_system_prompt
async def get_system_prompt(context_id: str, user_id: str, system_prompt_params: dict):
    return await character_service.get_system_prompt(
        character_id="YOUR_CHARACTER_ID",
        system_prompt_params=system_prompt_params
    )
```

This system prompt includes not only the character settings from `character_prompt`, but also the schedule for the day.


### Updating Diaries

Diaries can be automatically generated using `create_diary_with_generation`. The following information is used:

- Character settings
- Today's schedule
- Today's news (retrieved via web search)
- Previous day's diary

```python
# Generate diary from daily activities
diary = await character_service.create_diary_with_generation(
    character_id=character.id,
    diary_date=date.today()
)
```

The generated diary can be used as context for the LLM using `GetDiaryTool`. By setting `include_schedule=True`, the schedule information for the day is also retrieved (default is `True`).

```python
from aiavatar.character.tools import GetDiaryTool
llm.add_tool(
    GetDiaryTool(
        character_service=character_service,
        character_id=YOUR_CHARACTER_ID,
        include_schedule=True
    )
)
```


### Updating Schedules

Daily schedules can be automatically generated using `create_daily_schedule_with_generation`. The following information is used:

- Character settings
- Weekly schedule
- Previous day's schedule

```python
daily_schedule = await character_service.create_daily_schedule_with_generation(
    character_id=character.id,
    schedule_date=date.today()
)
```

### Automated Daily Updates

For a more realistic character experience, use a scheduler service (such as cron) to automatically update schedules and diaries:

- **Daily schedule**: Generate at the beginning of each day (e.g., 0:00 or 6:00)
- **Diary**: Generate at the end of each day (e.g., 23:00)

Example cron configuration:

```
# Generate daily schedule at 6:00 AM
0 6 * * * /usr/bin/python3 /path/to/generate_schedule.py

# Generate diary at 11:00 PM
0 23 * * * /usr/bin/python3 /path/to/generate_diary.py
```

Example script for `generate_schedule.py`:

```python
import asyncio
from datetime import date
from aiavatar.character import CharacterService

async def main():
    character_service = CharacterService(
        openai_api_key="YOUR_API_KEY"
    )
    await character_service.create_daily_schedule_with_generation(
        character_id="YOUR_CHARACTER_ID",
        schedule_date=date.today()
    )

asyncio.run(main())
```

### Batch Generation

You can batch generate daily schedules and diaries for a date range using `create_activity_range_with_generation`.

```python
await character_service.create_activity_range_with_generation(
    character_id=YOUR_CHARACTER_ID,
    start_date=date(2026, 1, 8),
    end_date=date(2026, 1, 16),  # Defaults to today if omitted
    overwrite=False,
)
```

This is useful for recovering data when automatic updates were stopped, or for building up initial data when creating a new character.

### Long-term Memory

This feature is **optional**. If you want to make diaries searchable as long-term memory, you can integrate with an external memory service by configuring `MemoryClient`:

```python
from aiavatar.character import CharacterService, MemoryClient

memory_client = MemoryClient(base_url="http://memory-service:8000")

character_service = CharacterService(
    openai_api_key="YOUR_API_KEY",
    memory_client=memory_client
)
```

Registered diaries can be included in search results using the `search` method.

```python
# In addition to diaries, conversation history with users and other knowledge are searched comprehensively
result = await character_service.memory.search(
    character_id="YOUR_CHARACTER_ID",
    user_id="YOUR_USER_ID",
    query="travel summer 2026"
)
```

The default `MemoryClient` uses [ChatMemory](https://github.com/uezo/chatmemory) as its backend, but you can also use other long-term memory services by inheriting from `MemoryClientBase`.


### Binding to Adapter

The `bind_character` function provides a convenient way to integrate character management with your AIAvatar application. It automatically configures the system prompt, user management, and character-related tools in a single call.

```python
from aiavatar.character import CharacterService
from aiavatar.character.binding import bind_character

character_service = CharacterService(
    openai_api_key="YOUR_API_KEY"
)

bind_character(
    adapter=aiavatar_app,
    character_service=character_service,
    character_id="YOUR_CHARACTER_ID",
    default_user_name="You"
)
```

This single function call sets up:

- **System prompt**: Automatically retrieves the character's system prompt with user-specific parameters
- **User management**: Creates a new user with `default_user_name` if the user doesn't exist
- **Username sync**: Sends the username and character name to the client on connection, and updates when changed
- **Tools**: Registers the following tools automatically:
  - `UpdateUsernameTool`: Allows the character to update the user's name during conversation
  - `GetDiaryTool`: Retrieves the character's diary and schedule
  - `MemorySearchTool`: Searches long-term memory (only if `memory_client` is configured)


### CharacterLoader (Lightweight Alternative)

`CharacterLoader` is a lightweight alternative to `CharacterService` that loads character settings from local files instead of a database. No database or external API is required — just plain markdown and JSON files.

This is ideal when you want to quickly set up a character without infrastructure, or when you prefer to manage character definitions as files.

#### Single file mode

The simplest usage is to point to a single markdown file containing the system prompt:

```python
from aiavatar.character.loader import CharacterLoader

loader = CharacterLoader("system_prompt.md")

# Bind to LLM service
loader.bind(adapter.sts.llm)
```

#### Directory mode

For richer character definitions, use directory mode with `split_initial_messages=True`. Initial messages are prepended to the conversation history as pseudo user/assistant turns, allowing you to inject character knowledge (episodes, attributes, conversation examples) without overloading the system prompt. Point to a directory containing:

```
my_character/
├── character.md                # Character settings (required with split_initial_messages)
├── response_instructions.md    # Response rules (optional, appended to system prompt)
├── message_templates.json      # Template definitions for initial messages
├── episode.md                  # Character's past experiences (optional)
├── attribute.md                # Likes, dislikes, personality traits (optional)
└── conversation_example.md     # Example dialogues for tone reference (optional)
```

```python
loader = CharacterLoader(
    "my_character",
    split_initial_messages=True,
    lang="ja",
    user_names={"user_001": "Alice"},
    default_user_name="You"
)

loader.bind(adapter.sts.llm)
```

The `message_templates.json` defines how initial messages and self-introduction are structured:

```json
{
    "initial_message_defs": {
        "ja": {
            "self_intro": "わかりました。{username}さんですね。",
            "episode": "わかりました。",
            "attribute": "わかりました。"
        }
    },
    "prefixes": {
        "ja": {
            "episode": "以下はあなたの過去の経験です。\n\n",
            "attribute": "以下はあなたの属性情報です。\n\n"
        }
    },
    "self_intro_template": {
        "ja": "$ユーザーの名前は{username}です。"
    }
}
```

#### Hot reload

All files are cached with mtime-based invalidation. Edit any file while the application is running, and changes will be reflected on the next request — no restart needed.

#### Custom user name resolution

Use the `@loader.get_user_name` decorator to resolve user names dynamically (e.g., from a database or external service):

```python
@loader.get_user_name
def get_user_name(user_id: str):
    return db.get_username(user_id)
```

#### Custom message formatting

Use the `@loader.format_messages` decorator to post-process initial messages before they are sent to the LLM:

```python
@loader.format_messages
def format_messages(messages):
    # Add timestamps, filter messages, etc.
    return messages
```

#### Comparison with CharacterService

| | CharacterLoader | CharacterService |
|---|---|---|
| Data source | Local files (`.md`, `.json`) | Database (SQLite / PostgreSQL) |
| Dependencies | None (standard library only) | `openai`, database libraries |
| Schedule / Diary generation | Not supported | Auto-generated via LLM |
| Long-term memory | Not supported | Supported via MemoryClient |
| Character tools | Not included | username update, diary, memory search |
| Hot reload | Supported (mtime-based) | Not supported |


## 🧩 API

You can host AIAvatarKit on a server to enable multiple clients to have independent context-aware conversations via RESTful API with streaming responses (Server-Sent Events) and WebSocket.

### 💫 RESTful API (SSE)

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


Next is the simplest example of a client program:

```python
import asyncio
from aiavatar.adapter.http.client import AIAvatarHttpClient

aiavatar_app = AIAvatarHttpClient(
    debug=True
)
asyncio.run(aiavatar_app.start_listening(session_id="http_session", user_id="http_user"))
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
    "text": "[face:joy]こんにちは！",   // Response text with info
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

data: {"type": "chunk", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "[face:joy]こんにちは！", "voice_text": "こんにちは！", "avatar_control_request": {"animation_name": null, "animation_duration": null, "face_name": "joy", "face_duration": 4.0}, "audio_data": "XXXX", "metadata": {"is_first_chunk": true}}

data: {"type": "chunk", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "今日はどんなことをお手伝いしましょうか？", "voice_text": "今日はどんなことをお手伝いしましょうか？", "avatar_control_request": {"animation_name": null, "animation_duration": null, "face_name": null, "face_duration": null}, "audio_data": "XXXX", "metadata": {"is_first_chunk": false}}

data: {"type": "final", "session_id": "6d8ba9ac-a515-49be-8bf4-cdef021a169d", "user_id": "user01", "context_id": "c37ac363-5c65-4832-aa25-fd3bbbc1b1e7", "text": "[face:joy]こんにちは！今日はどんなことをお手伝いしましょうか？", "voice_text": "こんにちは！今日はどんなことをお手伝いしましょうか？", "avatar_control_request": null, "audio_data": "XXXX", "metadata": {}}
```

To continue the conversation, include the `context_id` provided in the `start` response in your next request.

**NOTE:** When using the RESTful API, voice activity detection (VAD) must be performed client-side.

**NOTE:** To protect API with API Key, set `api_key=API_KEY_YOU_MAKE` to AIAvatarHttpServer and send `Authorization: Bearer {API_KEY_YOU_MAKE}` as HTTP header from client.


### 🔵 Dify-compatible API

`AIAvatarHttpServer` provides a Dify-compatible `/chat-messages` endpoint (SSE streaming only).
This allows you to connect frontend applications that use Dify as their backend.

For more details, refer to the [Dify API Guide](https://docs.dify.ai/en/guides/application-publishing/developing-with-apis)
or the API documentation of your self-hosted Dify application.


### 🔌 WebSocket

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


#### Connection and Disconnection Handling

You can register callbacks to handle WebSocket connection and disconnection events. This is useful for logging, session management, or custom initialization/cleanup logic.

```python
@aiavatar_app.on_connect
async def on_connect(request, session_data):
    print(f"Client connected: {session_data.id}")
    print(f"User ID: {session_data.user_id}")
    print(f"Session ID: {session_data.session_id}")
    
    # Custom initialization logic
    # e.g., load user preferences, initialize resources, etc.

@aiavatar_app.on_disconnect
async def on_disconnect(session_data):
    print(f"Client disconnected: {session_data.id}")
    
    # Custom cleanup logic
    # e.g., save session data, release resources, etc.
```

The `session_data` object contains information about the WebSocket session:

- `id`: Unique session identifier
- `user_id`: User identifier from the connection request
- `session_id`: Session identifier from the connection request
- Additional metadata passed during connection


### 🟩 LINE Bot

You can build a LINE Bot using the LINE Messaging API.

```python
# NOTE: Register https://{your.domain}/webhook as the "Webhook URL" in LINE Developers Console

# Create LINE Bot adapter
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer
aiavatar_app = AIAvatarLineBotServer(
    system_prompt="You are a cat.",
    openai_api_key=OPENAI_API_KEY,
    channel_access_token=LINEBOT_CHANNEL_ACCESS_TOKEN,
    channel_secret=LINEBOT_CHANNEL_SECRET,
    image_download_url_base="https://{your.domain}",
    debug=True
)

# Create FastAPI app
from fastapi import FastAPI
app = FastAPI()

# Set adapter endpoints
router = aiavatar_app.get_api_router()
app.include_router(router)
```

Note: `image_download_url_base` is optional. If omitted, images from users are embedded as base64 data URLs directly in the LLM context, eliminating the need for the LLM to fetch images from an external URL.

By default, the LINE Messaging API user ID is used as the AIAvatarKit user ID. To map channel user IDs to your own app-level user IDs, use `ChannelContextBridge`. See [Channel Context Bridge](#-channel-context-bridge) for details.

Customization hooks:

```python
@aiavatar_app.preprocess_request
async def preprocess_request(request: STSRequest):
    # Pre-process request before sending to LLM
    # e.g. edit request text
    request.text = "Pre-processed: " + request.text

@aiavatar_app.preprocess_response
async def preprocess_response(response: STSResponse):
    # Pre-process response before sending to LINE API
    # e.g. edit response voice_text (not text)
    response.voice_text = "Pre-processed: " + response.voice_text

@aiavatar_app.process_avatar_control_request
async def process_avatar_control_request(avatar_control_request: AvatarControlRequest, reply_message_request: ReplyMessageRequest):
    # Process facial expression
    # e.g. set `sender` to the message in reply_message_request to change icon
    face = avatar_control_request.face_name
    if face:
        reply_message_request.messages[0].sender = Sender(iconUrl=f"https://your_domain/path/to/icon/{face}.png")

@aiavatar_app.on_send_error_message
async def on_send_error_message(reply_message_request: ReplyMessageRequest, event: Event, ex: Exception):
    # Pre-process error message
    # e.g. edit error response
    text = make_user_friendly_error_message(event, ex)
    reply_message_request.messages[0] = TextMessage(text=text)

@aiavatar_app.event("postback")
async def handle_postback_event(event: Event, user_id: str, context_id: Optional[str]):
    # Process event
    # e.g. Register postback data
    await register_data(user_id, event.postback.data)
```


Context data is stored in `aiavatar.db` via SQLite by default. To use PostgreSQL, create a `PostgreSQLChannelContextBridge` and pass it to `AIAvatarLineBotServer` as `channel_context_bridge`. See [Channel Context Bridge](#-channel-context-bridge) for details.

```python
from aiavatar.adapter.channel_context_bridge.postgres import PostgreSQLChannelContextBridge
bridge = PostgreSQLChannelContextBridge(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

aiavatar_app = AIAvatarLineBotServer(
    system_prompt="You are a cat.",
    openai_api_key=OPENAI_API_KEY,
    channel_access_token=LINEBOT_CHANNEL_ACCESS_TOKEN,
    channel_secret=LINEBOT_CHANNEL_SECRET,
    image_download_url_base="https://{your.domain}",
    channel_context_bridge=bridge,    # <- Set PostgreSQL context bridge
    debug=True
)
```


### STT / TTS Endpoints

AIAvatarHttpServer provides REST API endpoints for Speech-to-Text (STT) and Text-to-Speech (TTS) functionality:

#### STT Endpoint
`POST /stt` - Converts audio to text.

```python
import requests

# Read audio file
with open("audio.wav", "rb") as f:
    audio_data = f.read()

# Send to STT endpoint
response = requests.post(
    "http://localhost:8000/stt",
    data=audio_data,
    headers={"Content-Type": "audio/wav"}
)

print(response.json())  # {"text": "recognized speech"}
```

#### TTS Endpoint
`POST /tts` - Converts text to speech.

```python
import requests

# Send text to TTS endpoint
response = requests.post(
    "http://localhost:8000/tts",
    json={"text": "Hello, this is AI Avatar speaking"}
)

# Save audio response
with open("output.wav", "wb") as f:
    f.write(response.content)
```


## 🛡️ Guardrail

You can apply guardrails to both requests and responses.
Guardrails are custom implementations created by developers, and can block or replace an incoming request, or replace an outgoing response when certain conditions are met.

Below is the implementation method and how to apply guardrails.

```python
from aiavatar.sts.llm import Guardrail, GuardrailRespose

# Define guardrails
class RequestGuardrail(Guardrail):
    async def apply(self, context_id, user_id, text, files = None, system_prompt_params = None):
        if text.lower() == "problematic input":
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=True,
                action="block",
                text="The problematic input has been blocked."  # Immediately returns this message to the user
            )
        elif text.lower() == "hello":
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=True,
                action="replace",
                text="こんにちは"   # Replaces the original request text with this value
            )
        else:
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=False
            )

class ResponseGuardrail(Guardrail):
    async def apply(self, context_id, user_id, text, files = None, system_prompt_params = None):
        if "ramen" in text.lower():
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=True,
                action="replace",
                text="The problematic output has been blocked." # Emits an additional replacement chunk for the response
            )
        else:
            return GuardrailRespose(
                guardrail_name=self.name,
                is_triggered=False
            )

# Apply guardrails
service.guardrails.append(RequestGuardrail(applies_to="request"))
service.guardrails.append(ResponseGuardrail(applies_to="response"))
```

**NOTE:** When multiple guardrails are defined, they run in parallel.
Processing stops when all guardrails have finished evaluating or when the first guardrail returns a response with `is_triggered=True`.

**NOTE:** Response guardrails are evaluated only after the LLM response stream finishes.
This means the problematic output may be briefly visible to the user.
When a response is received with `metadata.is_guardrail_triggered = true`, the client should handle this by replacing or modifying the output accordingly.


## 🌎 Platform Guide

AIAvatarKit is capable of operating on any platform that allows applications to hook into audio input and output. The platforms that have been tested include:

- VRChat
- cluster
- Vket Cloud

In addition to running on PCs to operate AI avatars on these platforms, you can also create a communication robot by connecting speakers, a microphone, and, if possible, a display to a Raspberry Pi.

### 🐈 VRChat

* __2 Virtual audio devices (e.g. VB-CABLE) are required.__
* __Multiple VRChat accounts are required to chat with your AIAvatar.__


#### Get started

First, run the commands below in python interpreter to check the audio devices.

```sh
$ python

>>> from aiavatar.device import AudioDevice
>>> AudioDevice().list_audio_devices()
0: Headset Microphone (Oculus Virt
    :
6: CABLE-B Output (VB-Audio Cable
7: Microsoft サウンド マッパー - Output
8: SONY TV (NVIDIA High Definition
    :
13: CABLE-A Input (VB-Audio Cable A
    :
```

In this example,

- To use `VB-Cable-A` for microphone for VRChat, index for `output_device` is `13` (CABLE-A Input).
- To use `VB-Cable-B` for speaker for VRChat, index for `input_device` is `6` (CABLE-B Output). Don't forget to set `VB-Cable-B Input` as the default output device of Windows OS.

Then edit `run.py` like below.

```python
# Create AIAvatar
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    input_device=6,     # Listen sound from VRChat
    output_device=13,   # Speak to VRChat microphone
)
```

Run it.

```bash
$ run.py
```

Launch VRChat as desktop mode on the machine that runs `run.py` and log in with the account for AIAvatar. Then set `VB-Cable-A` to microphone in VRChat setting window.

That's all! Let's chat with the AIAvatar. Log in to VRChat on another machine (or Quest) and go to the world the AIAvatar is in.


#### Face Expression

AIAvatarKit controls the face expression by [Avatar OSC](https://docs.vrchat.com/docs/osc-avatar-parameters).

LLM(ChatGPT/Claude/Gemini)
↓ *response with face tag* `[face:joy]Hello!` or `<face name="joy" />Hello!`
AIAvatarKit(VRCFaceExpressionController)  
↓ *osc* `FaceOSC=1`  
VRChat(FX AnimatorController)  
↓  
😆

So at first, setup your avatar the following steps:

1. Add avatar parameter `FaceOSC` (type: int, default value: 0, saved: false, synced: true).
1. Add `FaceOSC` parameter to the FX animator controller.
1. Add layer and put states and transitions for face expression to the FX animator controller.
1. (option) If you use the avatar that is already used in VRChat, add input parameter configuration to avatar json.


Next, use `VRChatFaceController`.

```python
from aiavatar.face.vrchat import VRChatFaceController

# Setup VRChatFaceContorller
vrc_face_controller = VRChatFaceController(
    faces={
        "neutral": 0,   # always set `neutral: 0`

        # key = the name that LLM can understand the expression
        # value = FaceOSC value that is set to the transition on the FX animator controller
        "joy": 1,
        "angry": 2,
        "sorrow": 3,
        "fun": 4
    }
)
```

Lastly, add face expression section to the system prompt.

```python
# Make system prompt
system_prompt = """
# Face Expression

* You have following expressions:

- joy
- angry
- sorrow
- fun

* If you want to express a particular emotion, please insert it at the beginning of the sentence like [face:joy].

Example
[face:joy]Hey, you can see the ocean! [face:fun]Let's go swimming.
"""

# Set them to AIAvatar
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    face_controller=vrc_face_controller,
    system_prompt=system_prompt
)
```

You can test it not only through the voice conversation but also via the [REST API](#-restful-apis).


### 🍓 Raspberry Pi

Now writing... ✍️


## ⚙️ Administration

AIAvatarKit provides a built-in admin panel for monitoring, configuring, and evaluating your AI avatar from a web browser.

### Admin Panel

Set up the Admin Panel with a single function call. Once configured, access it at `/admin/` on your server.

```python
import os

from aiavatar.admin import BasicAdminAuthenticator, setup_admin_panel

setup_admin_panel(
    app,
    adapter=aiavatar_app,
    authenticator=BasicAdminAuthenticator(
        os.environ["ADMIN_USERNAME"],
        os.environ["ADMIN_PASSWORD"],
    ),
)
```

The Admin Panel includes:

- **Metrics** — First-response statistics and a detailed latency breakdown measured from the end of the user's speech
- **Logs** — Searchable conversation messages grouped by context, with session filtering, voice playback, and per-turn timing details
- **Config** — Adjust pipeline, VAD, STT, LLM, TTS, and adapter settings at runtime
- **Evaluation** — Run dialog evaluation scenarios when an evaluator is available
- **Light/Dark themes** — Follow the operating system theme or switch it manually

Evaluation is configured automatically when the pipeline uses `ChatGPTService`. For other LLM services, pass a `DialogEvaluator` through the optional `evaluator` argument.

The same authenticator protects the HTML, static assets, and `/admin/api` endpoints. The frontend does not use a separate API key. In addition to `BasicAdminAuthenticator`, `authenticator` accepts any synchronous or asynchronous callable that receives a FastAPI `Request`, allowing integration with an SSO session or an authenticated reverse proxy.

Passing `authenticator=None` disables authentication and should be limited to local development. Use HTTPS when using Basic authentication in production.

Character and Control features are not part of the new Admin Panel. The previous UI and APIs remain available as an independent legacy package when an existing application still needs them:

```python
from aiavatar.admin_legacy import setup_admin_panel
```

See the [Admin Panel documentation](aiavatar/admin/README.md) for authentication examples, screen and API specifications, component responsibilities, time semantics, and frontend development instructions.

### REST API

Admin Panel operations are available under `/admin/api` and use the same authentication as the UI. See the interactive API documentation at `/docs` for request and response schemas, or the [Admin Panel API summary](aiavatar/admin/README.md#api) for an overview.

### 📈 Observability

You can monitor the entire sequence - what requests are sent to the LLM, how they are interpreted, which tools are invoked, and what responses are generated from specific results or data - to support AIAvatar quality improvements and governance.

AIAvatarKit accepts a pre-configured OpenAI-compatible client instance, so tracing
wrappers such as [Langfuse](https://langfuse.com) can be configured before the LLM
service is constructed.

```sh
pip install langfuse
```

```sh
export LANGFUSE_SECRET_KEY=sk-lf-XXXXXXXX
export LANGFUSE_PUBLIC_KEY=pk-lf-XXXXXXXX
export LANGFUSE_BASE_URL=http://localhost:3000
```

```python
from langfuse.openai import AsyncOpenAI

langfuse_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
llm = ChatGPTService(
    openai_client=langfuse_client,
    system_prompt="You are a helpful assistant.",
)
```

The same client-injection pattern works with `OpenAIResponsesService`. The raw
Responses WebSocket transport does not pass through the Langfuse OpenAI client;
manual instrumentation is required if WebSocket tracing is needed.


## 🦜 AI Agent

AIAvatarKit is not just a framework for creating chatty AI characters — it is designed to support agentic characters that can interact with APIs and external data sources (RAG).

### ⚡️ Tool Call

Register tool with spec by `@aiavatar_app.sts.llm.tool`. The spec should be in the format for each LLM.

```python
# Spec (for ChatGPT)
weather_tool_spec = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather and forecast for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"],
        },
    }
}

# Implement tool and register it with spec
@aiavatar_app.sts.llm.tool(weather_tool_spec)
async def get_weather(location: str):
    weather = await weather_api(location=location)  # Call weather API
    return weather  # {"weather": "clear", "temperature": 23.4}
```

Alternatively, register the same tool programmatically:

```python
from aiavatar.sts.llm import Tool

aiavatar_app.sts.llm.add_tool(
    Tool("get_weather", weather_tool_spec, get_weather)
)
```

**Note**: When you register a tool with `add_tool`, the spec is automatically converted to the correct format for GPT, Gemini, or Claude, so you can define it once and use it everywhere.


Before creating your own tools, start with the example tools:

```python
# Google Search
from aiavatar.sts.llm.tools.gemini_websearch import GeminiWebSearchTool
aiavatar_app.sts.llm.add_tool(GeminiWebSearchTool(gemini_api_key=GEMINI_API_KEY))

# Web Scraper
from aiavatar.sts.llm.tools.webscraper import WebScraperTool
aiavatar_app.sts.llm.add_tool(WebScraperTool())
```


### ⌛️ Tool Call with Streaming Progress

Sometimes you may want to provide feedback to the user when a tool takes time to execute. AIAvatarKit supports tools that return stream responses (via `AsyncGenerator`), which allows you to integrate advanced and costly operations — such as interactions with AI Agent frameworks — into real-time voice conversations without compromising the user experience.

Here’s an example implementation. Intermediate progress is yielded with the second return value set to `False`, and the final result is yielded with `True`.

```python
@service.tool(weather_tool_spec)
async def get_weather_stream(location: str):
    # Progress: Geocoding
    yield {"message": "Resolving location"}, False
    geocode = await geocode_api(location=location)

    # Progress: Weather
    yield {"message": "Calling weather api"}, False
    weather = await weather_api(geocode=geocode)  # Call weather API

    # Final result (yield with `True`)
    yield {"weather": "clear", "temperature": 23.4}, True
```

On the user side, the first value in each yield will be streamed as a `progress` response under the `ToolCall` response type.

Additionally, you can yield string values directly to provide immediate voice feedback to the user during processing:

```python
@service.tool(weather_tool_spec)
async def get_weather_stream(location: str):
    # Provide voice feedback during processing
    yield "Converting locaton to geo code. Please wait a moment."
    geocode = await geocode_api(location=location)
    
    yield "Getting weather information."
    weather = await weather_api(geocode=geocode)
    
    # Final result
    yield {"weather": "clear", "temperature": 23.4}, True
```

When you yield a string (str) value, the AI avatar will speak that text while continuing to process the request.


### 🔄 Background Tool Execution

For tools that take a long time to complete (e.g., AI agent calls, complex API orchestrations), AIAvatarKit supports **background execution**. Instead of blocking the conversation, the avatar immediately acknowledges the request and notifies the user when the result is ready via a callback.

To enable background execution, register an `on_completed` callback on the tool. This is the only requirement — the base `Tool` class handles task management, `task_id` generation, and metadata tracking automatically.

```python
from aiavatar.sts.llm import Tool

# Define tool as usual
heavy_task_spec = {
    "type": "function",
    "function": {
        "name": "run_heavy_task",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        },
    }
}

async def run_heavy_task(query: str, metadata: dict = None):
    result = await some_slow_api(query)  # Takes a long time
    return {"answer": result}

tool = Tool("run_heavy_task", heavy_task_spec, run_heavy_task)

# Enable background execution by registering on_completed callback
@tool.on_completed
async def on_completed(result, metadata):
    # result: return value from the tool function (or None on error)
    # metadata: dict containing task_id, user_id, context_id, session_id, channel, submitted_at, arguments, etc.
    answer = result["answer"]
    user_id = metadata["user_id"]
    context_id = metadata["context_id"]
    session_id = metadata["session_id"]

    async for resp in aiavatar_app.sts.invoke(
        STSRequest(
            session_id=session_id,
            user_id=user_id,
            context_id=context_id,
            text=f"Here is the result of the task:\n\n{answer}",
            wait_in_queue=True,
            skip_quick_response=True,
        )
    ):
        await aiavatar_app.handle_response(resp)

llm.add_tool(tool)
```

When background execution is enabled:

1. The tool function is called and runs in the background as an `asyncio.Task`
2. The avatar immediately responds with `immediate_message` (customizable) and a `task_id`
3. When the function completes, `on_completed` is called with the result and metadata

You can customize the immediate message:

```python
tool = Tool(
    "run_heavy_task", heavy_task_spec, run_heavy_task,
    immediate_message="Got it! I'll work on that and let you know when it's done."
)
```

Optionally, register an `on_submitted` callback to be notified when the task is accepted:

```python
@tool.on_submitted
async def on_submitted(task_id, metadata):
    print(f"Task {task_id} submitted")
```

#### Background Timeout (Hybrid Mode)

Sometimes a tool *might* complete quickly but *could* take a long time. With `background_timeout`, AIAvatarKit tries synchronous execution first and falls back to background execution only if the timeout is exceeded.

```python
tool = Tool(
    "run_task", task_spec, run_task,
    background_timeout=3.0  # Try sync for 3 seconds, then go background
)

@tool.on_completed
async def on_completed(result, metadata):
    # Called only when the task didn't complete within the timeout
    print(f"Background result: {result}")
```

- If the tool completes within `background_timeout` seconds → result is returned directly (same as synchronous mode)
- If the tool exceeds the timeout → switches to background mode, returns `immediate_message`, and calls `on_completed` when done

**Note**: `on_completed` (background execution) and `AsyncGenerator` (streaming progress) are mutually exclusive. A tool should use one pattern or the other.


### 📋 Tool Response Formatter (Direct Response)

By default, after a tool executes, the result is passed back to the LLM to generate a human-friendly response (2nd LLM call). However, in some cases you may want to **bypass the LLM and speak the tool result directly**:

- **Accuracy**: Critical information (e.g., order details, reservation IDs) that must not be paraphrased or hallucinated
- **Latency**: Eliminating the 2nd LLM call for faster response times

Use the `@response_formatter` decorator to define a function that converts the tool result into the exact text to speak. When a `response_formatter` is set, the 2nd LLM call is skipped entirely, and the formatted text is spoken directly.

```python
@llm.tool(weather_tool_spec)
async def get_weather(location: str = None):
    weather = await weather_api(location=location)
    return weather  # {"weather": "clear", "temperature": 23.4}

# Register response_formatter to speak the result directly
@llm.tools["get_weather"].response_formatter
def format_weather(result, arguments):
    return f"The weather in {arguments['location']} is {result['weather']}, with a temperature of {result['temperature']} degrees."
```

The formatter receives two arguments:

| Argument | Description |
|----------|-------------|
| `result` | The dict returned by the tool function |
| `arguments` | The dict of arguments passed to the tool by the LLM |

The tool call and its result are still saved to conversation context, so follow-up questions like "What was the temperature again?" work naturally. The formatted text is stored as the assistant's response.

**Note**: Tools without a `response_formatter` continue to work as before (2nd LLM call generates the response). You can mix both patterns: some tools with formatters and others without.

#### Continuing Tool Chains with `continue_chain`

By default, `response_formatter` terminates the tool chain. No further LLM call is made, which maximizes speed. However, if the LLM calls multiple tools in sequence (e.g., check balance first, then fetch campaign info), a direct-response tool would break the chain and prevent subsequent tools from being called.

Use `continue_chain=True` to allow the chain to continue after the direct response:

```python
@llm.tools["get_balance"].response_formatter(continue_chain=True)
def format_balance(result, arguments):
    return f"Your balance is {result['balance']:,} {result['currency']}."
```

| Decorator | Behavior |
|-----------|----------|
| `@tool.response_formatter` | Direct response, **chain stops** (default, fastest) |
| `@tool.response_formatter(continue_chain=True)` | Direct response, **chain continues** (LLM can call more tools) |

When `continue_chain=True`, the formatted text is spoken immediately, and the tool result is also sent back to the LLM so it can decide whether to call additional tools. The LLM's text response for this round is suppressed to avoid duplication, but any subsequent tool calls and their responses proceed normally.


### 📦 Structured Content (Client-side Data)

By default, tool results (`data`) are passed back to the LLM as context. If you also want to send **structured data directly to the client application** (e.g., for rendering UI components, displaying charts, or updating app state), use `structured_content` in `ToolCallResult`.

```python
from aiavatar.sts.llm import ToolCallResult

@llm.tool(weather_tool_spec)
async def get_weather(location: str):
    weather = await weather_api(location)
    return ToolCallResult(
        data={"summary": f"{weather['temperature']}°C, {weather['condition']}"},  # → passed to LLM
        structured_content={"temperature": weather["temperature"], "condition": weather["condition"], "forecast": weather["forecast"]}  # → passed to client
    )
```

`structured_content` propagates through the entire response pipeline (`LLMResponse` → `STSResponse` → `AIAvatarResponse`) and is delivered to the client as a **top-level field** in the JSON response:

```json
{
    "type": "tool_call",
    "structured_content": {"temperature": 23.4, "condition": "sunny", "forecast": [...]},
    "metadata": {"tool_call": {"name": "get_weather", ...}}
}
```

You can also use `structured_content` with async generators for streaming scenarios:

```python
@llm.tool(search_tool_spec)
async def search(query: str):
    yield ToolCallResult(data={"status": "searching"}, is_final=False, structured_content={"loading": True})
    results = await do_search(query)
    yield ToolCallResult(data={"results": results}, is_final=True, structured_content={"loading": False, "items": results})
```

| Field | Destination | Purpose |
|-------|-------------|---------|
| `data` | LLM (as context) | Model uses this to generate a response |
| `structured_content` | Client application | Program handles this for UI/logic |

**Note**: `structured_content` defaults to `None`. Existing tools that return plain `dict` or use shorthand return types are unaffected.


### 🪄 Dynamic Tool Call

AIAvatarKit supports **dynamic Tool Calls**.
When many tools are loaded up-front, it becomes harder to make the model behave as intended and your system instructions explode in size. With AIAvatarKit’s **Dynamic Tool Call** mechanism you load **only the tools that are actually needed at the moment**, eliminating that complexity.

The overall flow is illustrated below.

![Dynamic Tool Call Mechanism](documents/images/dynamic_tool_call.png)

#### 1. Create the tool definitions and implementations  
*(exactly the same as with ordinary tools)*

```python
# Weather
get_weather_spec = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather info at the specified location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        },
    }
}

async def get_weather(location: str):
    resp = await weather_api(location)
    return resp.json() # e.g. {"weather": "clear", "temperature": 23.4}

# Web Search
search_web_spec = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search info from the internet websites",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        },
    }
}
async def search_web(query: str) -> str:
    resp = await web_search_api(query)
    return resp.json() # e.g. {"results": [{...}]}
```

#### 2. Register the tools as dynamic in the AIAvatarKit LLM service

Setting `is_dynamic=True` tells the framework not to expose the tool by default;
AIAvatarKit will inject it only when the Trigger Detection Tool decides the tool is relevant.
You can also supply an `instruction` string that will be spliced into the system prompt on-the-fly.

```python
from aiavatar.sts.llm import Tool

llm = aiavatar_app.sts.llm

# Turn on Dynamic Tool Mode
llm.use_dynamic_tools = True

# Register as Dynamic Tools
llm.tools["get_weather"] = Tool(
    "get_weather",
    get_weather_spec,
    get_weather,
    instruction=(
        "## Use of `get_weather`\n\n"
        "Call this tool to obtain current weather or a forecast. "
        "Argument:\n"
        "- `location`: city name or geo-hash."
    ),
    is_dynamic=True,
)

llm.tools["search_web"] = Tool(
    "search_web",
    search_web_spec,
    search_web,
    instruction=(
        "## Use of `search_web`\n\n"
        "Call this tool to look up information on the public internet. "
        "Argument:\n"
        "- `query`: keywords describing what you want to find."
    ),
    is_dynamic=True,
)
```

Or, register via `add_tool`.

```python
# Difine tool without `is_dynamic` for other use cases
weather_tool = Tool("get_weather", get_weather_spec, get_weather, instruction="...")

# Register tool via `add_tool` with `is_dynamic`
llm.add_tool(weather_tool, is_dynamic=True)
```


#### 3. Tweak the system prompt so the model knows how to use tools

Append a concise “How to use external tools” section (example below).
Replace the example tools with those your application actually relies on for smoother behaviour.


```md
## Use of External Tools

When external tools, knowledge, or data are required to process a user's request, use the appropriate tools.  
The following rules **must be strictly followed** when using tools.

### Arguments

- Use only arguments that are **explicitly specified by the user** or that can be **reliably inferred from the conversation history**.
- **If information is missing**, ask the user for clarification or use other tools to retrieve the necessary data.
- **It is strictly forbidden** to use values as arguments that are not based on the conversation.

### Tool Selection

When a specialized tool is available for a specific purpose, use that tool.  
If you can use only `execute_external_tool`, use it.

Examples where external tools are needed:

- Retrieving weather information  
- Retrieving memory from past conversations  
- Searching for, playing, or otherwise controlling music  
- Performing web searches  
- Accessing real-world systems or data to provide better solutions
```

With these three steps, your AI agent stays lean—loading only what it needs—while still having immediate access to a rich arsenal of capabilities whenever they’re truly required.


#### Custom Tool Repository

By default AIAvatarKit simply hands the **entire list of dynamic tools** to the LLM and lets the model decide which ones match the current context. This approach works for a moderate number of tools, but the size of the prompt places a hard limit on how many candidates you can include.

For larger-scale systems, pair AIAvatarKit with a retrieval layer (e.g., a vector-search index) so that, out of thousands of available tools, only the handful that are truly relevant are executed.

AIAvatarKit supports this pattern through the `get_dynamic_tools` hook.
Register an async function decorated with `@llm.get_dynamic_tools`; it should return a list of **tool specification objects** for the current turn.

```python
@llm.get_dynamic_tools
async def my_get_dynamic_tools(messages: list, metadata: dict) -> list:
    # Retrieve candidate tools from your vector database (or any other store)
    tools = await search_tools_from_vector_db(messages, metadata)
    # Extract and return the spec objects (not the implementations)
    return [t.spec for t in tools]
```

### 🔌 MCP

AIAvatarKit supports tools provided as MCP.

First, install the required `FastMCP` dependency.

```sh
pip install fastmcp
```

The following steps show how to retrieve tools from MCP servers and register them to `LLMService`.

Both Streamable HTTP and standard I/O are supported. The simplest approach is shown in `mcp1` and `mcp3`, but you can also add authentication headers as in `mcp2`, filter tools to only what you need, or customize parts of the schema or execution logic.

```python
from aiavatar.sts.llm.chatgpt import ChatGPTService
llm = ChatGPTService(openai_api_key=OPENAI_API_KEY)

from aiavatar.sts.llm.tools.mcp import StreamableHttpMCP, StdioMCP

# MCP Server
mcp1 = StreamableHttpMCP(url=MCP1_URL)
mcp1.for_each_tool = llm.add_tool

# MCP Server with Auth
mcp2 = StreamableHttpMCP(url=MCP2_URL, headers={"Authorization": f"Bearer {MCP_JWT}"})
@mcp2.for_each_tool
def mcp2_tools(tool: Tool):
    # Do something here (e.g. edit schema or func)
    llm.add_tool(tool)

# MCP Server (Std I/O)
mcp3 = StdioMCP(server_script="weather.py") # supports .py and .js
mcp3.for_each_tool = llm.add_tool
```

### 🛠️ Built-in Tools

You can use the following tools out of the box 📦.

- 🔍 Web Search
    - Gemini Search
    - OpenAI Search
    - Grok Search
- 🌏 Web Scraper
- 🖼️ Image Generation
    - 🍌 Nano Banana
    - 🐓 Selfie

```python
# Web Search
from aiavatar.sts.llm.tools.gemini_websearch import GeminiWebSearchTool
google_search_tool = GeminiWebSearchTool(gemini_api_key=GEMINI_API_KEY)
llm.add_tool(google_search_tool)

from aiavatar.sts.llm.tools.openai_websearch import OpenAIWebSearchTool
web_search_tool = OpenAIWebSearchTool(openai_api_key=OPENAI_API_KEY)
llm.add_tool(web_search_tool)

from aiavatar.sts.llm.tools.grok_search import GrokSearchTool
grok_web_search_tool = GrokSearchTool(xai_api_key=XAI_API_KEY)
llm.add_tool(grok_web_search_tool)

# Web Scraper
from aiavatar.sts.llm.tools.webscraper import WebScraperTool
webscraper_tool = WebScraperTool()
# webscraper_tool = WebScraperTool(openai_api_key=OPENAI_API_KEY, return_summary=True)  # Provides summary instead of full innerText (recommended)
llm.add_tool(webscraper_tool)

# Image Generation
from aiavatar.sts.llm.tools.nanobanana import NanoBananaTool
nanobanana_tool = NanoBananaTool(gemini_api_key=GEMINI_API_KEY)
llm.add_tool(nanobanana_tool)

from aiavatar.sts.llm.tools.nanobanana import NanoBananaSelfieTool
selfie_tool = NanoBananaSelfieTool(gemini_api_key=GEMINI_API_KEY, reference_image=image_bytes_or_image_url_of_file_api)
llm.add_tool(selfie_tool)
```


### 🦞 OpenClaw / Hermes

`OpenClawTool` integrates [OpenClaw](https://openclaw.ai) or [Hermes](https://github.com/nousresearch/hermes-agent), versatile AI agents, as a tool for your avatar. When the LLM determines that the user's request requires autonomous task execution (web search, data analysis, code execution, etc.), it delegates the task to the agent.

```python
from aiavatar.sts.llm.tools.openclaw_tool import OpenClawTool

# OpenClaw (default harness)
openclaw_tool = OpenClawTool(
    openclaw_api_key=OPENCLAW_API_KEY,
    openclaw_base_url=OPENCLAW_BASE_URL,
    stream=True,
    debug=True,
)

# Hermes
openclaw_tool = OpenClawTool(
    openclaw_api_key=HERMES_API_KEY,
    openclaw_base_url=HERMES_BASE_URL,
    harness="hermes",
    stream=True,
    debug=True,
)

llm.add_tool(openclaw_tool)
```

The `harness` parameter selects the built-in request builder and response parser for each backend. Built-in harnesses are `"openclaw"` (default) and `"hermes"`. You can also register custom harnesses — see [Custom harness](#custom-harness) below.

When `on_completed` is registered, OpenClaw runs asynchronously in the background — the avatar immediately acknowledges the request and notifies the user when the result is ready. The approach for delivering the result depends on your adapter.

#### Push-based delivery (WebSocket / Local)

For adapters that support server-initiated messages, use `on_completed` to push the result back through the pipeline:

```python
@openclaw_tool.on_completed
async def on_completed(result, metadata):
    answer = result["answer"]
    user_id = metadata["user_id"]
    context_id = metadata["context_id"]
    session_id = metadata["session_id"]

    async for resp in aiavatar_app.sts.invoke(
        STSRequest(
            session_id=session_id,
            user_id=user_id,
            context_id=context_id,
            text=f"$OpenClaw has returned a response. Please relay the following to the user:\n\n{answer}",
            wait_in_queue=True,
            skip_quick_response=True,
        )
    ):
        await aiavatar_app.handle_response(resp)
```

#### Polling-based delivery (HTTP)

For HTTP adapters where the SSE stream has already closed by the time the background task completes, store results in a buffer and let the client poll for them. The tool returns a `task_id` in its response for this purpose.

Register callbacks to track task lifecycle:

```python
import time as time_module
task_results = {}
TASK_TIMEOUT = 300  # 5 minutes

@openclaw_tool.on_submitted
async def on_submitted(task_id: str, metadata: dict):
    task_results[task_id] = {
        "task_id": task_id,
        "submitted_at": metadata.get("submitted_at", time_module.time()),
        "answer": None,
    }

@openclaw_tool.on_completed
async def on_completed(result, metadata):
    task_id = metadata["task_id"]
    task_results[task_id]["answer"] = result["answer"]
```

Add a polling endpoint for the client to retrieve results:

```python
@app.get("/tasks/{task_id}")
async def get_task_result(task_id: str):
    result = task_results.get(task_id)
    if result is None:
        return Response(status_code=204)
    if result["answer"]:
        task_results.pop(task_id, None)
        return {"task_id": task_id, "answer": result["answer"], "status": "completed"}
    if time_module.time() - result["submitted_at"] > TASK_TIMEOUT:
        task_results.pop(task_id, None)
        return {"task_id": task_id, "answer": None, "status": "timeout"}
    return Response(status_code=204)
```

The client receives the `task_id` from the avatar's immediate response and polls `GET /tasks/{task_id}` until it gets a result (`status: "completed"`) or a timeout (`status: "timeout"`). A `204` response means the task is still in progress.

Once the client retrieves the answer, it can send it back to the avatar as a new request, for example `f"$OpenClaw has returned a response. Please relay the following to the user:\n\n{answer}"`, to have the avatar speak the result aloud.

#### Progress tracking

When OpenClaw runs asynchronously, users may ask "How's it going?" before the task completes. The built-in progress tracking lets the avatar answer with real-time status.

`OpenClawTool` automatically tracks running tasks and, when `stream=True`, updates progress with the agent's intermediate steps (tool calls, labels, etc.) as they stream in.

Register the check tool alongside the main tool:

```python
openclaw_tool = OpenClawTool(
    openclaw_api_key=OPENCLAW_API_KEY,
    openclaw_base_url=OPENCLAW_BASE_URL,
    stream=True,  # Enables detailed progress from streaming chunks
)

llm.add_tool(openclaw_tool)
llm.add_tool(openclaw_tool.create_check_tool())
```

That's it. When the user asks about progress, the LLM calls `check_running_openclaw_tasks` and gets the current status:

```json
{
  "running_tasks": [
    {
      "request": "Search for the latest news about AI",
      "progress": "Start processing...\n- 🔍 web_search: searching for AI news\n- 📄 read_page: reading article\n"
    }
  ]
}
```

You can customize the tool name and description:

```python
openclaw_tool.create_check_tool(
    name="check_agent_status",
    description="Check what the AI agent is currently working on."
)
```

#### Report channel routing

By default, task results are delivered back to the same channel (WebSocket, phone, LINE, etc.) that initiated the request. You can override this by specifying a `report_channel` — either at invocation time via the tool parameter, or dynamically while the task is running.

The LLM can set the channel at invocation:

```python
# LLM calls: send_query_to_openclaw(query="...", report_channel="linebot")
```

Or change it mid-flight using the set report channel tool:

```python
llm.add_tool(openclaw_tool.create_set_report_channel_tool())
```

This allows the LLM to call `set_openclaw_report_channel(task_id="...", report_channel="sms")` while the task is running, redirecting where the result will be reported.

#### Per-user configuration

In multi-user environments, each user can connect to their own OpenClaw or Hermes instance with independent credentials. Users without a configuration will receive an error message instead of calling the API.

```python
from aiavatar.sts.llm.tools.openclaw_tool import OpenClawTool, OpenClawConfig

openclaw_tool = OpenClawTool(
    openclaw_configs={
        "user_id_1": OpenClawConfig(
            openclaw_api_key=USER1_API_KEY,
            openclaw_base_url=USER1_BASE_URL,
        ),
        "user_id_2": OpenClawConfig(
            openclaw_api_key=USER2_API_KEY,
            openclaw_base_url=USER2_HERMES_URL,
            harness="hermes",
        ),
    },
    stream=True,
)
```

Per-user configs are merged with the tool-level defaults. Only the fields you specify are overridden — `harness` falls back to the tool-level default (`"openclaw"`). You can also manage configs at runtime:

```python
# Add or update
openclaw_tool.update_openclaw_config("user_id_3", OpenClawConfig(
    openclaw_api_key="new-key",
    openclaw_base_url="https://my-hermes.example.com",
    harness="hermes",
))

# Remove (reverts to tool defaults)
openclaw_tool.delete_openclaw_config("user_id_3")
```

#### Custom harness

You can register custom harnesses to support backends beyond OpenClaw and Hermes. A harness consists of a **request builder** and a **response parser**.

The **request builder** constructs the extra kwargs passed to the API call. It returns a dict that may include `model`, `extra_headers`, `extra_body`, etc.

```python
@openclaw_tool.request_builder("my_harness")
def my_request_builder(task_id, context_id):
    # Use a previously stored session key if available, otherwise use context_id
    session_key = openclaw_tool.get_session_key("my_harness", context_id) or context_id
    result = {"model": "my-model"}
    if session_key:
        result["extra_body"] = {"session_id": session_key}
    return result
```

The **response parser** processes each streaming chunk. It handles progress tracking, session key storage, and returns the content text (or `None`).

```python
@openclaw_tool.response_parser("my_harness")
def my_response_parser(task_id, context_id, chunk):
    # Store session key returned by the harness
    if hasattr(chunk, "session_id") and chunk.session_id:
        openclaw_tool.set_session_key("my_harness", context_id, chunk.session_id)

    # Track progress
    if hasattr(chunk, "tool") and chunk.tool:
        openclaw_tool.add_progress(task_id, f"- {chunk.tool}\n")

    # Return content
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        return delta.content
    return None
```

Assign the custom harness to users via `OpenClawConfig`:

```python
openclaw_tool.update_openclaw_config("user_id", OpenClawConfig(
    openclaw_api_key="key",
    openclaw_base_url="https://my-backend.example.com",
    harness="my_harness",
))
```


## 📡 Channel Adapter

A channel adapter connects a client or an external messaging service to an `STSPipeline`. An adapter can create its own pipeline from its convenience parameters, or attach to an existing pipeline through the `sts` parameter. Attaching multiple adapters to one pipeline lets all channels share the same VAD, STT, LLM, TTS, conversation store, and pipeline hooks.

Each adapter registers itself as a response handler when it is created. No additional registration is required. A `session_id` must identify only one active adapter session within a shared pipeline so that responses can be routed to the correct channel.

### Adapters

The following examples assume that `app` is a FastAPI application and `sts` is an existing `STSPipeline`:

```python
import os
from fastapi import FastAPI
from aiavatar.sts import STSPipeline
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

app = FastAPI()
sts = STSPipeline(
    stt=OpenAISpeechRecognizer(
        openai_api_key=os.environ["OPENAI_API_KEY"],
    ),
    llm_openai_api_key=os.environ["OPENAI_API_KEY"],
    llm_system_prompt="You are a helpful assistant.",
)
```

When `sts` is supplied to an adapter, configure pipeline-level behavior on that shared `STSPipeline`. The adapter's other arguments configure only its transport and channel-specific behavior.

#### WebSocket Adapter

`AIAvatarWebSocketServer` accepts streaming microphone audio and performs VAD on the server. Its default channel name is `"websocket"`.

```python
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer

websocket_adapter = AIAvatarWebSocketServer(
    sts=sts,
    channel="websocket",
    api_key="YOUR_WEBSOCKET_API_KEY",  # Optional
)
app.include_router(websocket_adapter.get_websocket_router(path="/ws"))
```

See [WebSocket](#-websocket) for the wire protocol and client example.

#### REST API Adapter

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

See [RESTful API (SSE)](#-restful-api-sse) for request and response formats.

#### Twilio Voice Adapter

`AIAvatarTwilioServer` connects Twilio Media Streams to the pipeline. Its default channel name is `"phone"`. If the router is mounted at `/twilio`, include that prefix in `webhook_base_url` so the generated WebSocket URL points to `/twilio/ws`.

```python
import os
from aiavatar.adapter.twilio.server import AIAvatarTwilioServer

twilio_voice_adapter = AIAvatarTwilioServer(
    sts=sts,
    account_sid=os.environ["TWILIO_ACCOUNT_SID"],
    auth_token=os.environ["TWILIO_AUTH_TOKEN"],
    phone_number=os.environ["TWILIO_PHONE_NUMBER"],
    webhook_base_url="https://your-domain.example/twilio",
    channel="phone",
)
app.include_router(twilio_voice_adapter.get_router(), prefix="/twilio")
```

Configure `https://your-domain.example/twilio/voice` as the Twilio voice webhook.

#### Asterisk Adapter

The Asterisk adapter connects ARI call control and a bidirectional Media
WebSocket to an existing `STSPipeline`. Setup, transfer strategies, lifecycle
behavior, Asterisk configuration examples, and operational constraints are
documented in the [Asterisk adapter guide](https://github.com/uezo/aiavatarkit/blob/main/aiavatar/adapter/asterisk/README.md).

#### Twilio SMS Adapter

Voice and SMS use separate adapters because they have different session and response-delivery mechanisms. `AIAvatarTwilioSMSServer` requires an existing pipeline and defaults to the `"sms"` channel. It can reuse the Twilio client created by the voice adapter.

```python
from aiavatar.adapter.twilio.server import AIAvatarTwilioSMSServer

twilio_sms_adapter = AIAvatarTwilioSMSServer(
    sts=sts,
    twilio_client=twilio_voice_adapter.twilio_client,
    phone_number=os.environ["TWILIO_PHONE_NUMBER"],
    channel="sms",
)
app.include_router(twilio_sms_adapter.get_router(path="/sms"), prefix="/twilio")
```

Configure `https://your-domain.example/twilio/sms` as the Twilio messaging webhook. If voice is not enabled, pass `account_sid` and `auth_token` directly instead of `twilio_client`.

#### LINE Bot Adapter

`AIAvatarLineBotServer` receives LINE Messaging API webhooks. Its default channel name is `"linebot"`.

```python
import os
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer

line_adapter = AIAvatarLineBotServer(
    sts=sts,
    channel_access_token=os.environ["LINEBOT_CHANNEL_ACCESS_TOKEN"],
    channel_secret=os.environ["LINEBOT_CHANNEL_SECRET"],
    channel="linebot",
)
app.include_router(line_adapter.get_api_router(), prefix="/line")
```

Configure `https://your-domain.example/line/webhook` as the webhook URL in the LINE Developers Console. See [LINE Bot](#-line-bot) for supported messages and customization hooks.

#### Chat Completions Adapter

`AIAvatarChatCompletionsServer` exposes an experimental OpenAI-compatible Chat Completions endpoint. Its default channel ID is `"chatcompletions"`.

```python
from aiavatar.adapter.chatcompletions.server import AIAvatarChatCompletionsServer

chat_completions_adapter = AIAvatarChatCompletionsServer(
    sts=sts,
    channel_id="chatcompletions",
)
app.include_router(chat_completions_adapter.get_api_router())
```

Every request must include a bearer token. The adapter uses that token as the channel-specific user key for context mapping, so callers should use a stable token for the same user and must not share a token between users.

### Connecting Multiple Channels

Create the pipeline once, then pass it to every additional adapter. For example, the following application creates the pipeline through the WebSocket adapter and then attaches a LINE Bot adapter to it:

```python
import os
from fastapi import FastAPI
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer

app = FastAPI()

# The first adapter creates and owns the shared pipeline configuration.
websocket_adapter = AIAvatarWebSocketServer(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    system_prompt="You are a helpful assistant.",
    channel="websocket",
)

# The second adapter attaches to the exact same pipeline instance.
line_adapter = AIAvatarLineBotServer(
    sts=websocket_adapter.sts,
    channel_access_token=os.environ["LINEBOT_CHANNEL_ACCESS_TOKEN"],
    channel_secret=os.environ["LINEBOT_CHANNEL_SECRET"],
    channel="linebot",
)

app.include_router(websocket_adapter.get_websocket_router(path="/ws"))
app.include_router(line_adapter.get_api_router(), prefix="/line")
```

This shares pipeline components and conversation storage, but it does not by itself establish that a WebSocket user and a LINE user are the same person. Use a channel context bridge when conversation continuity must follow a user across channels.

### Sharing Context Across Channels

`ChannelContextBridge` maps each `(channel_id, channel_user_id)` pair to an application-level `user_id`, then stores the latest `context_id` for that application user. Adapters mapped to the same application user therefore resume the same conversation until the bridge timeout expires.

Use one bridge instance for all adapters. LINE Bot and Chat Completions use their supplied bridge internally. Bind the bridge to the WebSocket adapter:

```python
import os
from aiavatar.adapter.channel_context_bridge import SQLiteChannelContextBridge
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer

bridge = SQLiteChannelContextBridge(
    db_path="channel_context_bridge.db",
    timeout=3600,
)

websocket_adapter = AIAvatarWebSocketServer(
    sts=sts,
    channel="websocket",
)
bridge.bind(websocket_adapter, channel_id="websocket")

line_adapter = AIAvatarLineBotServer(
    sts=sts,
    channel_access_token=os.environ["LINEBOT_CHANNEL_ACCESS_TOKEN"],
    channel_secret=os.environ["LINEBOT_CHANNEL_SECRET"],
    channel="linebot",
    channel_context_bridge=bridge,
)
```

The bridge's `channel_id` is the identity namespace used to look up a channel user. It should normally match the adapter's actual channel name—for example, bind an adapter created with `channel="websocket_m5"` using `channel_id="websocket_m5"`. A renamed tag such as `"desktop_robot"` in `insert_channel_tag` is only an LLM-facing label and is not used by the bridge.

By default, an automatically created mapping uses the channel user ID as the application user ID. If both channels provide the same stable user ID, they therefore share context without an explicit link. When the channel user IDs differ, link both identities to one application user from a trusted account-linking or startup flow:

```python
await bridge.link_channel_user(
    channel_id="websocket",
    channel_user_id="web-user-123",
    user_id="user-123",
)
await bridge.link_channel_user(
    channel_id="linebot",
    channel_user_id="U0123456789abcdef",
    user_id="user-123",
)
```

The WebSocket client should continue to send its channel-specific ID (`"web-user-123"`) as `user_id`; the bridge resolves it to `"user-123"` before the request reaches the pipeline. After `timeout` seconds without a context update, a request that does not supply its own `context_id` starts a new context.

For PostgreSQL storage and custom user ID generation, see [Channel Context Bridge](#-channel-context-bridge) in Deep Dive.

### Channel-aware Processing

Every adapter assigns a channel to its requests. The adapters described above default to `"websocket"`, `"http"`, `"phone"`, `"sms"`, `"linebot"`, and `"chatcompletions"`. Override the adapter's `channel` (or `channel_id` for Chat Completions) when the application needs a more specific name, such as `"websocket_m5"`.

Set `insert_channel_tag` on the shared pipeline to expose that channel to the LLM:

```python
sts.insert_channel_tag = [
    "phone",                                  # Keep the channel name.
    "sms",
    ("websocket_m5", "desktop_robot"),       # Rename it for the LLM.
]
```

The pipeline then transforms requests as follows before invoking the LLM:

```text
# phone
<channel name='phone' />Hello

# websocket_m5
<channel name='desktop_robot' />Hello
```

Channels not included in the list receive no tag. Set `insert_channel_tag=True` to insert every request's channel without filtering, or `False` to disable insertion.

Describe the desired behavior in the system prompt, for example:

```text
When <channel name='sms' /> is present, respond briefly without speech-oriented phrasing.
When <channel name='phone' /> is present, use natural spoken language.
When <channel name='desktop_robot' /> is present, you may refer to the robot's body and surroundings.
```

The LINE Bot, Twilio SMS, and Chat Completions adapters automatically add their channel names to the shared pipeline's `skip_tts_channels`. For another text-only adapter, add its channel explicitly:

```python
sts.skip_tts_channels.append("your_text_channel")
```

See [Channel-aware Processing](#-channel-aware-processing) in Deep Dive for additional details.


## 💻 Command Line Interface

The `aiavatar` command starts a ready-to-use WebSocket application when no script is supplied, or runs a custom Python ASGI application in script mode. Use `aiavatar --help` to list command-line options such as `--host` and `--port`.

### Built-in Application

The built-in application uses `SileroStreamSpeechDetector` with filler and Namo Turn gates, `OpenAISpeechRecognizer`, `OpenAIResponsesWebSocketService`, and the WebSocket Adapter. Japanese speech routes to `VoicevoxSpeechSynthesizer` with `AlphabetToKanaPreprocessor`; other languages route to `OpenAISpeechSynthesizer`.

If the Namo Turn optional dependencies are unavailable in an interactive
terminal, the command offers to install `aiavatar[namo-turn]` and continues the
same launch after installation. Declining starts the application without Namo
Turn. In a non-interactive environment, the command logs a warning and starts
without Namo Turn rather than attempting an installation. Script mode does not
inspect or install Namo Turn dependencies because the script owns its component
graph.

The command downloads the WebSocket example UI into `html/` only when that directory does not already exist. The application is then available at http://127.0.0.1:8000/, with the Admin Panel at http://127.0.0.1:8000/admin/.

See [`.env.example`](.env.example) for all built-in application settings. The command automatically loads `.env` from the current working directory without overriding variables already present in the process environment:

```sh
cp .env.example .env
# Edit OPENAI_API_KEY in .env
aiavatar
```

The Admin Config view can update safe members of the running Pipeline, components, and Adapter. These changes are intentionally volatile and are discarded when the process exits. Component composition remains owned by Python application code.

### OpenAI and LLM Configuration

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

The default LLM API is the OpenAI Responses WebSocket API. Set `AIAVATAR_LLM_API=chat-completions` or pass `--llm-api chat-completions` for an OpenAI-compatible service that only implements Chat Completions.

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

### Built-in TTS Routing

Japanese and non-Japanese TTS are independent routes. `AIAVATAR_JA_TTS` defaults to `voicevox`, while `AIAVATAR_MULTI_TTS` defaults to `openai`; either route can select `voicevox`, `openai`, or `instant`. The corresponding `AIAVATAR_JA_TTS_CONFIG` and `AIAVATAR_MULTI_TTS_CONFIG` JSON objects override that route's provider options. `--ja-tts` and `--multi-tts` override only the provider selection.

Shared VOICEVOX and OpenAI defaults remain available through `AIAVATAR_VOICEVOX_*` and `AIAVATAR_OPENAI_TTS_*`. Route config values take precedence. Japanese TTS enables `AlphabetToKanaPreprocessor` by default and the multi route disables it; set `"alphabet_to_kana": false` or `true` in the applicable route config to override that behavior.

`instant` maps the route config to `create_instant_synthesizer()`. It is intentionally limited to a single HTTP request whose raw response body is uncompressed PCM WAV audio. Authentication headers, request parameters, and JSON bodies are supplied directly in the config. More complex response parsing, encoded audio extraction, conversion, or authentication logic belongs in a Python application script.

For example, Aivis Cloud can be configured as an instant Japanese TTS. The API key is part of the private process environment and must not be committed:

```sh
AIAVATAR_JA_TTS=instant \
AIAVATAR_JA_TTS_CONFIG='{"method":"POST","url":"https://api.aivis-project.com/v1/tts/synthesize","headers":{"Authorization":"Bearer YOUR_AIVIS_API_KEY","Content-Type":"application/json"},"json":{"model_uuid":"YOUR_MODEL_UUID","text":"{text}","output_format":"wav","output_sampling_rate":16000,"output_audio_channels":"mono","use_ssml":false},"cache_dir":"ttscache/aivis"}' \
aiavatar
```

### Script Mode

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


## 🧪 Evaluation

AIAvatarKit includes a comprehensive evaluation framework for testing and assessing AI avatar conversations. The `DialogEvaluator` enables scenario-based conversation execution with automatic evaluation capabilities.

### Features

- **Scenario Execution**: Run predefined dialog scenarios against your AI system
- **Turn-by-Turn Evaluation**: Evaluate each conversation turn against specific criteria
- **Goal Assessment**: Evaluate overall scenario objective achievement
- **Result Management**: Save, load, and display evaluation results

### Basic Usage

```python
import asyncio
from aiavatar.eval.dialog import DialogEvaluator, Scenario, Turn
from aiavatar.sts.llm.chatgpt import ChatGPTService

async def main():
    # Initialize LLM services
    llm = ChatGPTService(api_key="your_api_key")
    evaluation_llm = ChatGPTService(api_key="your_api_key")
    
    # Create evaluator
    evaluator = DialogEvaluator(
        llm=llm,                    # LLM for conversation
        evaluation_llm=evaluation_llm  # LLM for evaluation
    )
    
    # Define scenario
    scenario = Scenario(
        name="Order tracking support",
        goal="Provide efficient and helpful customer service for order tracking inquiries",
        turns=[
            Turn(
                input_text="Hello, I need help with my order",
                evaluation_criteria="Responds politely and shows willingness to help"
            ),
            Turn(
                input_text="My order number is 12345",
                evaluation_criteria="Acknowledges the order number and proceeds appropriately"
            )
        ]
    )
    
    # Run evaluation
    results = await evaluator.run(
        dataset=[scenario],
        detailed=True,                # Enable turn-by-turn evaluation
        overwrite_execution=False,    # Skip if already executed
        overwrite_evaluation=False    # Skip if already evaluated
    )
    
    # Display results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")

if __name__ == "__main__":
    asyncio.run(main())
```

Example Output:

```
=== Scenario 1 ===
Goal: Provide helpful customer support

Turn 1:
  Input: Hello, I need help with my order
  Actual Output: Hello! I'd be happy to help you with your order. Could you please provide your order number?
  Result: ✓ PASS
  Reason: The response is polite, helpful, and appropriately asks for the order number.

Turn 2:
  Input: My order number is 12345
  Actual Output: Thank you for providing order number 12345. Let me look that up for you.
  Result: ✓ PASS
  Reason: Acknowledges the order number and shows willingness to help.

Summary: 2/2 turns passed (100.0%)

=== Overall Scenario Evaluation ===
Goal Achievement: ✓ SUCCESS
Reason: The AI successfully provided helpful customer support by responding politely and efficiently handling the order inquiry.
```

### File-Based Evaluation

Load scenarios from JSON files:

```json
{
  "scenarios": [
    {
      "goal": "Basic greeting and assistance",
      "turns": [
        {
          "input_text": "Hello",
          "expected_output": "Friendly greeting",
          "evaluation_criteria": "Responds warmly and appropriately"
        }
      ]
    }
  ]
}
```

```python
# Load and evaluate from file
results = await evaluator.run(dataset="test_scenarios.json")

# Save results back to file
evaluator.save_results(results, "results.json")
```

### Configuration Options

```python
# Execution modes
results = await evaluator.run(
    dataset=scenarios,
    detailed=True,                # Turn-by-turn evaluation
    overwrite_execution=True,     # Re-run conversations
    overwrite_evaluation=True     # Re-evaluate results
)

# Simple mode (scenario-level evaluation only)
results = await evaluator.run(
    dataset=scenarios,
    detailed=False
)
```

### Use via Config API

You can evaluate scenario on the fly via Config API:

```python
# Make evaluator
from aiavatar.eval.dialog import DialogEvaluator
eval_llm = ChatGPTService(openai_api_key=OPENAI_API_KEY)
evaluator = DialogEvaluator(llm=aiavatar_app.sts.llm, evaluation_llm=eval_llm)

# Activate Config API
from aiavatar.admin.config import ConfigAPI
config_router = ConfigAPI(aiavatar_app.sts, evaluator=evaluator).get_router()   # Set evaluator here
app.include_router(config_router)
```

### Logic-based evaluation

In addition to LLM-based evaluation using `evaluation_criteria`, you can evaluate more explicitly using custom logic functions.

```python
# Make evaluation function(s)
def evaluate_weather_tool_call(output_text, tool_call, evaluation_criteria, result, eval_result_text):
    if tool_call is not None and tool_call.name != "get_weather":
        # Overwrite result and reason
        return False, f"Incorrect tool call: {tool_call.name}"
    else:
        # Pass through
        return result, eval_result_text

# Register evaluation function(s)
evaluator = DialogEvaluator(
    llm=aiavatar_app.sts.llm,
    evaluation_llm=eval_llm,
    evaluation_functions={"evaluate_weather_tool_call_func": evaluate_weather_tool_call}
)

# Use evaluation function in scenario
scenario = Scenario(
    turns=[
        Turn(input_text="Hello", expected_output_text="Hi", evaluation_criteria="Greeting"),
        Turn(input_text="What is the weather in Tokyo?", expected_output_text="It's sunny.", evaluation_criteria="Answer the weather based on the result of calling get_weather tool.", evaluation_function_name="evaluate_weather_tool_call_func"),
    ],
    goal="Answer the weather in Tokyo based on the result of get_weather."
)
```


## 🤿 Deep dive

Advanced usases.


### 🐘 PostgreSQL

You can use PostgreSQL instead of the default SQLite. We strongly recommend using PostgreSQL in production environments for its scalability and performance benefits from asynchronous processing.

To use PostgreSQL, install asyncpg and create a `PostgreSQLPoolProvider` to manage the shared connection pool. Then pass it to the constructors of the components that need database access.


```sh
pip install asyncpg
```

```python
# DB_CONNECTION_STR = "postgresql://{user}:{password}@{host}:{port}/{databasename}"
DB_CONNECTION_STR = "postgresql://postgres:postgres@127.0.0.1:5432/aiavatar"

# PoolProvider
from aiavatar.database.postgres import PostgreSQLPoolProvider
pool_provider = PostgreSQLPoolProvider(
    connection_str=DB_CONNECTION_STR,
    # max_size=20,  # Max connection count (default: 20)
    # min_size=5    # Min connection count (default: 5)
)

# Character
from aiavatar.character import CharacterService
character_service = CharacterService(
    openai_api_key=OPENAI_API_KEY,
    db_pool_provider=pool_provider,     # Creates PostgreSQLCharacterRepository and PostgreSQLActivityRepository internally
)

# LLM
from aiavatar.sts.llm.context_manager.postgres import PostgreSQLContextManager
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt=SYSTEM_PROMPT,
    context_manager=PostgreSQLContextManager(
        get_pool=pool_provider.get_pool # Set `get_pool` to PostgreSQLContextManager
    )
)

# Adapter (Create pipeline internally)
ws_app = AIAvatarWebSocketServer(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    db_pool_provider=pool_provider,     # Creates PostgreSQLSessionStateManager and PostgreSQLPerformanceRecorder internally
)
```

**NOTE**: You can also pass PostgreSQL connection settings directly to each component's constructor to manage and use individual connections separately from the shared connection pool. However, this makes it difficult to manage the total number of connections, especially when using multiple workers. We recommend using the shared pool unless you have a specific reason not to.

**NOTE**: `PerformanceRecorder` runs in a separate thread from the main thread, so it does not use the shared connection pool. Instead, it retrieves only the connection information from the PoolProvider and creates its own dedicated connection pool. It writes performance information serially as it receives it through a queue, so it basically uses only a single connection. We recommend not changing this unless you have a specific reason.


### ⚠️ LLM Error Handling

You can handle errors that occur during LLM API calls by using the `on_error` decorator. This is useful for customizing avatar responses when content filters are triggered or when API errors occur.

```python
from aiavatar.sts.llm import LLMResponse

@llm.on_error
async def on_error(llm_response: LLMResponse):
    ex = llm_response.error_info.get("exception")   # Get exception
    error_json = llm_response.error_info.get("response_json", {})   # Get response JSON from OpenAI

    # Make response
    if error_json.get("error", {}).get("code") == "content_filter":
        llm_response.text = "[face:angry]You shouldn't say that!"
        llm_response.voice_text = "You shouldn't say that!"
    else:
        llm_response.text = "[face:sorrow]An error occurred"
        llm_response.voice_text = "An error occurred"
```

**NOTE**: When an error occurs, the conversation context is not updated. This is intentional because including the programmatically overwritten response in the context may cause unexpected LLM behavior in subsequent conversations.


### 🖍️ Custom Chat Logging

Use the `print_chat` decorator to customize how user/AI conversation turns are logged.

```python
@llm.print_chat
def print_chat(role, context_id, user_id, text, files):
    if role == "user":
        logger.info(f"\033[1;32mUser:\033[0m {text}")
    else:
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if think_match or answer_match:
            if think_match:
                logger.info(f"\033[3;38;5;246mThinking: {think_match.group(1).strip()}\033[0m")
            logger.info(f"\033[1;35mAI:\033[0m {answer_match.group(1).strip() if answer_match else text}")
        else:
            logger.info(f"\033[1;35mAI:\033[0m {text}")
```

**NOTE**: This example uses ANSI escape sequences optimized for console output. These escape codes will appear as noise in log files.


### 👀 Vision

AIAvatarKit captures and sends image to AI dynamically when the AI determine that vision is required to process the request. This gives "eyes" to your AIAvatar in metaverse platforms like VRChat.

```python
# Instruct vision tag in the system message
SYSTEM_PROMPT = """
## Using Vision

If you need an image to process a user's request, you can obtain it using the following methods:

- screenshot
- camera

If an image is needed to process the request, add an instruction like [vision:screenshot] to your response to request an image from the user.

By adding this instruction, the user will provide an image in their next utterance. No comments about the image itself are necessary.

Example:

user: Look! This is the sushi I had today.
assistant: [vision:screenshot] Let me take a look.
"""

# Create AIAvatar with the system prompt
aiavatar_app = AIAvatar(
    system_prompt=SYSTEM_PROMPT,
    openai_api_key=OPENAI_API_KEY
)

# Implement get_image_url
import base64
import io
import pyautogui    # pip install pyautogui
from aiavatar.device.video import VideoDevice   # pip install opencv-python
default_camera = VideoDevice(device_index=0, width=960, height=540)

@aiavatar_app.get_image_url
async def get_image_url(source: str) -> str:
    image_bytes = None

    if source == "camera":
        # Capture photo by camera
        image_bytes = await default_camera.capture_image("camera.jpg")
    elif source == "screenshot":
        # Capture screenshot
        buffered = io.BytesIO()
        image = pyautogui.screenshot(region=(0, 0, 1280, 720))
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

    if image_bytes:
        # Upload and get url, or, make base64 encoded url
        b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        b64_url = f"data:image/jpeg;base64,{b64_encoded}"
        return b64_url
```

> **Note:** XML-style tag is also supported: `<vision source="screenshot" />`


### 💾 Long-term Memory

To recall information from past conversations across different contexts, a long-term memory service is used.

To store conversation history, define a function decorated with `@aiavatar_app.sts.on_finish`. To retrieve memories from the conversation history, call the search function of the long-term memory service as a tool.

Below is an example using [ChatMemory](https://github.com/uezo/chatmemory).

```python
# Create client for ChatMemory
from aiavatar.character.memory import MemoryClient
memory_client = MemoryClient(
    base_url="http://localhost:8000"
)

# Add messages to ChatMemory service
@aiavatar_app.sts.on_finish
async def on_finish(request, response):
    await memory_client.add_messages(
        character_id=YOUR_CHARACTER_ID,  # Character ID registered via CharacterService, or any value to separate memory spaces
        request=request,
        response=response
    )

# Add MemorySearchTool to recall past events, conversations, or information about the user.
from aiavatar.character.tools import MemorySearchTool
llm.add_tool(
    MemorySearchTool(
        memory_client=memory_client,
        character_id=YOUR_CHARACTER_ID,
        debug=True
    )
)
```


### 🐓 Wakeword

Set `wakewords` when instantiating `AIAvatar`. Conversation will start when the AIAvatar recognizes one of the words in this list. You can also set `wakeword_timeout`, after which the AIAvatar will return to listening for the wakeword again.

```python
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    wakewords=["Hello", "こんにちは"],
    wakeword_timeout=60,
)
```


### 📋 System Prompt Parameters

You can embed parameters into your system prompt dynamically.

First, define your `AIAvatar` instance with a system prompt containing placeholders:

```python
aiavatar_app = AIAvatar(
    openai_api_key="YOUR_OPENAI_API_KEY",
    system_prompt="User's name is {name}."
)
```

When invoking, pass the parameters as a dictionary using `system_prompt_params`:

```python
aiavatar_app.sts.invoke(STSRequest(
    # (other fields omitted)
    system_prompt_params={"name": "Nekochan"}
))
```

Placeholders in the system prompt, such as `{name}`, will be replaced with the corresponding values at runtime.


### 🎛️ Inline LLM Parameters

When calling `LLMService.chat_stream` directly (outside the Speech-to-Speech pipeline), you can override model-specific parameters on a per-request basis using `inline_llm_params`.

```python
# Override provider-supported generation parameters for a single call
async for chunk in llm.chat_stream(
    context_id="ctx_001",
    user_id="user_001",
    text="Hello!",
    inline_llm_params={"reasoning_effort": "none", "temperature": 0.0}
):
    print(chunk.text, end="", flush=True)
```

The key-value pairs in `inline_llm_params` are merged into the underlying API call parameters, so any parameter accepted by the provider's API can be specified. AIAvatarKit does not validate combinations such as `temperature` plus `reasoning_effort`; the selected endpoint and model must support them. The exact keys depend on the LLM service:

| Service | Example keys |
|---|---|
| ChatGPTService | `model`, `temperature`, `reasoning_effort`, ... |
| ClaudeService | `model`, `temperature`, `max_tokens`, ... |
| GeminiService | `model`, `config`, ... |
| LiteLLMService | `model`, `temperature`, ... |

For a practical example, see [Quick Response](#-quick-response) — `QuickResponder` uses `inline_llm_params` to disable tool calls and reasoning for fast first-response generation.


### ⏰ Timestamp Insertion

You can insert timestamps into requests at regular intervals. This keeps AIAvatar responses anchored to real-world time.

```python
aiavatar_app = AIAvatar(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    timestamp_interval_seconds=600.0,   # Inserts a timestamp to the request every 600 seconds (10 minutes). Default is 0.
    timestamp_timezone="Asia/Tokyo",    # Default is 'UTC'
)
```

For example, a request of "Hello!" with timestamp insertion enabled becomes:

```
$Current date and time: 2025-12-24

Hello!
```

When `timestamp_interval_seconds` is set to 0, no timestamp is inserted (default).


### 🧵 Request merging

Request merging helps prevent conversation breakdown when speech recognition produces fragmented results. When enabled, consecutive requests within a specified time window are automatically merged into a single request, improving conversation continuity and user experience.


Example without request merging:

```
User: I'm feeling hungry...
AI: Would you... (interrupted mid-sentence while saying "Would you like me to book a restaurant? The place from last time has availability")
User: Uh-huh (misrecognized from "Um..." - a hesitant sound)
AI: Booking completed. (responded to "Uh-huh" and executed restaurant booking)
User: What are you talking about??
```

Example with request merging:

```
User: I'm feeling hungry...
AI: Would you... (interrupted mid-sentence while saying "Would you like me to book a restaurant? The place from last time has availability")
User: Uh-huh (misrecognized from "Um..." - a hesitant sound)
AI: Would you like me to book a restaurant? The place from last time has availability (responding to merged request "I'm feeling hungry... Uh-huh...")
User: Yes, please!
```

To enable this feature, set `merge_request_threshold > 0`.

```python
aiavatar_app.sts.merge_request_threshold = 2.0  # Merge requests within 2 seconds
```

You can also customize the merge prefix message. Here's an example of setting the prefix in Japanese:

```python
aiavatar_app.sts.merge_request_prefix = "$直前のユーザーの要求とあなたの応答はキャンセルされました。以下の要求に対して、あらためて応答しなおしてください:\n\n"
```

NOTE: Files from the previous request are preserved in the merged request


### 📥 Invoke Queue

AIAvatarKit provides three invoke modes for handling concurrent requests. By default, new requests interrupt any ongoing response. With queue mode enabled, you can control whether requests wait in line or still interrupt.

#### Invoke Modes

| Mode | Settings | Behavior |
|------|----------|----------|
| **Direct** (default) | `use_invoke_queue=False` | New requests immediately interrupt the current response. Suitable for most use cases. |
| **Queued (Interrupt)** | `use_invoke_queue=True`, `wait_in_queue=False` | Requests are queued but clear previous pending requests. The current response is interrupted. Default behavior when queue mode is enabled. |
| **Queued (Wait)** | `use_invoke_queue=True`, `wait_in_queue=True` | Requests wait in queue until previous ones complete. No interruption occurs. Useful when you need sequential processing, such as sending a follow-up request (e.g., with an image requested by the server) without interrupting the current response. |

#### Configuration

Enable queue mode on the pipeline:

```python
from aiavatar.sts import STSPipeline

pipeline = STSPipeline(
    # ... other settings ...
    use_invoke_queue=True,              # Enable queue mode
    invoke_queue_idle_timeout=10.0,     # Worker stops after 10s of inactivity
    invoke_timeout=60.0,                # Maximum time for a single invoke
)
```

Or on the AIAvatar instance:

```python
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    use_invoke_queue=True,
)
```

#### Per-Request Behavior

When queue mode is enabled, control per-request behavior via `wait_in_queue`:

```python
from aiavatar.sts import STSRequest

# Interrupt mode (default): clears queue and interrupts current response
request = STSRequest(
    session_id="session1",
    text="Hello!",
    wait_in_queue=False  # default
)

# Wait mode: queues and waits for previous requests to complete
request = STSRequest(
    session_id="session1",
    text="What's next?",
    wait_in_queue=True
)
```

#### Caveats

- **Python 3.11+ required**: Queue mode uses `asyncio.timeout()` which is only available in Python 3.11 and later.
- **Session-based queues**: Each session has its own independent queue. Requests from different sessions do not affect each other.
- **Do not mix modes**: The `use_invoke_queue` setting should remain consistent for a pipeline instance. Changing it at runtime is not supported.
- **Cancelled responses**: When a queued request is cleared (by a non-waiting request), it receives a response with `type="cancelled"`.


### 🧺 Shared Context

Context is typically shared only between an individual user and the AI character. With AIAvatarKit, you can manage histories that define how broadly the context is shared, for example, making it common to every user.

This lets you inject context with general events that are independent of any single user interaction, such as public news or actions the AI character has taken.

```python
# Add character-wide shared messages identified by context_id="shared_context_id"
now = datetime.now(ZoneInfo(self.timezone))
await self.llm.context_manager.add_histories(
    context_id="shared_context_id",
    data_list=[
        {
            "role": "user",
            "content": f"$Current datetime: {now.strftime('%Y/%m/%d %H:%M:%S')}\nToday's news: {news}"
        },
        {
            "role": "assistant",
            "content": "I recognized current datetime and today's news."
        },
    ],
    context_schema="chatgpt"
)
```

```python
# Pass "shared_context_id" via `shared_context_ids` to load the shared history
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt="You are a helpful virtual assistant.",
    shared_context_ids=["shared_context_id"]
)
```


### 🔈 Audio device

You can specify the audio devices to be used in components by device index.

First, check the device indexes you want to use.

```sh
$ python

>>> from aiavatar.device import AudioDevice
>>> AudioDevice().list_audio_devices()
{'index': 0, 'name': '外部マイク', 'max_input_channels': 1, 'max_output_channels': 0, 'default_sample_rate': 44100.0}
{'index': 1, 'name': '外部ヘッドフォン', 'max_input_channels': 0, 'max_output_channels': 2, 'default_sample_rate': 44100.0}
{'index': 2, 'name': 'MacBook Airのマイク', 'max_input_channels': 3, 'max_output_channels': 0, 'default_sample_rate': 44100.0}
{'index': 3, 'name': 'MacBook Airのスピーカー', 'max_input_channels': 0, 'max_output_channels': 2, 'default_sample_rate': 44100.0}
```

Set indexes to AIAvatar.

```python
aiavatar_app = AIAvatar(
    input_device=2,     # MacBook Airのマイク
    output_device=3,    # MacBook Airのスピーカー
    openai_api_key=OPENAI_API_KEY
)
```


### 🐆 Quick Response

To reduce the first response latency, `QuickResponder` generates a short acknowledgment phrase (e.g. "Sure!" or "なるほど。") and sends it to the user immediately, before the main LLM response is ready. This keeps the conversation feeling responsive while the full answer is being generated.

```python
from aiavatar.sts import QuickResponder, DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA, DEFAULT_REQUEST_PREFIX_JA
from aiavatar.sts.models import STSRequest

quick_responder = QuickResponder(
    llm=llm,
    tts=tts,
    quick_response_prompt_prefix=DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA,
    request_prefix=DEFAULT_REQUEST_PREFIX_JA
)

@aiavatar_app.sts.on_before_llm
async def on_before_llm(request: STSRequest):
    await quick_responder.respond(request)
```

`QuickResponder` uses the provided LLM to generate a brief phrase and synthesizes it with the provided TTS (with caching). The generated quick response is stored in the request and yielded by the pipeline as the first chunk. It then rewrites `request.text` so the main LLM response continues naturally without repeating the quick response.

> **Note:** If the main LLM response occasionally includes the quick response content, adding few-shot examples to the initial messages can help stabilize the behavior. You can set them directly via `llm.initial_messages`, or use `CharacterLoader.format_messages` to extend the messages when using `CharacterLoader`.
>
> ```python
> @character_loader.format_messages
> def format_messages(messages):
>     messages.append({"role": "user", "content": quick_responder.quick_response_prompt_prefix + "\n\nHello!"})
>     messages.append({"role": "assistant", "content": "Hello!"})
>     messages.append({"role": "user", "content": quick_responder.request_prefix + "\n\nHello!"})
>     messages.append({"role": "assistant", "content": "<think>Respond warmly to the greeting.</think><answer>Hello! How can I help you today?</answer>"})
>     messages.append({"role": "user", "content": "You repeated 'Hello!' which was already sent. Always continue from where the previous output left off."})
>     messages.append({"role": "assistant", "content": "<think>Noted the mistake. Will not repeat already-sent text next time.</think><answer>Got it.</answer>"})
>     return messages
> ```

#### QuickResponderPro

`QuickResponderPro` is a performance-tuned variant that bypasses `LLMService` and calls the OpenAI-compatible API directly with `stream=False`. It manages its own context through a dedicated `ContextManager`, cleans conversation history for few-shot learning, and supports a custom system prompt — giving you full control over how quick responses are generated.

```python
from aiavatar.sts.quick_responder.pro import QuickResponderPro, DEFAULT_QRP_SYSTEM_PROMPT_JA
from aiavatar.sts.llm.context_manager.postgres import PostgreSQLContextManager
from aiavatar.sts.models import STSRequest

quick_responder_pro = QuickResponderPro(
    api_key="YOUR_OPENAI_API_KEY",
    model="gpt-4.1-nano",
    tts=tts,
    context_manager=PostgreSQLContextManager(get_pool=pool_provider.get_pool),
    language="ja",
    system_prompt=DEFAULT_QRP_SYSTEM_PROMPT_JA + "\n\n# Character\nYour character description here.",
    timeout=1.5,
)

@aiavatar_app.sts.on_before_llm
async def on_before_llm(request: STSRequest):
    await quick_responder_pro.respond(request)
```

**How it works:**

1. Builds messages from system prompt + cleaned history + user utterance
2. Calls the API with `stream=False` for minimum latency
3. Synthesizes the response with TTS (with caching)
4. Rewrites `request.text` with a deduplication prefix so the main LLM continues naturally

**Pre-generation during silence:** When using `SileroStreamSpeechDetector`, you can start generating the quick response during the segment silence period — before turn-end is confirmed. This overlaps LLM + TTS work with the remaining silence wait, noticeably reducing perceived latency.

```python
@vad.on_speech_detecting
async def on_speech_detecting(text, vad_session):
    await quick_responder_pro.create_generation_task(
        text,
        vad_session.session_id,
        vad_session.data.get("context_id")
    )
```

If the user resumes speaking, the pending task is automatically cancelled and a new one starts. If the user stays silent and turn-end is confirmed, `respond()` picks up the pre-generated result instead of generating from scratch.

**History cleaning:** When reading back conversation history, `QuickResponderPro` automatically cleans it for the QR context:
- **Quick response turns** (prompt_prefix) — kept as-is, serving as few-shot examples
- **Main LLM turns** (request_prefix) — replaced with a short continuation message to avoid confusing duplicate utterances
- **Assistant content** — `<think>`/`<answer>` tags and `[control:tags]` are stripped to plain text

**Azure OpenAI / Custom client:** You can pass a pre-configured client instead of `api_key`/`base_url`:

```python
from openai import AsyncAzureOpenAI

quick_responder_pro = QuickResponderPro(
    client=AsyncAzureOpenAI(
        api_key="YOUR_AZURE_API_KEY",
        api_version="2025-01-01-preview",
        azure_endpoint="https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2025-01-01-preview"
    ),
    model="your-deployment-name",
    tts=tts,
    context_manager=context_manager,
)
```

**extra_body:** For providers that require additional request parameters (e.g. disabling thinking for Claude):

```python
quick_responder_pro = QuickResponderPro(
    api_key="YOUR_ANTHROPIC_API_KEY",
    base_url="https://api.anthropic.com/v1/",
    model="claude-haiku-4-5",
    extra_body={"thinking": {"type": "disabled"}},
    tts=tts,
    context_manager=context_manager,
)
```

> **Note:** As with `QuickResponder`, adding few-shot examples to the main LLM's initial messages helps prevent the main response from repeating the quick response. Use `CharacterLoader.format_messages` or set `llm.initial_messages` directly:
>
> ```python
> @character_loader.format_messages
> def format_messages(messages):
>     messages.append({"role": "user", "content": quick_responder_pro.prompt_prefix + "\n\nHello!"})
>     messages.append({"role": "assistant", "content": f"<think>{quick_responder_pro.think_tag_content}</think><answer>Hello!</answer>"})
>     messages.append({"role": "user", "content": quick_responder_pro.request_prefix.format(quick_response_text="Hello!") + "\n\nHello!"})
>     messages.append({"role": "assistant", "content": "<think>Respond warmly to the greeting.</think><answer>How can I help you today?</answer>"})
>     messages.append({"role": "user", "content": "You repeated 'Hello!' which was already sent. Always continue from where the previous output left off."})
>     messages.append({"role": "assistant", "content": "<think>Noted the mistake. Will not repeat already-sent text next time.</think><answer>Got it.</answer>"})
>     return messages
> ```


### 🎭 Custom Behavior

You can invoke custom implementations `on_response(response_type)`. In the following example, show "thinking" face expression while processing request to enhance the interaction experience with the AI avatar.

```python
# Set face when the character is thinking the answer
@aiavatar_app.on_response("start")
async def on_start_response(response):
    await aiavatar_app.face_controller.set_face("thinking", 3.0)

# Reset face before answering
@aiavatar_app.on_response("chunk")
async def on_chunk_response(response):
    if response.metadata.get("is_first_chunk"):
        aiavatar_app.face_controller.reset()
```


### ✅ Request Validation

You can filter out unwanted requests before they reach the LLM by implementing a `validate_request` hook. Return a reason string to cancel the request, or `None` to proceed.

```python
from aiavatar.sts.models import STSRequest

@aiavatar_app.sts.validate_request
async def validate_request(request: STSRequest):
    # Reject text that is too short
    if len(request.text) < 3:
        return "Text too short"

    # Reject requests with too many files
    if request.files and len(request.files) > 5:
        return "Too many files attached"

    # Reject specific users
    if request.user_id == "blocked_user":
        return "User is blocked"

    return None  # Proceed with the request
```

This is useful for:
- Filtering out noise or accidental triggers (e.g., coughs, short utterances)
- Limiting file attachments
- Implementing user-based access control
- Any custom validation logic based on `STSRequest` fields

#### Early Validation with AzureStreamSpeechDetector

When using `AzureStreamSpeechDetector`, you can validate recognized text even earlier—before the STS pipeline is invoked. This is more efficient for filtering out short or invalid utterances since it skips the entire pipeline processing.

```python
from aiavatar.sts.vad.azure_stream import AzureStreamSpeechDetector

speech_detector = AzureStreamSpeechDetector(
    azure_subscription_key=AZURE_SUBSCRIPTION_KEY,
    azure_region=AZURE_REGION,
    azure_language="ja-JP",
)

@speech_detector.validate_recognized_text
def validate_recognized_text(text: str) -> str | None:
    # Reject text that is too short
    if len(text) < 3:
        return "Text too short"

    # Reject specific patterns (e.g., filler words)
    if text in ["えーと", "あの", "うーん"]:
        return "Filler word detected"

    return None  # Proceed with the request
```

Note: This decorator uses a synchronous function (not `async`) because it runs within the Azure Speech SDK's callback thread.


### 🎚️ Noise Filter

AIAvatarKit automatically adjusts the noise filter for listeners when you instantiate an AIAvatar object. To manually set the noise filter level for voice detection, set `auto_noise_filter_threshold` to `False` and specify the `volume_threshold_db` in decibels (dB).

```python
aiavatar_app = AIAvatar(
    openai_api_key=OPENAI_API_KEY,
    auto_noise_filter_threshold=False,
    volume_threshold_db=-40   # Set the voice detection threshold to -40 dB
)
```


### 🔄 Migration Guide: From v0.6.x to v0.7.0

In version **v0.7.0**, the internal Speech-to-Speech pipeline previously provided by the external `LiteSTS` library has been fully integrated into AIAvatarKit.

### What Changed?

- The functionality remains the same — **no API behavior changes**.
- However, **import paths have been updated**.

### 🔧 Required Changes

All imports from `litests` should now be updated to `aiavatar.sts`.

For example:

```python
# Before
from litests import STSRequest, STSResponse
from litests.llm.chatgpt import ChatGPTService

# After
from aiavatar.sts import STSRequest, STSResponse
from aiavatar.sts.llm.chatgpt import ChatGPTService
```

This change ensures compatibility with the new internal structure and removes the need for `LiteSTS` as a separate dependency.
