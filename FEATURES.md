# AIAvatarKit Feature List

**Version 0.8.8** (February 2026)

AIAvatarKit is a Speech-to-Speech framework for building real-time voice-interactive AI avatars. This document systematically summarizes the features of each component.

---

## Table of Contents

1. [VAD (Voice Activity Detection)](#1-vad-voice-activity-detection)
2. [STT (Speech-to-Text)](#2-stt-speech-to-text)
3. [LLM (Large Language Model)](#3-llm-large-language-model)
4. [TTS (Text-to-Speech)](#4-tts-text-to-speech)
5. [Voice Pipeline](#5-voice-pipeline-stspipeline)
6. [Channels/Adapters](#6-channelsadapters)
7. [Database (PostgreSQL)](#7-database-postgresql)
8. [Character Management](#8-character-management)
9. [Device Control](#9-device-control)
10. [Avatar Control](#10-avatar-control)
11. [Administration](#11-administration)
12. [Evaluation](#12-evaluation)

---

## 1. VAD (Voice Activity Detection)

A component that detects the start and end of speech to extract utterance segments. **Accurately detecting when the user finishes speaking** enables natural turn-taking in conversations.

### 1.1 Supported Implementations

| Implementation | Detection Method | Characteristics | Use Case |
|----------------|------------------|-----------------|----------|
| **StandardSpeechDetector** | Volume threshold-based | Fast, lightweight, runs locally | Simple detection in quiet environments |
| **SileroSpeechDetector** | Deep learning model | High accuracy, noise resistant | Noisy environments or when high precision is required |
| **SileroStreamSpeechDetector** | Deep learning model + Streaming STT | Real-time segment recognition during recording | Low-latency response with partial transcription feedback |
| **AzureStreamSpeechDetector** | Cloud API | Can retrieve real-time recognition text | When displaying recognition results during speech |

### 1.2 Common Features

| Feature | Description |
|---------|-------------|
| **Silence detection threshold** | `silence_duration_threshold`: Silence duration (seconds) to determine end of speech. Too short cuts off mid-speech; too long delays response |
| **Maximum recording time** | `max_duration`: Maximum recording time per utterance. Limits long monologues and controls memory usage |
| **Minimum recording time** | `min_duration`: Minimum speech duration to trigger callback. Ignores short sounds from noise or false detections |
| **Pre-roll buffer** | Retains audio before speech starts to prevent cutting off the beginning. Prevents "Hello" becoming "ello" |
| **Session management** | Supports concurrent processing of multiple sessions (users). Handles simultaneous connections in web services |
| **Mute control** | Dynamic mute control via `should_mute` callback. Ignores user voice while avatar is speaking |
| **Session data storage** | Stores custom data per session. Retains metadata like user_id |

### 1.3 SileroSpeechDetector-Specific Features

| Feature | Description |
|---------|-------------|
| **Speech probability threshold** | `speech_probability_threshold`: Confidence threshold for speech detection (0.0-1.0). Lower is more sensitive, higher is less sensitive |
| **Model pooling** | `model_pool_size`: Pool multiple models for concurrent processing. Improves throughput in multi-user environments |
| **VAD iterator mode** | `use_vad_iterator`: Smoothing through state tracking. Prevents false detections from brief silence |
| **Custom model** | `model_path`: Specify custom Silero model file. Use models trained for specific environments |
| **Hybrid detection** | Can combine volume threshold and VAD model. Filters out quiet voices as noise |

### 1.4 SileroStreamSpeechDetector-Specific Features

Extends `SileroSpeechDetector` with segment-based speech recognition. Performs partial transcription during recording, **eliminating STT latency after VAD completes**.

**Any `SpeechRecognizer` can be used** for segment recognition. This enables converting non-streaming STT services into streaming ones with turn-end detection.

| Feature | Description |
|---------|-------------|
| **Flexible STT integration** | `speech_recognizer`: Any `SpeechRecognizer` implementation (Google, Azure, OpenAI, etc.) |
| **Segment recognition** | Runs STT on speech segments while recording continues. Returns partial results before final transcription |
| **Segment silence threshold** | `segment_silence_threshold`: Silence duration to trigger segment recognition |
| **on_speech_detecting callback** | Callback for partial recognition text. Display intermediate results in UI |
| **Text validation** | `@validate_recognized_text` decorator to filter invalid recognition results |
| **Custom trigger condition** | `@should_trigger_recording_started` decorator for custom callback trigger logic |

### 1.5 AzureStreamSpeechDetector-Specific Features

| Feature | Description |
|---------|-------------|
| **Real-time recognition** | `on_speech_detecting`: Callback for partial recognition text during speech. Display input text in UI |
| **Multi-language support** | `azure_language`: Specify recognition language |
| **Pre-roll time specification** | `preroll_buffer_sec`: Specify pre-roll time in seconds |
| **Text validation** | `@validate_recognized_text` decorator to filter invalid recognition results |

### 1.6 VAD Callbacks

Extension points for custom processing at various VAD stages. **Multiple callbacks can be registered for each event**.

| Callback | Description |
|----------|-------------|
| **@on_speech_detected** | Triggered when speech ends. Receives final audio data for STT processing |
| **@on_speech_detecting** | Triggered during speech (stream detectors). Receives partial recognition text |
| **@on_recording_started** | Triggered when recording becomes meaningful. Stop AI speech on user barge-in |
| **@on_voiced** | Triggered when voice activity is first detected. Immediate notification of speech start |
| **@on_speech_recognition_error** | Triggered on STT errors (stream detectors). Handle recognition failures gracefully |

```python
# Multiple callbacks can be registered for the same event
@vad.on_speech_detected
async def callback_1(session_id, audio_data, recognized_text):
    await log_speech(session_id, recognized_text)

@vad.on_speech_detected
async def callback_2(session_id, audio_data, recognized_text):
    await update_ui(session_id, recognized_text)
```

---

## 2. STT (Speech-to-Text)

A component that converts speech to text. Converts utterance segments extracted by VAD into text for LLM input.

### 2.1 Supported Implementations

| Implementation | Provider | Characteristics | Use Case |
|----------------|----------|-----------------|----------|
| **GoogleSpeechRecognizer** | Google Cloud | Multi-language support, alternative language support | General-purpose speech recognition |
| **AzureSpeechRecognizer** | Azure | Two API modes, custom model support | When specialized terminology is common |
| **OpenAISpeechRecognizer** | OpenAI Whisper | Automatic language detection, high accuracy | Multi-language mixed environments |
| **AmiVoiceSpeechRecognizer** | AmiVoice | Japanese-specialized, grammar engine selection | High-accuracy Japanese recognition |

### 2.2 Common Features

| Feature | Description |
|---------|-------------|
| **Language specification** | `language`: Primary language code (e.g., "ja-JP", "en-US"). Improves recognition accuracy |
| **Alternative languages** | `alternative_languages`: Fallback languages for multi-language detection. Auto-switching in multilingual environments |
| **Preprocessing hook** | `@preprocess` decorator for audio data preprocessing. Noise removal and format conversion |
| **Postprocessing hook** | `@postprocess` decorator for recognition text postprocessing. Correction of misrecognitions and normalization |
| **Retry feature** | `max_retries`: Auto-retry count on failure. Handles temporary API failures |
| **Downsampling** | Audio sample rate conversion. Matches API format requirements |

### 2.3 Azure-Specific Features

| Feature | Description |
|---------|-------------|
| **Classic mode** | v1 API, custom model support |
| **Fast mode** | v2024-11-15 API, automatic multi-language detection |
| **Custom model** | `cid`: Specify custom model ID |

### 2.4 Speaker Identification

A feature to identify **who is speaking** in multi-person environments. Useful for meeting rooms and event venues.

#### MainSpeakerGate (Speaker Gate)
Identifies the main speaker within a session and filters out other speakers' utterances. **Ignores voices other than the conversation partner**.

| Feature | Description |
|---------|-------------|
| **Pair lock** | Locks as main speaker after two consecutive similar utterances. Prevents misidentification |
| **Accept threshold** | Similarity threshold after lock (default: 0.55). Tolerates voice variations (e.g., having a cold) |
| **Lock threshold** | Similarity threshold for main speaker lock (default: 0.72). Ensures reliable identity confirmation |
| **Embedding generation** | Generates 256-dimensional embeddings with Resemblyzer VoiceEncoder |

#### SpeakerRegistry (Speaker Registry)
A system to register and match known speakers. Can **identify "who is speaking" by name**.

| Feature | Description |
|---------|-------------|
| **Top-K matching** | Returns multiple candidates for most similar speakers. Confidence available |
| **Metadata management** | Stores custom metadata per speaker. Name, title, etc. |
| **Persistence** | InMemoryStore (file) or PGVectorStore (PostgreSQL) |

---

## 3. LLM (Large Language Model)

A component that generates conversation responses.

### 3.1 Supported Implementations

| Implementation | Provider | Characteristics |
|----------------|----------|-----------------|
| **ChatGPTService** | OpenAI / Azure OpenAI | Tool calling, o1/o3 reasoning model support |
| **ClaudeService** | Anthropic | Content block streaming |
| **GeminiService** | Google | Extended thinking mode, image download support |
| **LiteLLMService** | Multiple providers | Model-agnostic unified interface |
| **DifyService** | Dify | External workflow integration |

### 3.2 Common Features

| Feature | Description |
|---------|-------------|
| **System prompt** | Defines character settings and instructions |
| **Temperature control** | `temperature`: Controls response randomness (0.0=deterministic, 1.0=creative) |
| **Streaming response** | Streams text in chunks. Reduces response latency by starting TTS processing in parallel with LLM generation |
| **Context management** | Saves/retrieves conversation history (SQLite/PostgreSQL). Maintains context for multi-turn conversations |
| **Shared context** | `shared_context_ids`: Integrates context from multiple sessions. Used for knowledge sharing between bots or injecting global information |
| **Initial messages** | `initial_messages`: Sample messages for few-shot learning. Guides LLM output by demonstrating response style and format |

### 3.3 Text Splitting (Low-Latency Voice Response)

A feature that splits LLM response streams at punctuation and **executes TTS processing in parallel with LLM generation** to speed up voice response delivery to users.

| Feature | Description |
|---------|-------------|
| **Split characters** | `split_chars`: Split at sentence-ending marks (default: . ? ! \n). Sends to TTS when sentence is complete |
| **Optional split** | `option_split_chars`: Additional splitting at commas, etc. Splits long sentences midway for earlier vocalization |
| **Split threshold** | `option_split_threshold`: Minimum character count for optional splitting. Prevents too-short splits |
| **Control tag split** | `split_on_control_tags`: Split before [tag:value]. Controls timing of facial expression changes, etc. |
| **Voice text tag** | `voice_text_tag`: Extract TTS text via XML tags. Vocalize only the answer portion without reading thinking process |

```
[Processing Flow Example]
LLM output: "Hello. Today is" → Split at "." → Send "Hello" to TTS
LLM output: "nice weather." → Split at "." → Send "Today is nice weather" to TTS
                ↓
User starts hearing "Hello" before LLM completes full text generation
```

### 3.4 Zero-shot CoT (Think Before Answering)

A feature that enables Chain-of-Thought (CoT) where AI "thinks before answering". The thinking process is not vocalized; only the answer portion is sent to TTS.

| Feature | Description |
|---------|-------------|
| **think/answer tags** | Have LLM output in `<think>thinking</think><answer>response</answer>` format |
| **voice_text_tag** | Setting `voice_text_tag="answer"` vocalizes only text within `<answer>` tags |
| **Non-vocalization of thinking** | Thinking process (within `<think>`) is not vocalized but contributes to response quality |

```python
# Usage example: Instruct CoT format in system prompt
system_prompt = """
Think within <think> tags before answering.
Write your answer within <answer> tags.
Example: <think>This question is...</think><answer>Yes, that's right.</answer>
"""

llm = ChatGPTService(
    system_prompt=system_prompt,
    voice_text_tag="answer"  # Vocalize only answer tag content
)
```

### 3.5 Tool Calling

Function Calling feature for LLM to call external functions (APIs, databases, web search, etc.).

| Feature | Description |
|---------|-------------|
| **Tool registration** | Register tool functions with `@service.tool` decorator. Auto-converts to OpenAI/Claude/Gemini formats |
| **Dynamic tool selection** | `use_dynamic_tools`: When many tools exist, LLM selects only relevant ones. Reduces prompt size |
| **Tool filtering** | Automatic relevant tool selection by LLM. Narrows 100 tools to 5, etc. |
| **Streaming results** | Stream intermediate results of tool execution. Notifies user of progress for long-running processes |
| **MCP integration** | Model Context Protocol server integration via StdioMCP/StreamableHttpMCP |

### 3.6 Built-in Tools

Ready-to-use built-in tools.

| Tool | Description |
|------|-------------|
| **GrokSearchTool** | Web search via X.AI Grok. Used for retrieving latest information |
| **GeminiWebSearchTool** | Google search (via Gemini). Can stream progress |
| **OpenAIWebSearchTool** | OpenAI Web search. Region/language specifiable |
| **WebScraperTool** | Web scraping using Playwright. Gets content after JavaScript rendering |
| **NanoBananaTool** | Gemini image generation. Generates images from text |
| **NanoBananaSelfieTool** | Character selfie generation. Specify expression, outfit, background |

### 3.7 Guardrails

Safety features for filtering inappropriate content and detecting/replacing specific patterns. **Parallel execution** and **async evaluation** ensure safety without compromising user experience.

#### Parallel Execution & Async Evaluation Features

Typical guardrail implementations cause delays by making users wait until evaluation completes. AIAvatarKit's guardrails take a different approach:

**Request Guardrails (Parallel Execution, Sync Evaluation)**
- Execute multiple guardrails **in parallel** via `asyncio.as_completed`
- Automatically cancel remaining tasks when first guardrail triggers
- **Wait for evaluation to complete** before proceeding to LLM processing (sync evaluation)
- On block: immediately return error message; on replace: continue LLM processing with modified text

**Response Guardrails (Parallel Execution, Async Evaluation)**
- Execute multiple guardrails **in parallel** via `asyncio.as_completed`
- LLM response stream **starts returning to user immediately** (don't wait for evaluation = async evaluation)
- Guardrails evaluate after stream completes
- On issue detection: **output corrective utterance** (e.g., "Let me correct what I just said...")

This design provides:
- **Low latency**: Don't block response start. Users begin receiving responses immediately
- **Natural correction**: Issues are corrected via additional utterance, like natural human conversation
- **Scalable**: Response start time unaffected as guardrail count increases

| Feature | Description |
|---------|-------------|
| **Request guardrails** | Filter user input. Block/replace inappropriate input |
| **Response guardrails** | Filter LLM responses. Output corrective utterance on issue detection |
| **Block action** | Interrupt processing on condition match, return error message |
| **Replace action** | Replace text on condition match. Request: modify and continue; Response: corrective utterance |
| **Parallel execution** | Process multiple guardrails in parallel via `asyncio.as_completed` |
| **Early cancellation** | Automatically cancel remaining tasks on first trigger |

```python
# Response guardrail behavior example
# 1. LLM response: "I love ramen" → Immediately starts streaming to user
# 2. After stream completes, guardrail detects "ramen"
# 3. Outputs corrective utterance: "Let me correct what I just said..."
# → User isn't blocked waiting; issues are naturally corrected
```

### 3.8 Callbacks/Hooks

Extension points for inserting custom logic at each processing stage.

| Hook | Description |
|------|-------------|
| **@request_filter** | Preprocess request text. Used for normalization and correction |
| **@update_context_filter** | Filter before saving history. Remove sensitive information, etc. |
| **@get_system_prompt** | Dynamic system prompt generation. Switch prompts based on user or situation |
| **@get_dynamic_tools** | Dynamic tool selection logic. Control selection from large tool sets |
| **@on_before_tool_calls** | Process before tool execution. Used for logging or approval flow insertion |
| **@on_error** | Handle LLM API errors. Customize avatar response on content filter or API failures |
| **@print_chat** | Customize conversation logging. Custom formatting for user/AI turns |

```python
# Error handling example
@llm.on_error
async def on_error(llm_response: LLMResponse):
    if "content_filter" in str(llm_response.error_info):
        llm_response.text = "Sorry, I cannot respond to that."
    else:
        llm_response.text = "An error occurred. Please try again."

# Custom chat logging example
@llm.print_chat
def print_chat(role, context_id, user_id, text, files):
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = "\033[94m" if role == "user" else "\033[92m"
    print(f"{color}[{timestamp}] {role}: {text}\033[0m")
```

### 3.9 Observability

A feature to **monitor** request contents to LLM, interpretation results, tool calls, and generated responses. Used for quality improvement and governance.

| Feature | Description |
|---------|-------------|
| **OpenAI module replacement** | Replace OpenAI client via `custom_openai_module` parameter |
| **Langfuse integration** | Pass Langfuse-compatible module to auto-collect traces and logs |
| **Trace visualization** | Visualize the entire flow from request → tool calls → response |

```python
# Langfuse integration example
from langfuse.openai import openai as langfuse_openai

llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt="You are a helpful assistant.",
    model="gpt-4.1",
    custom_openai_module=langfuse_openai,  # Set Langfuse-compatible module
)
# → LLM requests/responses are automatically recorded in Langfuse dashboard
```

---

## 4. TTS (Text-to-Speech)

A component that converts text to speech. Outputs LLM responses as the **avatar's voice**.

### 4.1 Supported Implementations

| Implementation | Provider | Characteristics | Use Case |
|----------------|----------|-----------------|----------|
| **VoicevoxSpeechSynthesizer** | VOICEVOX/AivisSpeech | Japanese-specialized, two-stage synthesis | Japanese character voices |
| **AzureSpeechSynthesizer** | Azure | SSML support | Multi-language, emotional expression |
| **GoogleSpeechSynthesizer** | Google Cloud | Multi-language support | General-purpose speech synthesis |
| **OpenAISpeechSynthesizer** | OpenAI | Voice instruction feature, Azure OpenAI support | Natural English voice |
| **SpeechGatewaySpeechSynthesizer** | Custom | General gateway | Style-Bert-VITS2 integration, etc. |
| **create_instant_synthesizer** | Custom | Factory function for custom TTS creation | Integration with proprietary TTS services |

### 4.2 Common Features

| Feature | Description |
|---------|-------------|
| **Timeout** | `timeout`: HTTP request timeout |
| **Debug mode** | `debug`: Detailed log output |
| **Follow redirects** | `follow_redirects`: Follow HTTP redirects |

### 4.3 Style Application

A feature to **dynamically switch voice style (speaker/emotion)** based on keywords in LLM output.

| Feature | Description |
|---------|-------------|
| **Style mapper** | `style_mapper`: Dictionary mapping keywords to voice styles |
| **Dynamic speaker selection** | Switch speaker/style based on LLM output keywords. Angry lines in angry voice |
| **VOICEVOX speaker ID** | Convert style name → speaker ID (integer) |

```python
# Style mapper example: Switch to angry voice if "angry" is in LLM output
style_mapper = {
    "angry": "3",  # VOICEVOX: Zundamon (angry)
    "happy": "1",  # VOICEVOX: Zundamon (happy)
}

# Have LLM output tags like [face:angry],
# extract style info in @process_llm_chunk and pass to TTS
```

### 4.4 Pronunciation Dictionary (Preprocessors)

A feature to **convert strings TTS can't pronounce correctly into readable format**. Controls pronunciation of numbers and foreign words.

#### PatternMatchPreprocessor
Regex-based text replacement.

| Feature | Description |
|---------|-------------|
| **String replacement** | Simple string replacement. "AIAvatar" → "A I Avatar" |
| **Regex replacement** | Use regex patterns with `regex=True`. Flexible pattern matching |
| **Callback replacement** | Custom replacement logic via functions. Handles complex conversion rules |
| **Number replacement** | `add_number_dash_pattern()`: 123-456 → one two three dash four five six |
| **Phone number replacement** | `add_phonenumber_pattern()`: 090-1234-5678 → zero nine zero... |

```python
# Usage example
preprocessor = PatternMatchPreprocessor()
preprocessor.add_pattern("OpenAI", "Open A I")
preprocessor.add_number_dash_pattern()  # Number-dash pattern replacement

tts = VoicevoxSpeechSynthesizer(preprocessors=[preprocessor])
```

#### AlphabetToKanaPreprocessor
Converts foreign words to Katakana using LLM. Supports **kana_map** for storing word-to-reading mappings to reduce latency on repeated words.

| Feature | Description |
|---------|-------------|
| **AI conversion** | Convert foreign words to Katakana reading via OpenAI GPT. Apple → アップル |
| **kana_map** | Pre-register word-reading mappings and automatically add LLM results. Avoids repeated API calls for known words |
| **special_chars** | Characters that connect words (default: `.'-'−–`). Words like `Mr.`, `You're`, `Wi-Fi` are always processed |
| **Case-insensitive** | Matches `API`, `api`, and `Api` with a single kana_map entry |
| **Detection threshold** | `alphabet_length`: Minimum consecutive characters for conversion (default: 3). Excludes short abbreviations |
| **Persistence** | Export/import `kana_map` as JSON for reuse across sessions |
| **Debug logging** | Logs `[KanaMap]` for cached hits and `[LLM]` for new readings with elapsed time |

### 4.5 Audio Format Conversion

#### AudioConverter
ffmpeg-based audio format conversion. **Converts TTS output to match playback environment**.

| Feature | Description |
|---------|-------------|
| **Output format** | wav, mp3, flac, etc. Convert to format supported by client |
| **Sample rate conversion** | Convert to any sample rate. Supports telephony (8kHz) or web (16kHz) |
| **Channel conversion** | Mono/stereo conversion |

### 4.6 Local Gateway Mode

A feature to execute TTS processing **directly in-process** without HTTP overhead. Available for `SpeechGatewaySpeechSynthesizer`.

| Feature | Description |
|---------|-------------|
| **Local execution** | `use_local_gateway=True`: Execute TTS locally using `speech_gateway` library |
| **Zero network latency** | Eliminates HTTP request overhead for faster synthesis |
| **Easy gateway setup** | `add_local_gateway()`: Single method to register gateway with speaker and language settings |

```python
# Local gateway mode example
from speech_gateway.gateway import SBV2Gateway

tts = SpeechGatewaySpeechSynthesizer(use_local_gateway=True)
tts.add_local_gateway(
    name="sbv2",
    gateway=SBV2Gateway(base_url="http://127.0.0.1:5000"),
    speaker="default",
    languages=["ja-JP"],
    default=True
)
```

### 4.7 Per-Session Speech Speed Control

A feature to **dynamically adjust speech speed per session**. Enables personalized voice output for each user.

| Feature | Description |
|---------|-------------|
| **Session-based speed** | Set `speed` in `style_info` via `@process_llm_chunk` hook |
| **User preference** | Store speech speed in session data for personalization |
| **Real-time adjustment** | Change speed without restarting the application |

```python
# Apply speech speed per session
@aiavatar_app.sts.process_llm_chunk
async def process_llm_chunk(llm_stream_chunk: LLMResponse, session_id: str, user_id: str) -> dict:
    if session_data := aiavatar_app.sessions.get(session_id):
        if speed := session_data.data.get("tts_speed"):
            return {"speed": float(speed)}
```

---

## 5. Voice Pipeline (STSPipeline)

An end-to-end pipeline integrating VAD → STT → LLM → TTS. Coordinates components to execute the **complete flow from voice input to voice response**.

### 5.1 Request Merging

A feature to combine consecutive utterances into a single request. When users say "um" "well" and rephrase, **processes multiple utterances as one question**.

| Setting | Description |
|---------|-------------|
| **merge_request_threshold** | Time threshold for merging (seconds). 0 disables. 3 seconds merges utterances within 3 seconds |
| **merge_request_prefix** | Prefix to inform LLM during merge. "Previous utterance was canceled", etc. |
| **allow_merge** | Control merge permission per request. Don't merge specific utterances, etc. |

```python
# Example: Merge consecutive utterances within 3 seconds
# User: "Tomorrow's" (1 second later) "weather please" → Processed as "Tomorrow's weather please"
pipeline = STSPipeline(
    merge_request_threshold=3.0,
    merge_request_prefix="$Previous request and response were canceled. Please respond to the following request:\n\n"
)
```

### 5.2 Timestamp Insertion

Automatically inserts current datetime into LLM context. Enables character to **answer "What time is it?"**.

| Setting | Description |
|---------|-------------|
| **timestamp_interval_seconds** | Insertion interval (seconds). 0 disables. 60 updates every minute |
| **timestamp_prefix** | Timestamp prefix. "Current datetime: ", etc. |
| **timestamp_timezone** | Timezone (e.g., "Asia/Tokyo"). Matches user's region |

### 5.3 Wake Word

A mode that activates on specific keywords. **Smart speaker-like behavior that doesn't respond until called with "Hey, XX"**.

| Setting | Description |
|---------|-------------|
| **wakewords** | List of activation keywords. ["Hey Avatar", "OK Avatar"], etc. |
| **wakeword_timeout** | Validity period after activation (seconds). 60 seconds means no wake word needed for 1 minute |

### 5.4 Queue Management

Controls sequential request processing. Controls **behavior when multiple requests arrive simultaneously**.

| Setting | Description |
|---------|-------------|
| **use_invoke_queue** | Enable queue-based processing. Sequential processing prevents mixed responses |
| **invoke_queue_idle_timeout** | Worker idle timeout. Resource release |
| **invoke_timeout** | Request processing timeout. Prevents infinite waiting |
| **wait_in_queue** | true: Wait for previous request completion, false: Cancel previous request |

### 5.5 Performance Recording

Detailed processing time recording. Used for **bottleneck identification** and **quality improvement**.

| Metric | Description |
|--------|-------------|
| **stt_time** | Speech recognition time. STT provider performance evaluation |
| **stop_response_time** | Previous response stop time. Barge-in delay |
| **llm_first_chunk_time** | LLM first token time (TTFT). Directly affects perceived latency |
| **llm_first_voice_chunk_time** | First voice text time. Answer portion start time when using CoT |
| **processing_time** | Time from LLM first chunk to first voice chunk. Includes tool execution and preprocessing |
| **llm_time** | Total LLM processing time |
| **tts_first_chunk_time** | TTS first audio time. Time until audio playback starts |
| **tts_time** | Total TTS processing time |
| **total_time** | End-to-end time. Total user-perceived delay |
| **tool_calls** | Detailed tool call logging. Function name, arguments, and results |
| **error_info** | Error tracking. Captures STT/LLM/TTS failures with traceback |

### 5.6 Voice Recording

Request/response audio saving. Used for **quality verification, debugging, and training data collection**.

| Setting | Description |
|---------|-------------|
| **voice_recorder_enabled** | Enable recording |
| **voice_recorder_dir** | Save directory |
| **voice_recorder_response_audio_format** | Response audio format |

### 5.7 Callbacks/Hooks

Insert custom logic at each pipeline stage.

| Hook | Description |
|------|-------------|
| **@validate_request** | Filter requests before LLM. Return cancel reason or None to proceed |
| **@on_before_llm** | Before LLM processing. Request modification or logging |
| **@on_before_tts** | Before TTS processing. Voice synthesis preparation (speaker selection, etc.) |
| **@on_finish** | After response completion. Logging or post-processing |
| **@process_llm_chunk** | Parse LLM chunks. Extract style info from [face:joy] tags and pass to TTS |
| **handle_response** | Customize response sending. Send processing for WebSocket/HTTP, etc. |
| **stop_response** | Customize response stop. Audio playback stop processing |

```python
# Request validation example
@aiavatar_app.sts.validate_request
async def validate_request(request: STSRequest):
    if len(request.text) < 2:
        return "Text too short"  # Request canceled with this reason
    if len(request.files) > 5:
        return "Too many files"
    return None  # Proceed with request
```

### 5.8 Session State Management

Tracks state per session. Enables **state isolation in multi-user environments**.

| Feature | Description |
|---------|-------------|
| **active_transaction_id** | Currently processing transaction ID. New requests cancel old processing |
| **previous_request_timestamp** | Previous request timestamp. Used for merge judgment |
| **previous_request_text** | Previous request text. Combined during merge |
| **timestamp_inserted_at** | Last timestamp insertion time. Used for interval control |

---

## 6. Channels/Adapters

Connection interfaces to external systems. An abstraction layer for **running avatars on various platforms**.

### 6.1 Supported Adapters

| Adapter | Purpose | Characteristics | Use Case |
|---------|---------|-----------------|----------|
| **HTTP** | REST API | SSE streaming, Dify compatible | Web apps, API integration |
| **WebSocket** | Real-time bidirectional | PCM chunking, barge-in support | Interactive UI |
| **WebSocket STT** | Streaming speech recognition | VAD + STT only, no LLM/TTS | Speech-to-text microservices |
| **LINE Bot** | LINE Messenger | Session persistence, image upload | Customer support on LINE |
| **Local** | Local execution | No network required, embedded | Desktop apps, embedded devices |

### 6.2 HTTP Adapter Features

#### Server-side
| Feature | Description |
|---------|-------------|
| **POST /chat** | SSE streaming response. Low-latency chunk-by-chunk response |
| **POST /chat-messages** | Dify-compatible endpoint. Integration with Dify platform |
| **POST /transcribe** | Speech → text conversion. Standalone STT usage |
| **POST /synthesize** | Text → speech conversion. Standalone TTS usage |
| **Speaker identification** | SpeakerRegistry integration. Identify who is speaking |
| **Bearer authentication** | Authentication via api_key. Prevent unauthorized access |

#### Client-side
| Feature | Description |
|---------|-------------|
| **VAD integration** | Speech detection via Silero VAD. Controls server send timing |
| **Auto noise threshold** | Threshold adjustment based on ambient noise. Auto-adjusts for quiet rooms vs noisy places |
| **Barge-in** | Interruption detection during speech. Interrupts avatar speech to prioritize user speech |
| **Echo cancellation** | Suppress mic input during playback. Don't pick up sound from speakers |

### 6.3 WebSocket Adapter Features

Real-time bidirectional communication. Optimal for **streaming voice transmission/reception**.

#### Message Types
| Type | Direction | Description |
|------|-----------|-------------|
| **start** | C→S | Session start. Set user_id, context_id |
| **invoke** | C→S | Request submission. Text or audio data |
| **data** | C→S | Audio data streaming. Real-time transmission |
| **config** | C→S | VAD configuration change. Sensitivity adjustment, etc. |
| **stop** | C→S | Session end. Resource release |
| **connected** | S→C | Connection confirmation |
| **chunk** | S→C | Response chunk. Partial response with text + audio |
| **final** | S→C | Final response. Full text |

#### PCM Chunking
Split audio into small chunks for transmission. Enables **streaming playback**.

| Setting | Description |
|---------|-------------|
| **response_audio_chunk_size** | Audio chunk size (0=no splitting). Smaller means lower latency |
| **PCM format metadata** | sample_rate, channels, sample_width. Needed for client-side reconstruction |

### 6.4 WebSocket STT Server Features

Standalone streaming speech recognition server. Provides **VAD + STT as a microservice** without LLM/TTS processing.

#### StreamSpeechRecognitionServer
| Feature | Description |
|---------|-------------|
| **WebSocket streaming** | Real-time audio streaming from client to server |
| **VAD integration** | Supports both SileroStreamSpeechDetector and batch VAD modes |
| **Partial results** | Stream mode sends partial recognition results during speech |
| **Voiced event** | Immediate notification when voice activity is detected |
| **Connection callbacks** | `@on_connect` and `@on_disconnect` decorators for session lifecycle |

#### Message Types
| Type | Direction | Description |
|------|-----------|-------------|
| **start** | C→S | Initialize session with configuration |
| **data** | C→S | Base64-encoded audio chunk |
| **config** | C→S | Update VAD configuration mid-session |
| **stop** | C→S | End session |
| **connected** | S→C | Session established confirmation |
| **partial** | S→C | Partial recognition result (stream mode) |
| **final** | S→C | Final recognition result |
| **voiced** | S→C | Voice activity detected notification |
| **error** | S→C | Recognition error |

### 6.5 Local Adapter Features

In-process execution without network overhead. Optimal for **desktop applications and embedded devices**.

#### AIAvatarLocalServer
| Feature | Description |
|---------|-------------|
| **Direct pipeline integration** | Direct STSPipeline integration without HTTP/WebSocket overhead |
| **Response queue** | Queue-based communication between client and server |
| **Session management** | Session data management through server interface |
| **Avatar control parsing** | Parse face/animation tags from LLM output |
| **Callback support** | Full support for on_session_start/on_request/on_response hooks |

### 6.6 LINE Bot Adapter Features

Integration with LINE Messenger. Specialized for **text-based conversations**.

| Feature | Description |
|---------|-------------|
| **Session management** | Session persistence via SQLite/PostgreSQL. Continuous conversations with users |
| **Session timeout** | Auto-deletion of inactive sessions. Reset old conversations |
| **Image upload** | Save/reference user images. Integration with Vision feature |
| **Message parser** | Text/image/sticker/location. Handles various LINE message types |
| **Event handler** | Custom event processing. Follow/block events, etc. |

### 6.7 Callback Hooks

Extension points available across all adapters. Insert custom logic at session and request lifecycle stages.

| Hook | Description |
|------|-------------|
| **@on_session_start** | Called when session is initialized. Use for user initialization, loading preferences |
| **@on_request** | Called before processing each request. Use for request logging, validation, modification |
| **@on_response** | Called for each response chunk. Use for custom response processing, metadata enrichment |

```python
# Callback hook example
@adapter.on_session_start
async def handle_session_start(request, session_data):
    # Initialize user, load preferences, etc.
    pass

@adapter.on_request
async def handle_request(request):
    # Log request, validate, modify, etc.
    pass

@adapter.on_response
async def handle_response(response, sts_response):
    # Process response, add metadata, etc.
    pass
```

### 6.8 Common Features

Features available across all adapters.

| Feature | Description |
|---------|-------------|
| **Avatar control tags** | Parse [face:name], [animation:name]. Extract avatar control from LLM output |
| **Language detection** | Parse [language:code] tags. Multi-language TTS switching |
| **Vision request** | Parse [vision:source] tags. Camera image retrieval request |
| **Base64 encoding** | Base64 conversion of audio data. Send/receive via JSON |

---

## 7. Database (PostgreSQL)

Data persistence layer for production environments. **SQLite is used by default for development**, but PostgreSQL is recommended for production deployments requiring scalability and reliability.

### 7.1 Supported Components

Multiple components can share a single PostgreSQL connection pool for efficient resource usage.

| Component | SQLite | PostgreSQL | Description |
|-----------|--------|------------|-------------|
| **ContextManager** | ○ | ○ | Conversation history persistence |
| **SessionStateManager** | ○ | ○ | Dialogue state management |
| **PerformanceRecorder** | ○ | ○ | Response time and metrics recording |
| **SpeakerRegistry** | File | pgvector | Speaker embedding storage with vector search |
| **LineBotSessionManager** | ○ | ○ | LINE Bot per-user sessions |
| **CharacterService** | - | ○ | Character schedules, diaries, memory (**PostgreSQL required**) |

### 7.2 PostgreSQL Pool Provider

Centralized connection pool management. **All components share a single pool** to avoid connection exhaustion.

| Feature | Description |
|---------|-------------|
| **Shared pool** | Multiple components share one connection pool |
| **Lazy initialization** | Pool created on first access, not at startup |
| **Configurable size** | `min_size` (default: 5), `max_size` (default: 20) |
| **Pool statistics** | `get_stats()` returns pool utilization metrics |
| **Thread-safe** | Safe for concurrent access from multiple coroutines |

```python
from aiavatar.database.postgres import PostgreSQLPoolProvider

# Create shared pool provider
pool_provider = PostgreSQLPoolProvider(
    connection_str="postgresql://user:pass@host:5432/db",
    min_size=5,
    max_size=30
)

# Pass to components - all share the same pool
from aiavatar.sts.context.postgres import PostgreSQLContextManager
from aiavatar.sts.session.postgres import PostgreSQLSessionStateManager
from aiavatar.sts.performance.postgres import PostgreSQLPerformanceRecorder
from aiavatar.character import CharacterService

context_manager = PostgreSQLContextManager(get_pool=pool_provider.get_pool)
session_manager = PostgreSQLSessionStateManager(get_pool=pool_provider.get_pool)
performance_recorder = PostgreSQLPerformanceRecorder(get_pool=pool_provider.get_pool)
character_service = CharacterService(db_pool_provider=pool_provider)
```

### 7.3 Why Use Pool Provider?

Without centralized pool management, each component creates its own connections:

| Approach | Connections | Issue |
|----------|-------------|-------|
| **Without Pool Provider** | 5 components × 20 connections = 100 | Connection exhaustion, resource waste |
| **With Pool Provider** | 1 pool × 30 connections = 30 | Efficient sharing, predictable resource usage |

**Best Practices:**
- Create `PostgreSQLPoolProvider` once at application startup
- Pass `get_pool` function to all components
- Monitor pool stats in production via `pool_provider.get_stats()`
- Set `max_size` based on your database's `max_connections` setting

### 7.4 Migration from SQLite

AIAvatarKit uses SQLite by default (`aiavatar.db`). For production:

1. **Install dependencies**: `pip install asyncpg` (and `pgvector` for speaker registry)
2. **Create PostgreSQL database** with required extensions
3. **Replace managers** with PostgreSQL versions
4. **Share pool** via `PostgreSQLPoolProvider`

```python
# Before (SQLite - default)
aiavatar_app = AIAvatar(openai_api_key=API_KEY)

# After (PostgreSQL)
pool_provider = PostgreSQLPoolProvider(connection_str=DB_URL)
aiavatar_app = AIAvatar(
    openai_api_key=API_KEY,
    context_manager=PostgreSQLContextManager(get_pool=pool_provider.get_pool),
    session_state_manager=PostgreSQLSessionStateManager(get_pool=pool_provider.get_pool),
    performance_recorder=PostgreSQLPerformanceRecorder(get_pool=pool_provider.get_pool),
)
```

### 7.5 pgvector for Speaker Registry

Speaker identification requires vector similarity search. Use `pgvector` extension for PostgreSQL.

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

```python
from aiavatar.sts.stt.speaker_registry.pgvector import PGVectorStore

speaker_store = PGVectorStore(get_pool=pool_provider.get_pool)
speaker_registry = SpeakerRegistry(store=speaker_store)
```

---

## 8. Character Management

AI character lifecycle management. Features for making **characters behave as if "alive"**.

### 7.1 Character Model

| Field | Description |
|-------|-------------|
| **id** | Unique identifier (UUID) |
| **name** | Character name |
| **prompt** | Prompt defining personality and behavior. The character's "soul" |
| **metadata** | Custom attributes (appearance, background, hobbies, etc.) |

#### CharacterService Configuration

| Feature | Description |
|---------|-------------|
| **OpenAI support** | Standard OpenAI API for schedule/diary generation |
| **Azure OpenAI support** | Auto-detects Azure when model name contains "azure". Extracts api-version from base_url |

### 7.2 Schedule Management

Defines **"what the character is currently doing"**. Provides context like "I'm in class now so I'll reply briefly".

#### Weekly Schedule
| Feature | Description |
|---------|-------------|
| **AI generation** | Auto-generate weekly schedule from character settings |
| **Markdown table** | Hourly activity table. School, club activities, hobbies, etc. |
| **CRUD operations** | Create, read, update, delete |

#### Daily Schedule
| Feature | Description |
|---------|-------------|
| **AI generation** | Generate from weekly schedule + previous day's diary. "I was tired yesterday so I'll rest today", etc. |
| **Context storage** | Save generation input and reasoning process. For debugging |
| **Continuity** | Considers continuity with previous day. Natural daily flow |

### 7.3 Diary System

Records **"what happened today"** for the character. Can remember past events and bring them up in conversation.

| Feature | Description |
|---------|-------------|
| **AI generation** | Generate diary from daily schedule + news |
| **Topic extraction** | Auto-extract important topics. Topic selection |
| **News integration** | Reflect day's news in diary. "I saw XX on today's news..." |
| **Emotional expression** | Character's perspective and feelings. Joy, anger, sorrow, pleasure |
| **Length control** | Specify character count with diary_length |

### 7.4 Memory System

**Remembers past conversations**, can recall "that XX we talked about before".

| Feature | Description |
|---------|-------------|
| **Semantic search** | Search from past conversations. "The food you said you liked before", etc. |
| **Date filter** | Specify period with since/until. "Last week's conversation", etc. |
| **Message storage** | Async storage of conversation pairs. No performance impact |
| **Diary integration** | Register diaries into memory system. Diary contents also searchable |

### 7.5 MemorySearchTool

A tool for LLM to **recall the past during conversation**.

```python
# Usage: LLM searches past conversations as needed
memory_tool = MemorySearchTool(
    memory_client=memory_client,
    character_id="char_123"
)
llm.add_tool(memory_tool)
# User: "Do you remember the movie title we talked about before?"
# LLM: [Calls memory_tool to search] → "It was Your Name, right!"
```

### 7.6 User Management

Manages **user profiles** for personalized conversations. Enables the character to remember user information across sessions.

#### User Model
| Field | Description |
|-------|-------------|
| **id** | Unique identifier (UUID) |
| **name** | User's name |
| **metadata** | Custom attributes (preferences, notes, etc.) |
| **created_at** | Account creation timestamp |
| **updated_at** | Last update timestamp |

#### User Operations
| Operation | Description |
|-----------|-------------|
| **create** | Create new user with default name and metadata |
| **get** | Retrieve user information by ID |
| **update** | Update user name or metadata |
| **delete** | Delete user account |

#### UpdateUsernameTool
A tool for LLM to **update user's name during conversation**.

```python
# Usage: Character learns and remembers user's name
# User: "My name is Alice"
# LLM: [Calls UpdateUsernameTool with "Alice"]
# Future conversations: "Hello Alice!"
```

### 7.7 bind_character Helper

A single-function setup for **complete character integration**. Reduces boilerplate from 50+ lines to one function call.

```python
from aiavatar.character.binding import bind_character

bind_character(
    adapter=aiavatar_app,
    character_service=character_service,
    character_id="char_123",
    default_user_name="Guest"
)
```

#### Automatic Setup
| Feature | Description |
|---------|-------------|
| **Session initialization** | Auto-creates user if not exists, assigns user_id to request |
| **Dynamic system prompt** | Injects character prompt with user's name as parameter |
| **Response metadata** | Sends username and character name to client on connection |
| **Tool registration** | Auto-registers UpdateUsernameTool, GetDiaryTool, MemorySearchTool |

### 7.8 System Prompt Management

Generates the **final prompt** integrating character settings and schedule.

| Feature | Description |
|---------|-------------|
| **Template** | Integrates character settings + daily schedule |
| **Cache** | Cache system prompts by date. Don't generate every time |
| **Parameter substitution** | Dynamic substitution like {username}. Insert user names |
| **Face tags** | Support for [face:emotion] format. Add emotions to responses |
| **Language switching** | Support for [language:code] format. Multi-language support |

---

## 9. Device Control

Audio/video device management. **Control microphone, speaker, and camera in local environment**.

### 9.1 Audio Devices

#### AudioDevice
Microphone/speaker selection and management.

| Feature | Description |
|---------|-------------|
| **Device enumeration** | Get list of input/output devices. Check available devices |
| **Name search** | Search devices by partial match. Mics containing "USB", etc. |
| **Interactive selection** | Device selection via CLI. Select by number input |

#### AudioPlayer
TTS audio playback.

| Feature | Description |
|---------|-------------|
| **Queue playback** | Queue processing in background thread. Play audio sequentially |
| **Dynamic stream** | Auto-reinitialize on parameter change. Handles format changes |
| **WAV format** | Auto-parse WAV headers. Auto-detect sample rate, etc. |
| **Stop control** | Queue clear and stop events. Immediate stop on barge-in |

#### AudioRecorder
Microphone input capture.

| Feature | Description |
|---------|-------------|
| **Async streaming** | Generate audio data via AsyncGenerator. Real-time processing |
| **Configurable parameters** | sample_rate, channels, chunk_size. Match STT requirements |

#### NoiseLevelDetector
Ambient noise measurement. Used to **auto-adjust VAD threshold**.

| Feature | Description |
|---------|-------------|
| **Noise measurement** | Measure ambient noise level for ~3 seconds |
| **dB calculation** | Amplitude → decibel conversion |
| **Median filter** | Stable noise estimation. Excludes momentary noise |

### 9.2 Video Devices

#### VideoDevice
Image capture from camera. Integrates with **Vision feature**.

| Feature | Description |
|---------|-------------|
| **Frame capture** | Async image capture. Called via [vision:camera] tag |
| **JPEG output** | Encode as JPEG format. For sending to LLM |
| **Resolution/FPS settings** | Width, height, frame rate configuration |
| **Device enumeration** | Detect available cameras |

---

## 10. Avatar Control

VRChat avatar facial expression and animation control. Enables **more natural conversation experience by moving avatar in sync with AI responses**.

### 10.1 Facial Expression Control (FaceController)

A feature to change avatar facial expressions based on LLM response content. **Expresses emotions visually, not just through text**, enhancing conversation immersion.

#### VRChatFaceController
| Feature | Description | Purpose |
|---------|-------------|---------|
| **OSC communication** | Send expression parameters to VRChat via UDP | Integration via VRChat's standard protocol |
| **Expression mapping** | Map expression name → integer value | Convert from `[face:joy]` tag to actual parameter |
| **Auto reset** | Automatically return to neutral after duration | Prevent forgetting to reset expression |
| **Default expressions** | neutral, joy, angry, sorrow, fun, surprise | Cover basic emotional expressions |

```python
# LLM response example: "I'm happy! [face:joy]"
# → Send "/avatar/parameters/FaceOSC" = 1 to VRChat
# → Avatar smiles
```

### 10.2 Animation Control (AnimationController)

A feature to execute **gestures and actions** on the avatar. Waving when greeting, hands on hips when angry, etc. **Complements expressions that words alone can't convey**.

#### VRChatAnimationController
| Feature | Description | Purpose |
|---------|-------------|---------|
| **OSC communication** | Send animation parameters to VRChat via UDP | Move avatar in real-time |
| **Animation mapping** | Map animation name → integer value | Convert from `[animation:wave]` tag to actual parameter |
| **Auto reset** | Return to idling after duration | Automatically return to idle state after animation |
| **Default animations** | idling, angry_hands_on_waist, waving_arm, etc. | Built-in commonly used animations |

```python
# LLM response example: "See you! [animation:waving_arm]"
# → Send "/avatar/parameters/VRCEmote" = 3 to VRChat
# → Avatar waves
```

---

## 11. Administration

Runtime configuration changes and control. **Change settings dynamically without restarting the application**.

Why administration features are needed:
- **Zero-downtime** parameter adjustment in production (prompt changes, model switching, etc.)
- **Immediate response to issues** during operation (stop listener, check logs)
- Enable **A/B testing** and gradual model migration

### 11.1 Admin Panel

A web-based dashboard for **monitoring, controlling, and evaluating** your AI avatar from a browser.

| Feature | Description |
|---------|-------------|
| **Metrics** | Real-time STS pipeline performance visualization with charts |
| **Logs** | Conversation history with audio playback support |
| **Control** | Send messages and control avatar directly |
| **Config** | Dynamic adjustment of pipeline, VAD, STT, LLM, TTS, and adapter settings |
| **Evaluation** | Run dialogue evaluation scenarios |
| **Characters** | Manage character info, schedules, diaries, and users (requires CharacterService) |

```python
# Setup admin panel
from aiavatar.admin import setup_admin_panel

panel = setup_admin_panel(app, adapter=aiavatar_app)

# Dynamic adapter registration after setup
panel.add_adapter(another_adapter, name="secondary")
```

#### Authentication
| Feature | Description |
|---------|-------------|
| **Basic authentication** | Optional username/password protection |
| **API key** | Bearer token authentication for REST API |
| **Custom HTML** | Serve custom admin panel HTML |

### 11.2 Configuration API (ConfigAPI)

API to **get/update component settings via HTTP endpoints**.

#### STT Configuration
**Dynamic speech recognition adjustment**. Change language settings or model if recognition accuracy issues arise.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /stt/config | GET | Get STT configuration |
| /stt/config | POST | Update STT configuration |

Configurable items: language, alternative_languages, timeout, debug, model, base_url, sample_rate

#### LLM Configuration
**Dynamically change AI behavior**. Reflect prompt improvements or model switches in real-time.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /llm/config | GET | Get LLM configuration |
| /llm/config | POST | Update LLM configuration |

Configurable items: system_prompt, model, temperature, split_chars, voice_text_tag, use_dynamic_tools, etc.

#### TTS Configuration
**Dynamic voice quality/speed adjustment**. Can change in real-time based on user preferences.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /tts/config | GET | Get TTS configuration |
| /tts/config | POST | Update TTS configuration |

Configurable items: style_mapper, timeout, debug, voice, speaker, speed, pitch, volume, etc.

#### Component Switching
**Dynamically switch STT/LLM/TTS providers**. Example: Gradual migration from OpenAI → Claude.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /sts/component | GET | Get available component list |
| /sts/component | POST | Switch active component |

#### System Log
**Troubleshooting during operation**. Check logs to identify causes when errors occur.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /system/log | GET | Get latest logs (max 1000 lines) |

### 11.3 Control API (ControlAPI)

API for **external control of avatar behavior**.

#### Listener Control
**Control voice input start/stop**. Temporarily disable voice input during maintenance or specific situations.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /listener/start | POST | Start voice listener |
| /listener/stop | POST | Stop voice listener |
| /listener/status | GET | Get listener status |

#### Avatar Control
**Operate avatar from external systems**. Execute specific expressions or animations on event triggers.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /avatar/status | GET | Get current expression/animation |
| /avatar/face | POST | Set expression |
| /avatar/face | GET | Get current expression |
| /avatar/animation | POST | Execute animation |
| /avatar/animation | GET | Get current animation |
| /avatar/perform | POST | Integrated speech + expression + animation execution |

#### Conversation Control
**Conversation triggers from non-voice sources**. Handle input from text chat or system integration.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /conversation | POST | Send text message |
| /conversation/histories | GET | Get conversation history |

### 11.4 Dummy Components for Testing

Configurable dummy implementations for **performance and load testing without external API dependencies**.

| Component | Description |
|-----------|-------------|
| **SpeechRecognizerDummy** | Returns configurable `recognized_text` after `wait_sec` delay |
| **LLMServiceDummy** | Returns configurable `response_text` after `wait_sec` delay |
| **SpeechSynthesizerDummy** | Returns configurable `synthesized_bytes` after `wait_sec` delay |

```python
# Load testing setup example
from aiavatar.sts.stt.dummy import SpeechRecognizerDummy
from aiavatar.sts.llm.dummy import LLMServiceDummy
from aiavatar.sts.tts.dummy import SpeechSynthesizerDummy

stt = SpeechRecognizerDummy(recognized_text="Hello", wait_sec=0.1)
llm = LLMServiceDummy(response_text="Hi there!", wait_sec=0.5)
tts = SpeechSynthesizerDummy(synthesized_bytes=b"audio_data", wait_sec=0.2)
```

---

## 12. Evaluation

Dialogue system quality evaluation. A mechanism for **continuously measuring and improving AI dialogue quality**.

Why evaluation features are needed:
- **Quantitatively assess impact** of prompt changes (not accidentally making things worse while trying to improve)
- Function as **regression tests** (verify existing scenarios aren't broken)
- **Quality comparison when switching LLM models** (GPT-4 vs Claude comparison evaluation)

### 12.1 Scenario Evaluation

Structure to **simulate actual dialogue flows** for evaluation. Judge not just single responses but whether **entire multi-turn conversations** are appropriate.

| Concept | Description | Purpose |
|---------|-------------|---------|
| **Turn** | One dialogue exchange (input → response) | Evaluate individual response quality |
| **Scenario** | Test case with multiple Turns and goals | Evaluate overall conversation flow |
| **Dataset** | Collection of multiple scenarios | Comprehensive quality evaluation |

### 12.2 Turn Configuration

Define **what to evaluate** at each turn.

| Field | Description | Purpose |
|-------|-------------|---------|
| **input_text** | User input | Utterance to simulate |
| **expected_output_text** | Expected response (reference) | Reference information for LLM evaluation |
| **evaluation_criteria** | Evaluation criteria | Pass/Fail judgment criteria |
| **evaluation_function_name** | Custom evaluation function name | Evaluation with special logic |

### 12.3 Evaluation Methods

#### LLM-Based Evaluation
**Evaluation LLM** judges response appropriateness. Automates human subjective evaluation.

- Evaluation LLM judges response appropriateness
- Success marker: `[result:true]`
- Evaluate at both Turn level and Scenario level

```python
# Evaluation criteria example
evaluation_criteria = "Does it appropriately answer the user's question?"
# → Evaluation LLM judges "appropriate/inappropriate" and outputs reason
```

#### Custom Evaluation Functions
Handle **cases LLM evaluation can't judge** (specific keyword presence, tool call verification, etc.).

```python
def custom_eval(output_text, tool_call, criteria, llm_result, llm_reason):
    # Example: Check if specific tool was called
    if tool_call and tool_call.name == "search_database":
        return (True, "Search tool was correctly called")
    return (False, "Search tool was not called")
```

### 12.4 Evaluation API

API to **execute evaluation in background** and retrieve results. Handles large-scale scenario evaluation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /evaluate | POST | Start background evaluation |
| /evaluate/{id} | GET | Get evaluation results |

### 12.5 Evaluation Metrics

**Quantitative quality indicators**. Track changes over time to understand improvement/degradation.

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Turn result** | Pass/Fail for each Turn | Individual response quality |
| **Goal achievement** | Goal achievement judgment for entire scenario | Overall conversation success/failure |
| **Turn success rate** | Pass count / Total Turn count | Overall quality score |
| **Reason** | Detailed reason for each judgment | Identify improvement points |

---

## Appendix: Quick Reference

Reference tables for provider and feature selection.

### Provider Support Matrix

List of available providers. **Select based on use case and budget**.

| Category | Providers | Selection Criteria |
|----------|-----------|-------------------|
| **STT** | Google, Azure, OpenAI Whisper, AmiVoice | Accuracy, language support, cost |
| **LLM** | ChatGPT, Claude, Gemini, LiteLLM, Dify | Capability, latency, cost |
| **TTS** | VOICEVOX, Azure, Google, OpenAI, SpeechGateway | Voice quality, voice variety, cost |
| **VAD** | Standard (volume), Silero (ML), Azure (cloud) | Accuracy, computational resources |

### Database Support

**SQLite for development/small-scale**, **PostgreSQL for production/large-scale**.

| Feature | SQLite | PostgreSQL | Notes |
|---------|--------|------------|-------|
| Context management | ○ | ○ | Conversation history persistence |
| Session state | ○ | ○ | Dialogue state management |
| Performance recording | ○ | ○ | Response time recording, etc. |
| Speaker registry | File | pgvector | Vector search for speaker identification |
| LINE Bot sessions | ○ | ○ | Per-user sessions |
| Character management | ○ | ○ | Schedules, diaries, memory |
| User management | ○ | ○ | User profiles and metadata |

#### PostgreSQL Pool Provider

Centralized connection pool management for production environments.

| Feature | Description |
|---------|-------------|
| **Shared pool** | Multiple components share a single connection pool |
| **Lazy initialization** | Pool created on first access |
| **Configurable size** | min_size (default: 5), max_size (default: 20) |
| **Pool statistics** | `get_stats()` returns pool utilization metrics |
| **Thread-safe** | Safe for concurrent access |

```python
from aiavatar.database.postgres import PostgreSQLPoolProvider

pool_provider = PostgreSQLPoolProvider(
    connection_str="postgresql://user:pass@host:5432/db",
    min_size=5,
    max_size=30
)

# Pass to components
context_manager = PostgreSQLContextManager(get_pool=pool_provider.get_pool)
session_manager = PostgreSQLSessionStateManager(get_pool=pool_provider.get_pool)
character_service = CharacterService(db_pool_provider=pool_provider)
```

### Control Tag List

Tags embedded in LLM responses to **control avatar and voice**.

| Tag | Purpose | Example | Effect |
|-----|---------|---------|--------|
| [face:name] | Expression control | [face:joy] | Avatar smiles |
| [animation:name] | Animation | [animation:wave] | Wave hand animation |
| [language:code] | Language switch | [language:en-US] | Switch TTS language to English |
| [vision:source] | Image capture | [vision:camera] | Send camera image to LLM |
