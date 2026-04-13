# AIAvatarKit Feature Summary Table

A concise feature overview of AIAvatarKit (v0.8.14, April 2026).

AIAvatarKit is a Speech-to-Speech framework for building real-time voice-interactive AI avatars.

---

## 1. VAD (Voice Activity Detection)

| Feature | Description |
|---------|-------------|
| Supported implementations | Volume-based (Standard), ML-based (Silero), Streaming ML (SileroStream), Cloud-based (Azure Stream), Cloud-based (AWS Amazon Transcribe Stream) |
| Streaming VAD with any STT | Convert non-streaming STT to streaming with segment recognition |
| Pre-roll buffer | Retain audio before speech starts to prevent cutting off beginnings |
| Multi-session support | Concurrent processing of multiple users/sessions |
| Dynamic mute control | Callback-based mute to ignore user voice while avatar speaks |
| Hybrid detection | Combine volume threshold and ML model |
| Event callbacks | Multiple callbacks for speech detected, recording started, voiced, errors |
| Text validation | Filter out invalid recognition results via `validate_recognized_text` (stream detectors) |
| Custom recording trigger | Customize `on_recording_started` firing conditions via `should_trigger_recording_started` |
| Silence accumulation | Accumulate multiple recognition results within silence threshold (Amazon Transcribe) |

---

## 2. STT (Speech-to-Text)

| Feature | Description |
|---------|-------------|
| Supported providers | Google Cloud, Azure Speech, OpenAI Whisper, AmiVoice |
| Multi-language support | Primary and alternative language configuration |
| Preprocessing hook | Audio data preprocessing before recognition (with metadata) |
| Postprocessing hook | Recognition text postprocessing (with metadata from preprocessing) |
| Auto-retry on failure | Automatic retry for temporary API failures |
| Custom model support | Domain-specific trained models (Azure) |

### Speaker Identification

| Feature | Description |
|---------|-------------|
| Main speaker gate | Lock to main speaker and filter out others |
| Speaker registry | Register and match known speakers by name |
| Vector similarity matching | Embedding-based speaker identification |
| Persistence | File or PostgreSQL (pgvector) storage |

---

## 3. LLM (Large Language Model)

| Feature | Description |
|---------|-------------|
| Supported providers | OpenAI ChatGPT (o1/o3), OpenAI Responses API (REST/WebSocket), Anthropic Claude, Google Gemini, LiteLLM, Dify |
| OpenAI-compatible APIs | Use ChatGPTService with Grok, Gemini, Claude via base_url (non-reasoning) |
| Streaming response | Stream text chunks for parallel TTS processing (HTTP and WebSocket) |
| Context persistence | SQLite/PostgreSQL conversation history, server-side via previous_response_id (Responses API) |
| Shared context | Integrate context from multiple sessions |
| Few-shot learning | Initial messages to guide response style |
| Inline LLM parameters | Override model, temperature, etc. per-request via `inline_llm_params` |
| Error handling | Custom error response via `on_error` decorator (e.g., content filter handling) |
| Custom chat logging | Customizable conversation logging format via `print_chat` decorator |

### Low-Latency Response

| Feature | Description |
|---------|-------------|
| Sentence-based splitting | Split at punctuation for immediate TTS |
| Optional mid-sentence split | Split at commas for earlier vocalization |
| Control tag splitting | Split before [tag:value] for timing control |

### Chain-of-Thought (CoT)

| Feature | Description |
|---------|-------------|
| Think/answer separation | Vocalize only answer, not thinking process |
| Custom voice text tag | Extract specific XML tags for TTS (single tag or list of tags) |

### Tool Calling

| Feature | Description |
|---------|-------------|
| Function calling | Register tools with decorator or `add_tool`, auto-convert formats across providers |
| Dynamic tool selection | LLM selects relevant tools from large sets via pre-flight filtering |
| Custom tool repository | `get_dynamic_tools` hook for vector-search-based tool retrieval at scale |
| Streaming tool results | Stream intermediate results and voice feedback for long operations |
| Background tool execution | Run long-running tools in background with `on_completed`/`on_submitted` callbacks |
| Background timeout | Hybrid mode: try sync first, fall back to background if timeout exceeded |
| Response formatter | Bypass 2nd LLM call, speak tool result directly via template |
| Structured content | Send structured data directly to client via `ToolCallResult.structured_content` |
| MCP integration | Model Context Protocol server support (Streamable HTTP and Stdio) |

### Built-in Tools

| Feature | Description |
|---------|-------------|
| Web search | Grok, Gemini, OpenAI web search |
| Web scraping | Playwright-based JavaScript rendering, optional LLM summary |
| Image generation | Gemini-based image generation (NanoBanana) and selfie generation |
| OpenClaw | Autonomous AI agent delegation with background execution support |

### Guardrails

| Feature | Description |
|---------|-------------|
| Request filtering | Block/replace inappropriate input (sync evaluation) |
| Response filtering | Corrective utterance for issues (async evaluation) |
| Parallel execution | Multiple guardrails run concurrently |
| Non-blocking response | Response starts immediately, issues corrected later |

### Callbacks/Hooks

| Feature | Description |
|---------|-------------|
| Request/context filters | Preprocess request text and filter before saving history |
| Dynamic system prompt | Generate prompt based on user/situation via `get_system_prompt` |
| Tool call hooks | Logging or approval before execution, error handling |

### Observability

| Feature | Description |
|---------|-------------|
| Langfuse integration | Auto-collect traces and logs |
| Custom OpenAI module | Replace client for monitoring |

---

## 4. TTS (Text-to-Speech)

| Feature | Description |
|---------|-------------|
| Supported providers | VOICEVOX/AivisSpeech, Azure Speech, Google Cloud, OpenAI, SpeechGateway (Style-Bert-VITS2, Aivis Cloud API) |
| Instant TTS Synthesizer | Quick setup for custom TTS HTTP APIs (ElevenLabs, Kotodama, Coefont, Amazon Polly, etc.) via `create_instant_synthesizer` |
| Custom TTS integration | Implement `SpeechSynthesizer` interface for full control |
| Dynamic style switching | Switch voice style based on LLM output keywords |
| TTS caching | Cache synthesized audio to reduce latency and cost (all synthesizers) |
| Autonomous multi-language | Dynamic per-sentence language switching via [language:code] tags |
| Per-session customization | Dynamically adjust speed per user session via style_info |
| Audio format conversion | ffmpeg-based format/sample rate conversion via AudioConverter |

### Preprocessing

| Feature | Description |
|---------|-------------|
| Alphabet to Katakana | LLM-based foreign word conversion with kana_map caching |
| Pattern match | String/regex replacement with built-in number-dash and phone patterns |
| Custom preprocessors | Implement `TTSPreprocessor` interface, chain multiple preprocessors |

---

## 5. Voice Pipeline

| Feature | Description |
|---------|-------------|
| End-to-end pipeline | Integrated VAD → STT → LLM → TTS |
| Barge-in (interruption) | Interrupt avatar speech when user starts speaking |
| Request merging | Combine consecutive utterances within threshold, customizable prefix |
| Timestamp injection | Auto-insert current datetime into LLM context at configurable intervals |
| Wake word activation | Activate on specific keywords with timeout |
| Queue management | Three modes: Direct (default), Queued Interrupt, Queued Wait |
| Performance metrics | Detailed timing with tool call logging and error tracking |
| Voice recording | Save request/response audio |
| Quick Response | QuickResponder / QuickResponderPro for immediate acknowledgment before main LLM response |
| Request validation | Filter unwanted requests via `validate_request` hook before LLM processing |
| Pipeline hooks | on_before_llm, on_before_tts, process_llm_chunk, on_finish, on_response |
| Custom behavior | Hook into response types (start/chunk/final) for avatar control |
| System prompt parameters | Embed dynamic parameters into system prompt via placeholders |
| Vision | Capture and send images to AI via [vision:source] tags |

---

## 6. Channels/Adapters

| Feature | Description |
|---------|-------------|
| Supported adapters | HTTP/REST (SSE), WebSocket, LINE Bot, Twilio, Local (in-process) |
| SSE streaming | Low-latency chunk-by-chunk response (HTTP) |
| Dify-compatible API | `/chat-messages` endpoint compatible with Dify frontend apps |
| STT/TTS endpoints | Standalone REST API endpoints for speech-to-text and text-to-speech |
| PCM audio streaming | Split audio into chunks for streaming playback (WebSocket) |
| Client-side features | VAD, auto noise threshold, echo cancellation |
| Lifecycle hooks | Session start, connect/disconnect, request, response hooks |
| Channel Context Bridge | Map channel user IDs to app-level users, persist context across channels (SQLite/PostgreSQL) |

---

## 7. Database

| Feature | Description |
|---------|-------------|
| SQLite support | Default for development (context, session, performance, LINE Bot, channel context) |
| PostgreSQL support | Production-ready with shared connection pool |
| pgvector integration | Vector similarity search for speaker identification |
| Pool provider | Centralized pool management to prevent connection exhaustion |

---

## 8. Character Management

| Feature | Description |
|---------|-------------|
| Character model | Name, prompt, metadata storage |
| AI-generated content | Weekly/daily schedules and diaries with continuity |
| Batch generation | Generate schedules and diaries for a date range (`create_activity_range_with_generation`) |
| Long-term memory | Semantic search of past conversations and diaries via MemoryClient |
| User management | User profiles with name and metadata |
| LLM tools | Memory search, username update, diary retrieval |
| Easy integration | `bind_character` helper for one-line setup |
| CharacterLoader | Lightweight file-based alternative (markdown/JSON, no database required) |
| Hot reload | File-based character definitions with mtime-based cache invalidation (CharacterLoader) |
| Split initial messages | Inject episodes, attributes, conversation examples as pseudo turns (CharacterLoader) |

---

## 9. Device Control

| Feature | Description |
|---------|-------------|
| Audio devices | Enumeration, playback queue, recording stream |
| Noise level detection | Measure ambient noise for VAD threshold |
| Camera capture | Async image capture for Vision feature |

---

## 10. Avatar Control

| Feature | Description |
|---------|-------------|
| VRChat OSC integration | Send expression/animation parameters via UDP |
| Tag-based control | Map [face:name] and [animation:name] tags (bracket and XML-style) |
| Auto reset | Return to neutral/idle after duration |

---

## 11. Administration

| Feature | Description |
|---------|-------------|
| Admin Panel | Web dashboard with metrics, logs, config, evaluation, and character management |
| Custom HTML | Supply your own HTML to customize the admin page |
| Runtime config | Change pipeline/VAD/STT/LLM/TTS/adapter settings without restart |
| Control APIs | Listener, avatar, conversation control endpoints |
| Authentication | Basic auth and API key support |
| Dummy components | Configurable STT/LLM/TTS for load testing |

---

## 12. Evaluation

| Feature | Description |
|---------|-------------|
| Scenario-based evaluation | Multi-turn dialogue testing with goal achievement |
| LLM-based evaluation | Auto-judge response appropriateness |
| Custom evaluation functions | Handle cases LLM can't judge via registered functions |
| File-based scenarios | Load scenarios from JSON files |
| Background evaluation API | Run large-scale evaluations async |
| Config API integration | Run evaluations on the fly via admin panel |
