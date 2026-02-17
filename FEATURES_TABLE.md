# AIAvatarKit Feature Summary Table

A concise feature overview of AIAvatarKit (v0.8.8, February 2026).

AIAvatarKit is a Speech-to-Speech framework for building real-time voice-interactive AI avatars.

---

## 1. VAD (Voice Activity Detection)

| Feature | Description |
|---------|-------------|
| Supported implementations | Volume-based, ML-based (Silero), Streaming ML (SileroStream), Cloud-based (Azure) |
| Streaming VAD with any STT | Convert non-streaming STT to streaming with segment recognition |
| Pre-roll buffer | Retain audio before speech starts to prevent cutting off beginnings |
| Multi-session support | Concurrent processing of multiple users/sessions |
| Dynamic mute control | Callback-based mute to ignore user voice while avatar speaks |
| Hybrid detection | Combine volume threshold and ML model |
| Event callbacks | Multiple callbacks for speech detected, recording started, voiced, errors |

---

## 2. STT (Speech-to-Text)

| Feature | Description |
|---------|-------------|
| Supported providers | Google Cloud, Azure Speech, OpenAI Whisper, AmiVoice |
| Multi-language support | Primary and alternative language configuration |
| Preprocessing hook | Audio data preprocessing before recognition |
| Postprocessing hook | Recognition text postprocessing |
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
| Supported providers | OpenAI ChatGPT (o1/o3), Anthropic Claude, Google Gemini, LiteLLM, Dify |
| Streaming response | Stream text chunks for parallel TTS processing |
| Context persistence | SQLite/PostgreSQL conversation history |
| Shared context | Integrate context from multiple sessions |
| Few-shot learning | Initial messages to guide response style |

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
| Custom voice text tag | Extract specific XML tags for TTS |

### Tool Calling

| Feature | Description |
|---------|-------------|
| Function calling | Register tools with decorator, auto-convert formats |
| Dynamic tool selection | LLM selects relevant tools from large sets |
| Streaming tool results | Stream intermediate results for long operations |
| MCP integration | Model Context Protocol server support |

### Built-in Tools

| Feature | Description |
|---------|-------------|
| Web search | Grok, Gemini, OpenAI web search |
| Web scraping | Playwright-based JavaScript rendering |
| Image generation | Gemini-based image/selfie generation |

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
| Dynamic system prompt | Generate prompt based on user/situation |
| Tool call hooks | Logging or approval before execution, error handling |
| Custom logging | Customizable conversation logging format (prettify) |

### Observability

| Feature | Description |
|---------|-------------|
| Langfuse integration | Auto-collect traces and logs |
| Custom OpenAI module | Replace client for monitoring |

---

## 4. TTS (Text-to-Speech)

| Feature | Description |
|---------|-------------|
| Supported providers | VOICEVOX/AivisSpeech, Azure Speech, Google Cloud, OpenAI |
| Custom TTS integration | Factory function for proprietary services |
| Dynamic style switching | Switch voice style based on LLM output keywords |
| Pronunciation preprocessing | String/regex replacement, LLM-based foreign word conversion with caching |
| Audio format conversion | ffmpeg-based format/sample rate conversion |
| TTS caching | Cache synthesized audio to dramatically reduce latency and cost |
| Autonomous multi-language | Dynamic per-sentence language switching via [language:code] tags |
| Per-session customization | Dynamically adjust speed per user session |

---

## 5. Voice Pipeline

| Feature | Description |
|---------|-------------|
| End-to-end pipeline | Integrated VAD → STT → LLM → TTS |
| Barge-in (interruption) | Interrupt avatar speech when user starts speaking |
| Request merging | Combine consecutive utterances within threshold |
| Timestamp injection | Auto-insert current datetime into LLM context |
| Wake word activation | Activate on specific keywords |
| Queue management | Sequential processing with wait/cancel modes |
| Performance metrics | Detailed timing with tool call logging and error tracking |
| Voice recording | Save request/response audio |
| Pipeline hooks | Request validation, before LLM/TTS, chunk processing, finish |

---

## 6. Channels/Adapters

| Feature | Description |
|---------|-------------|
| Supported adapters | HTTP/REST, WebSocket, WebSocket STT-only, LINE Bot, Local (in-process) |
| SSE streaming | Low-latency chunk-by-chunk response (HTTP) |
| PCM audio streaming | Split audio into chunks for streaming playback (WebSocket) |
| Client-side features | VAD, auto noise threshold, echo cancellation |
| Lifecycle hooks | Session start, request, response hooks |

---

## 7. Database

| Feature | Description |
|---------|-------------|
| SQLite support | Default for development (context, session, performance, LINE Bot) |
| PostgreSQL support | Production-ready with shared connection pool |
| pgvector integration | Vector similarity search for speaker identification |
| Pool provider | Centralized pool management to prevent connection exhaustion |

---

## 8. Character Management

| Feature | Description |
|---------|-------------|
| Character model | Name, prompt, metadata storage |
| AI-generated content | Weekly/daily schedules and diaries with continuity |
| Long-term memory | Semantic search of past conversations and diaries |
| User management | User profiles with name and metadata |
| LLM tools | Memory search, username update, diary retrieval |
| Easy integration | bind_character helper for one-line setup |

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
| Tag-based control | Map [face:name] and [animation:name] tags |
| Auto reset | Return to neutral/idle after duration |

---

## 11. Administration

| Feature | Description |
|---------|-------------|
| Admin Panel | Web dashboard with metrics, logs, config, and evaluation |
| Runtime config | Change STT/LLM/TTS settings without restart |
| Control APIs | Listener, avatar, conversation control endpoints |
| Authentication | Basic auth and API key support |
| Dummy components | Configurable STT/LLM/TTS for load testing |

---

## 12. Evaluation

| Feature | Description |
|---------|-------------|
| Scenario-based evaluation | Multi-turn dialogue testing with goal achievement |
| LLM-based evaluation | Auto-judge response appropriateness |
| Custom evaluation functions | Handle cases LLM can't judge |
| Background evaluation API | Run large-scale evaluations async |
