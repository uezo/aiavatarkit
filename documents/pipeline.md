# Pipeline

`STSPipeline` owns a turn from the moment audio arrives until the last chunk of speech goes
back out. Adapters hand it an `STSRequest` and consume the stream of `STSResponse` objects
it yields; everything in this page configures what happens in between.

## The four identifiers

Almost every hook, callback, and record is scoped by one of four ids. Getting them right is
what keeps concurrent users from bleeding into each other.

| Id | Scope | Lives as long as |
| --- | --- | --- |
| `session_id` | One connection — a WebSocket, a phone call, a browser tab | the transport stays open |
| `context_id` | One conversation — the history the LLM sees | the context timeout, or until you rotate it |
| `user_id` | One person, across sessions and channels | your application decides |
| `transaction_id` | One turn — request in, response out | a single turn |

A session and a context are not the same thing. Reconnecting gives a new `session_id` but
can keep the same `context_id`, which is how a conversation survives a dropped connection.
The same person reaching you on a second channel keeps the same `user_id`, which is what
[the channel context bridge](adapters.md) uses to resume their conversation.

## A turn, end to end

1. The adapter receives audio (or text) and passes it to the pipeline.
2. The speech detector decides a turn has ended — possibly after
   [turn-end gates](vad-turn-end.md) confirm it.
3. Recognised text becomes an `STSRequest`. Request merging, timestamp insertion, wake word
   matching, and request validation all act here, before the LLM sees anything.
4. A quick response may start speaking immediately while the real answer is generated.
5. The LLM streams back text. It is split into speakable units and handed to TTS as each
   unit completes, so audio starts before generation finishes.
6. Control tags in the response become avatar actions and are stripped from the spoken text.
7. The pipeline records timings and, optionally, the audio itself.

## Configuring the pipeline

`STSPipeline` accepts either ready-made components or the parameters to build default ones:

```python
from aiavatar.sts import STSPipeline
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

sts = STSPipeline(
    stt=OpenAISpeechRecognizer(openai_api_key=OPENAI_API_KEY),
    llm_openai_api_key=OPENAI_API_KEY,
    llm_reasoning_effort="none",
    llm_system_prompt="You are a helpful assistant.",
)
```

Adapters accept the same arguments, and build the pipeline for you when you do not pass one.
Pass `sts=` to attach an adapter to a pipeline that already exists — that is how
[multiple channels](adapters.md) share one conversation.

`llm_temperature` and `llm_reasoning_effort` default to `None`, which leaves the
corresponding API parameter unspecified. Set `llm_reasoning_effort="none"` explicitly
when you want the default OpenAI model to skip reasoning for lower latency.

## Wakeword

Set `wakewords` when instantiating the adapter. Conversation will start when the pipeline recognizes one of the words in this list. You can also set `wakeword_timeout`, after which it will return to listening for the wakeword again.

```python
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=OPENAI_API_KEY,
    wakewords=["Hello", "こんにちは"],
    wakeword_timeout=60,
)
```

## Timestamp Insertion

You can insert timestamps into requests at regular intervals. This keeps responses anchored to real-world time.

```python
aiavatar_app = AIAvatarWebSocketServer(
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

## Request merging

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

## Invoke Queue

A turn can arrive while the previous one is still going. Three modes decide what happens
then.

### Invoke Modes

| Mode | Settings | What the new request does |
|------|----------|----------|
| **Direct** (default) | `use_invoke_queue=False` | Runs immediately, concurrently with the previous turn. The previous generation loop notices a newer transaction and stops, and playback is stopped too. |
| **Queued (Interrupt)** | `use_invoke_queue=True`, `wait_in_queue=False` | Joins the queue after **discarding every request still waiting in it**. Those get a `cancelled` response. Playback of the previous answer is stopped when this request starts. |
| **Queued (Wait)** | `use_invoke_queue=True`, `wait_in_queue=True` | Joins the back of the queue and waits. Nothing is discarded and nothing is stopped. Use it for follow-ups that must not cut the current answer off — a vision request carrying an image, for instance. |

The distinction between the two queued modes is about the **queue**, not about generation.
A queue is served by one worker, one request at a time, so nothing is ever being generated
concurrently there. `wait_in_queue=False` clears the backlog so the newest request is served
next instead of last; `wait_in_queue=True` leaves the backlog intact.

Only Direct mode interrupts generation that is actually in flight, because only Direct mode
lets two turns run at once.

### Configuration

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

Or on the adapter:

```python
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key=OPENAI_API_KEY,
    use_invoke_queue=True,
)
```

### Per-Request Behavior

When queue mode is enabled, control per-request behavior via `wait_in_queue`:

```python
from aiavatar.sts.models import STSRequest

# Interrupt mode (default): clears the pending queue, stops previous playback
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

### Caveats

- **Python 3.11+ required**: Queue mode uses `asyncio.timeout()` which is only available in Python 3.11 and later.
- **Session-based queues**: Each session has its own independent queue. Requests from different sessions do not affect each other.
- **Do not mix modes**: The `use_invoke_queue` setting should remain consistent for a pipeline instance. Changing it at runtime is not supported.
- **Cancelled responses**: When a queued request is cleared (by a non-waiting request), it receives a response with `type="cancelled"`.

## Quick Response

To reduce the first response latency, `QuickResponder` generates a short acknowledgment phrase (e.g. "Sure!" or "なるほど。") and sends it to the user immediately, before the main LLM response is ready. This keeps the conversation feeling responsive while the full answer is being generated.

```python
from aiavatar.sts.quick_responder import (
    QuickResponder,
    DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA,
    DEFAULT_REQUEST_PREFIX_JA,
)
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

### QuickResponderPro

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

## Custom Behavior

Both sides of the connection can react to responses as they pass, but they use different
hooks. Reaching for the wrong one is the most common mistake here, because the names are
similar.

### On the client

`AIAvatarClientBase` — so `AIAvatarWebSocketClient` and `AIAvatarHttpClient` — offers
`on_response(response_type)`, a decorator factory keyed by response type. This is where
avatar-side behaviour belongs, because the client is what owns the face and the animation.

```python
from aiavatar.adapter.websocket.client import AIAvatarWebSocketClient

client = AIAvatarWebSocketClient(url="ws://localhost:8000/ws")

# Set face when the character is thinking the answer
@client.on_response("start")
async def on_start_response(response):
    await client.face_controller.set_face("thinking", 3.0)

# Reset face before answering
@client.on_response("chunk")
async def on_chunk_response(response):
    if response.metadata.get("is_first_chunk"):
        client.face_controller.reset()
```

`face_controller` exists only on the client. See [Avatar control](avatar.md).

### On the server

`Adapter` offers `on_response(func)` — a plain decorator, not keyed by type. It is called for
every response chunk with both the outgoing `AIAvatarResponse` and the pipeline's
`STSResponse`, which makes it the place for logging, metrics, and rewriting what goes out.

```python
from aiavatar.adapter.models import AIAvatarResponse
from aiavatar.sts.models import STSResponse

@aiavatar_app.on_response
async def on_response(aiavatar_response: AIAvatarResponse, sts_response: STSResponse):
    if sts_response.type == "start":
        logger.info("Turn started: %s", sts_response.context_id)
```

Branch on `response.type` yourself, since this hook receives everything. Rewriting
`aiavatar_response` here changes what the client receives — that is how
[artifact validation](artifacts.md#validating-what-reaches-the-browser) is done.

## Request Validation

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

### Early Validation with AzureStreamSpeechDetector

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

## Performance recording

Every turn is timed stage by stage and written to a `PerformanceRecord`. This is how you
find out where a slow response actually went.

```python
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder

sts = STSPipeline(
    performance_recorder=SQLitePerformanceRecorder(db_path="aiavatar.db"),
    # ...
)
```

A record carries the identifiers (`transaction_id`, `session_id`, `context_id`, `user_id`,
`channel`), the component names actually used (`stt_name`, `llm_name`, `tts_name`), the
request and response text, and the timings:

**Every timing is a cumulative lap, not a phase duration.** Each field records how long had
elapsed since the origin when that point was reached — so `tts_first_chunk_time` already
contains the STT and LLM time before it. To get the cost of one stage, subtract the previous
lap.

| Field | Elapsed at the moment… |
| --- | --- |
| `silence_threshold_time` | the silence threshold was reached |
| `stt_after_threshold_time` | recognition still outstanding at that point finished |
| `turn_end_gate_time` | the turn-end gates decided (`turn_end_gate_held` says whether one held) |
| `stt_time` | speech recognition completed |
| `stop_response_time` | any previous response had been stopped |
| `before_llm_time` | preprocessing finished and the LLM call was about to go out |
| `llm_first_chunk_time` | the model's first token arrived |
| `llm_first_voice_chunk_time` | the first *speakable* chunk arrived |
| `llm_time` | generation finished |
| `tts_first_chunk_time` | the first audio was synthesized |
| `tts_time` | synthesis finished |
| `total_time` | the turn ended |

`voice_length` is the odd one out: it is the duration of the user's utterance, not a lap.
`quick_response_text`, `tool_calls`, and `error_info` are recorded alongside.

So the number that matters most for a voice avatar — time to first audio — is
`tts_first_chunk_time` read directly, and the LLM's own cost is
`llm_time - before_llm_time`.

### Where the clock starts

`time_origin` decides what zero means, and the default is **`user_speech_end`**.

| Origin | Zero is | Effect |
| --- | --- | --- |
| `user_speech_end` (default) | The moment the user stopped speaking | The VAD, gate, and recognition time before the pipeline ran is added on top of every pipeline lap |
| `pipeline_start` | The moment the pipeline was invoked | Laps are stored exactly as measured, excluding everything before |

The default is the honest one: it measures what the user actually waits through, including
the silence the detector spent deciding the turn was over. Switch to `pipeline_start` when
you are profiling the pipeline itself and want the VAD out of the picture.

`PostgreSQLPerformanceRecorder` in `aiavatar.sts.performance_recorder.postgres` stores the
same records in PostgreSQL. The Admin Panel reads them directly — see
[Administration](admin.md).

## Voice recording

The pipeline can persist the audio of each turn, which is what you want for building
evaluation sets and for working out why recognition failed on a particular utterance.

```python
from aiavatar.sts.voice_recorder.file import FileVoiceRecorder

sts = STSPipeline(
    voice_recorder=FileVoiceRecorder(record_dir="recorded_voices"),
    voice_recorder_enabled=True,
    # ...
)
```

Recording is enabled by default and writes to `recorded_voices`. Both the request audio and
the response audio chunks are captured, keyed by `transaction_id`. Writing happens on a
background worker through a queue, so it never blocks the turn.
`AzureBlobVoiceRecorder` in `aiavatar.sts.voice_recorder.azure_storage` uploads to Azure
Blob Storage instead of the local filesystem.

## See also

- [Speech detector](vad.md) — how a turn begins and ends
- [LLM](llm.md) — generation, tags, and context
- [Adapters](adapters.md) — how channels feed the pipeline
- [Database](database.md) — where context, state, and records are stored
- [Administration](admin.md) — reading performance records in the browser

---

[← Documentation index](../README.md#-documentation)
