# Built-in tools

Ready-made tools you can register without writing an integration, plus the OpenClaw and
Hermes bridges for handing a job to a full agent harness.

You can use the following tools out of the box 📦.

- 🔍 Web Search
    - Gemini Search
    - OpenAI Search
    - Grok Search
- 🌏 Web Scraper
- 🖼️ Image Generation
    - 🍌 Nano Banana
    - 🐓 Selfie
- 📝 [Character and shared memory](#character-and-shared-memory)

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

# Web Scraper (pip install playwright && playwright install chromium)
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

## Character and shared memory

`MemoryTool` keeps two scopes of memory for each user:

- `character`: memories intended for this character, including its relationship with
  the user, how it addresses the user, expression preferences, and shared experiences.
- `shared`: memories intended to carry across characters, such as user facts and
  preferences, common response rules, and task procedures. This does not share memories
  between users.

Classification follows the intended scope, not just the topic. A requested way of
being addressed defaults to `character`, including plain names, hiragana, and requests
without an honorific. Explicit requests to apply across characters use `shared`.
Facts such as birthdays, likes, or home locations, and any rule or procedure, can also
be character-specific when the user scopes them that way.

| Request | Scope |
| --- | --- |
| “Call me はる, without an honorific.” | `character` |
| “Every character should call me はる.” | `shared` |
| “My birthday is May 5; remember it across characters.” | `shared` |
| “Only you should confirm the outline before preparing reports for me.” | `character` |

Both scopes are loaded in full as Markdown; there is no retrieval tool or per-skill
selection step. The tool is opt-in: registering it and adding its text to your prompt
are explicit application changes. Existing memory tools and stored data are not
migrated or reclassified automatically.

```python
import os
from openai import AsyncOpenAI
from aiavatar.sts.llm.tools.memory import MemoryTool

# Use an OpenAI-compatible Chat Completions endpoint with JSON-object responses.
memory_update_client = AsyncOpenAI(
    api_key=os.environ["MEMORY_API_KEY"],
    base_url=os.environ["MEMORY_BASE_URL"],
)
memory = MemoryTool(
    character_memory_dir="memories/alice",
    update_client=memory_update_client,
    # shared_memory_dir="memories/shared",  # Optional explicit override.
    # update_model="gpt-6-sol",  # Default; override for another model/deployment.
    # reasoning_effort="low",   # Optional; None omits the API parameter.
    # debug=True,               # Optional INFO logs; defaults to False.
)
llm.add_tool(memory)

# Use this example when the application uses the default system prompt.
@llm.get_system_prompt
async def get_system_prompt(context_id, user_id, system_prompt_params):
    base_prompt = await llm.get_system_prompt_default(
        context_id, user_id, system_prompt_params
    )
    memory_prompt = await memory.get_prompt(user_id)
    return "\n\n".join(part for part in (base_prompt, memory_prompt) if part)
```

`update_client` accepts an `openai.AsyncOpenAI` client. The update endpoint and model
must support `response_format={"type": "json_object"}`.
`update_model` defaults to `"gpt-6-sol"`. `reasoning_effort` defaults to `None`, which
omits the parameter and leaves the effective effort to the endpoint's default.
Set it to `"low"` for faster updates or `"high"` for more reasoning. The supplied value
is forwarded unchanged; there is no fallback for models that do not support it.
If you already have a `get_system_prompt` hook, append `await memory.get_prompt(user_id)`
inside that hook, preserving its existing character or application prompt construction.
Registering another hook replaces the previous one. Close `memory_update_client` with
`await memory_update_client.close()` during application shutdown; the tool does not own
the supplied client.

Set `debug=True` to trace memory updates with the application's logger at `INFO`.
JSON logs prefixed `MemoryTool: ` record `input`, `request`, `raw_response`, `decision`,
`result`, `retry`, and `error` events, correlated by `operation_id` and selected request
IDs. The request includes the actual update messages, model, and any configured
reasoning effort; the raw response is logged before parsing, followed by the decision's
scope, reason, and operations and the resulting status and available revision.
Failure logs identify the stage and, for exceptions, their class without the exception
message. Client/API-key settings and full metadata are not serialized.

The updater receives the tool's `content`, current memories, and classification
prompt; it does not receive the original conversation. The conversation model must
therefore preserve relevant dialogue intent, subject, actor, and conditions in
`content`: an answer to “What should I call you?” must retain the requested form of
address, rather than being reduced to a profile-name fact. Missing context that makes
scope ambiguous should produce `needs_clarification`, without inventing a scope. The updater's `reason` is
a short model-provided explanation, not hidden reasoning. Logs contain memory content
and potentially personal information, so enable this option only when needed for debugging.

Background execution uses the existing `Tool.on_completed` mechanism. To use it,
configure the memory tool with the desired effort (for example, `reasoning_effort="high"`)
and replace `llm.add_tool(memory)` above with this block. Register the callback and set
the acceptance message **before** adding the tool, since registration may copy it.
Here `aiavatar_app` is your existing `AIAvatarWebSocketServer` using the same `llm`:

```python
from aiavatar.sts.models import STSRequest

memory.immediate_message = "Memory update accepted; it has not finished yet."

@memory.on_completed
async def on_memory_completed(result, metadata):
    status = result.get("status") if result is not None else "error"
    if status == "needs_clarification":
        reason = result.get("reason") or "Please specify which memory to update."
        message = f"The memory update is on hold and needs clarification: {reason}"
    elif status == "error":
        message = "The memory update failed. Please try again."
    else:
        return  # No completion notification for saved or unchanged.

    async for response in aiavatar_app.sts.invoke(
        STSRequest(
            session_id=metadata["session_id"],
            user_id=metadata["user_id"],
            context_id=metadata["context_id"],
            channel=metadata.get("channel"),
            text=(
                "$Internal memory-update notification. Briefly relay the following "
                "to the user, asking for clarification if needed. Do not retry "
                f"the memory update in response to this notification.\n\n{message}"
            ),
            allow_merge=False,
            wait_in_queue=True,
            skip_quick_response=True,
        )
    ):
        await aiavatar_app.handle_response(response)

llm.add_tool(memory)
```

This callback generates the notification through the usual LLM/TTS pipeline and
delivers each response to the originating WebSocket session. It sends completion
notifications only for `error` (including a `None` result from an exception) and
`needs_clarification`. The latter preserves the current memory and supplies the update
model's explanation in `reason`. The initial acceptance response is separate from
these completion notifications. `wait_in_queue=True` queues the notification when
the pipeline's invocation queue is enabled; `allow_merge=False` keeps it separate
from the preceding user request.

With the default `background_timeout=None`, the tool returns an acceptance response
without waiting for the update. The completion callback receives the final result;
it does not automatically send a user notification. New memories appear in prompts
after the save finishes. Without an `on_completed` callback, tool execution waits for
the save and returns its result normally. Directly awaiting `save_memory()` also waits
for the save regardless of callback registration.

`shared_memory_dir` is optional. When omitted or `None`, it defaults to
`Path(character_memory_dir).parent / "shared"`. The example above therefore uses these
separate files for `user_id="uezo"`:

```text
memories/alice/uezo.json
memories/shared/uezo.json
```

Changing `character_memory_dir` to `"memories/bob"` switches character memories while
keeping the same user's shared memories. Set `shared_memory_dir` explicitly when
characters in different parent directories should use the same shared store.

Lowercase ASCII letters, digits, `_`, and `-` remain readable in filenames. Other UTF-8
bytes, including uppercase letters and dots, are percent-escaped. Lowercase Windows
device names and 64-character lowercase hexadecimal IDs escape their first byte; very long
encoded IDs use a bounded prefix plus `~` and the full SHA-256 hash. Existing files
named solely by the SHA-256 hash continue to be read and updated in place, without
renaming. If both the readable and legacy filenames exist for the same user and
category, access fails rather than selecting or merging them.

Saves require `metadata["user_id"]`; the model does not choose the user or a filesystem
path. JSON contains a revision and entries with `id` and `content`. To edit it manually,
stop saves while editing, preserve the document schema and entry IDs, and retain or
increment `revision`. Conflict checks compare the whole document. IDs are used by the
update planner; `get_prompt()` emits only readable content under separate character
and shared headings.

The conversation model calls `save_memory(content, forget=False)` with one fact or rule
at a time. An update LLM classifies it as `character` or `shared` and proposes only
the necessary additions, replacements, or deletions. Explicit forgetting uses
`forget=True`. Only existing entries referenced by a validated patch can be changed; unrelated
entries are preserved without rewriting or summarizing them. Semantic conflict detection
still depends on the update model. Duplicates require the same subject, meaning,
scope, and conditions. Conflicts concern the same subject, actor, and scope where
conditions overlap; corrections preserve unaffected conditions. A character-specific
exception can coexist with a shared default. The result status is `saved`, `unchanged`,
`needs_clarification`, or `error`, so an uncertain or failed save is not reported as
successful.

Updates use atomic file replacement and file locking, and retry when the stored
document has changed. Only the most recent previous revision is retained in a
`.previous.json` file, and a `.lock` sidecar is also kept. Forgetting removes content from active memory
and future prompts; it does not erase that backup. Missing files are treated as empty
independently: a new character with no memory file still loads the user's shared
memories. If both scopes are empty, `get_prompt()` returns an empty string. Reading
does not create files or directories.

To pass the user's shared facts, preferences, and procedures to a harness, read them
when preparing its request:

```python
shared_prompt = await memory.get_prompt(user_id, kind="shared")
query_with_memories = "\n\n".join(
    part for part in (query, shared_prompt) if part
)
# Pass query_with_memories to your existing harness call.
```

The same memories can guide AIAvatarKit's own tool selection through the system prompt
above. The default tool name is `save_memory`; use `name="save_character_shared_memory"`
if another registered tool already has that name.

## OpenClaw / Hermes

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

### Push-based delivery (WebSocket / Local)

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

### Polling-based delivery (HTTP)

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

### Progress tracking

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

### Report channel routing

This applies once you have registered an `on_completed` callback, which is what makes the
tool run in the background. Without one, `OpenClawTool` behaves like any other tool: the
result goes back to the model and the avatar speaks it, and `report_channel` is irrelevant.
See [Background Tool Execution](tools.md#background-tool-execution).

With the callback registered, `report_channel` records **where the LLM intends the answer to
go**. Nothing delivers it for you: AIAvatarKit stores the value and hands it back with the
result, and your `on_completed` decides what to do with it. There is no automatic routing,
not even back to the originating channel — see
[Push-based delivery](#push-based-delivery-websocket--local) for the code that actually sends
the answer.

The LLM can name a channel when it invokes the tool:

```python
# LLM calls: send_query_to_openclaw(query="...", report_channel="linebot")
```

Or change it mid-flight, once it knows the `task_id`:

```python
llm.add_tool(openclaw_tool.create_set_report_channel_tool())
```

That lets the LLM call `set_openclaw_report_channel(task_id="...", report_channel="sms")`
while the task is still running.

Either way the value arrives on the result, and routing on it is your job:

```python
@openclaw_tool.on_completed
async def on_completed(result, metadata):
    channel = result.get("report_channel") or metadata.get("channel")
    answer = result["answer"]

    if channel == "linebot":
        await line_adapter.handle_push_request(
            user_id=metadata["user_id"],
            text=answer,
            context_id=metadata.get("context_id"),
        )
    else:
        # Push it back through the pipeline on the originating channel
        ...
```

`metadata["channel"]` carries the channel the request came in on, which is the sensible
fallback when the LLM did not set one.

### Per-user configuration

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

### Custom harness

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

## Grok Bot

`GrokBotTool` sends a job to Grok Bot over HTTP and waits for its webhook response.
This integration uses a custom webhook contract that you need to agree on with
your Grok Bot.

### Agree on the webhook integration with Grok Bot

Ask Grok Bot to accept tasks from your agent and return the results via webhook.
The message below explains the JSON formats expected by `GrokBotTool` and asks
Grok Bot to provide a request webhook URL and an API key to protect it.

Before sending the message, prepare a callback endpoint reachable from Grok Bot
and choose a key to authenticate callbacks. Replace `<CALLBACK_URL>` and
`<RESPONSE_KEY>` with those values. The example in "Connect the tool" below shows
how to add the callback endpoint to your app.
If you change the tool's `timeout`, adjust the five-minute interval in the message.

```text
I'd like to send you tasks from another AI agent I use. I plan to exchange both
requests and responses through webhook calls, using the JSON formats described
below. Please provide a webhook URL for sending you requests and an API key to
protect that endpoint.

Accept HTTP POST requests with Content-Type: application/json. Authenticate the
Authorization header, which will contain "Bearer " followed by the request key
you provide.

The request JSON will look like this:
{"text": "The task to perform", "request_id": "request-123", "requested_at": "2026-01-01T00:00:00+00:00"}

text is the task, request_id identifies this request, and requested_at is its
original submission time in UTC.
After accepting the request, return HTTP 200 or 202 immediately and perform the
task in the background. The HTTP response body can be a simple acknowledgement;
send the actual answer through the callback below.

For callbacks, POST JSON to <CALLBACK_URL> with these headers:
Authorization: Bearer <RESPONSE_KEY>
Content-Type: application/json

Always copy the original request_id unchanged. Use these status values:

Progress update:
{"request_id": "request-123", "status": "progress", "text": "Finished checking the sources; now preparing the answer."}

Final answer:
{"request_id": "request-123", "status": "done", "text": "The final answer."}

Cancellation:
{"request_id": "request-123", "status": "canceled"}

progress and done require a string text. canceled does not require text.
For long tasks, send progress describing what is finished or currently happening
before five minutes pass without an update. Each progress update resets the
five-minute wait. Finish each request with one done or canceled callback.

Use canceled only when a request is withdrawn or a later correction makes its
result unnecessary. Send canceled with the old request's ID, and handle the new
request under its own ID. Independent requests should continue independently;
a newer request alone is not a reason to cancel an earlier one.
Cancellation is silent on the user's side. If a task fails or cannot be completed,
send done with an explanation in text so the user receives an answer.

You may include requested_at in callbacks, but it is optional. If included, copy
the original value unchanged.
```

Configure the URL and request key returned by Grok Bot as `GROKBOT_WEBHOOK_URL`
and `GROKBOT_WEBHOOK_KEY` in your app. Set `GROKBOT_WEBHOOK_RESPONSE_KEY` to the
response key you supplied in the message. The callback URL is configured on
Grok Bot's side, rather than included in each request.

### Connect the tool

`requested_at` is the original submission time in ISO 8601 UTC. Grok Bot can echo
it in callbacks, but the tool retains its own copy, so the echo is optional. Include
this timestamp and the original request in completion instructions so the LLM can
recognize a request superseded or canceled in later conversation. Older independent
requests remain useful; the tool does not discard their results automatically.

Add the following to your existing FastAPI app, using its `app`, `llm`, and
`aiavatar_app` instances. Register the completion callback **before** `llm.add_tool`
to enable background execution and deliver the answer to the originating session:

```python
import os
from fastapi import HTTPException, Request
from aiavatar.sts.llm.tools.grokbot_tool import GrokBotTool
from aiavatar.sts.models import STSRequest

grokbot_tool = GrokBotTool(
    webhook_url=os.environ["GROKBOT_WEBHOOK_URL"],
    webhook_key=os.environ["GROKBOT_WEBHOOK_KEY"],
)
response_key = os.environ["GROKBOT_WEBHOOK_RESPONSE_KEY"]

@grokbot_tool.on_completed
async def on_grokbot_completed(result, metadata):
    if result is None:
        notification = "Grok Bot request failed."
    else:
        notification = (
            f"Request ID: {result['request_id']}\n"
            f"Requested at: {result['requested_at']}\n"
            f"Original request: {result['request']}\n\n"
            f"Answer: {result['answer']}"
        )
    async for response in aiavatar_app.sts.invoke(
        STSRequest(
            session_id=metadata["session_id"],
            user_id=metadata["user_id"],
            context_id=metadata["context_id"],
            channel=metadata.get("channel"),
            text=(
                "$Grok Bot returned a response. Consider later conversation: "
                "if this request was superseded or canceled, remain silent. "
                "Do not suppress an independent request merely because it is older. "
                "Otherwise, relay the relevant result to the user.\n\n"
                f"{notification}"
            ),
            allow_merge=False,
            wait_in_queue=True,
            skip_quick_response=True,
        )
    ):
        await aiavatar_app.handle_response(response)

llm.add_tool(grokbot_tool)
llm.add_tool(grokbot_tool.create_check_tool())

@app.post("/grokbot/callback")
async def grokbot_callback(request: Request):
    if request.headers.get("Authorization") != f"Bearer {response_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    payload = await request.json()
    grokbot_tool.receive_response(
        payload["request_id"], payload.get("text"), payload.get("status", "done")
    )
    return {"ok": True}
```

The LLM calls `send_query_to_grokbot(query="...")`. Its `request_id` is the background
`task_id`, or a new UUID when called without one. `requested_at` uses the tool's
`submitted_at` metadata when available, otherwise the current time.
`receive_response(request_id, text=None, status="done")` returns whether it accepted
the update for a pending request. `done` completes the request with the result
`{"answer": text, "request_id": "...", "requested_at": "...", "request": query}`.
Without an `on_completed` callback, execution waits for that result normally.
See [Background Tool Execution](tools.md#background-tool-execution) for other
delivery patterns.

`canceled` ends the request silently, discards its accumulated progress, and skips
`on_completed`. Later callbacks for that request ID are ignored; other requests
continue normally. Use `canceled` for requests whose results are no longer needed.
To report a failure to the user, send `done` with explanatory `text` instead.

Progress messages accumulate until completion or cancellation without triggering
spoken notifications, like OpenClaw/Hermes progress tracking. The check tool's default name is
`check_running_grokbot_tasks`; it returns `running_tasks` with `request_id`, `request`,
`requested_at`, and `progress`, scoped to the current context and user. This lets the
avatar answer questions such as "How's it going?" from the received updates.

Requests run independently, so a slow earlier request does not block a later result.
With the pipeline's invocation queue enabled, completion notifications are queued in
the order they reach the queue, rather than the original submission order.

`timeout=300.0` controls the callback inactivity timeout in seconds. The initial
five-minute window starts after the outbound HTTP request completes; each accepted
`progress` update resets that request's deadline to five minutes from the update
(or the configured `timeout`). A job can therefore keep running longer than five
minutes while progress continues. If no further progress or final response arrives
within that window, it times out. Updates for one request do not extend other requests.
`http_timeout=30.0` separately controls the outbound HTTP request and is not reset
by progress updates.
Pending requests live in memory, so run the tool and callback in one worker;
restarting loses pending requests. Call
`await grokbot_tool.aclose()` in your application's shutdown/lifespan cleanup.

## See also

- [Tools](tools.md) — registering tools and shaping their output
- [MCP](tools-mcp.md) — tools from MCP servers

---

[← Documentation index](../README.md#-documentation)
