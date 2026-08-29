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

## See also

- [Tools](tools.md) — registering tools and shaping their output
- [MCP](tools-mcp.md) — tools from MCP servers

---

[← Documentation index](../README.md#-documentation)
