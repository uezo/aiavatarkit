# Tools

Tool calling in a voice conversation has a constraint that text agents do not share: silence
reads as a fault. A tool that takes eight seconds is fine in a chat window and unacceptable
when someone is waiting to be spoken to. Everything on this page exists to keep the
conversation moving while work happens.

Register through `add_tool` and a spec written once works on GPT, Gemini, and Claude alike —
it is rebuilt for whichever service it lands on, so switching provider does not mean
rewriting your tools. The `@llm.tool()` decorator does not convert; see
[which registration path converts the spec](#which-registration-path-converts-the-spec).

AIAvatarKit is not just a framework for creating chatty AI characters — it is designed to support agentic characters that can interact with APIs and external data sources (RAG).

## Tool Call

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

### Which registration path converts the spec

The two paths are not equivalent, and this is worth getting right before you write many
tools.

| | `@llm.tool(spec)` | `llm.add_tool(Tool(...))` |
| --- | --- | --- |
| What it does with the spec | Stores it as given | Rebuilds it for the target service |
| Spec format you must supply | That provider's own | Any of the three; it is converted |

`@llm.tool()` reads the name straight out of the structure it expects, so the format is not
interchangeable:

| Service | Spec shape `tool()` expects |
| --- | --- |
| `ChatGPTService`, `LiteLLMService` | Chat Completions: `{"type": "function", "function": {...}}` |
| `GeminiService` | Gemini: `{"functionDeclarations": [{...}]}` |
| `ClaudeService` | Anthropic: `{"name": ..., "description": ..., "input_schema": {...}}` |
| `OpenAIResponsesService`, `OpenAIResponsesWebSocketService` | Chat Completions, converted to the Responses shape for you |

`add_tool()` is the portable path. It parses whichever of the three shapes you hand it and
rebuilds the spec for the service it is being registered on, so one definition works
everywhere:

```python
llm.add_tool(Tool("get_weather", weather_tool_spec, get_weather))
```

Pass `use_original=True` to skip conversion and register the spec exactly as written.

**Note:** conversion recognises the target service by class name, matching `gpt`, `gemini`,
or `claude`. `LiteLLMService` and `DifyService` match none of them, so `add_tool()` raises
`ValueError` there — use `@llm.tool()` with the format that service expects instead.


Before creating your own tools, start with the example tools:

```python
# Google Search
from aiavatar.sts.llm.tools.gemini_websearch import GeminiWebSearchTool
aiavatar_app.sts.llm.add_tool(GeminiWebSearchTool(gemini_api_key=GEMINI_API_KEY))

# Web Scraper
from aiavatar.sts.llm.tools.webscraper import WebScraperTool
aiavatar_app.sts.llm.add_tool(WebScraperTool())
```

## Tool Call with Streaming Progress

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

## Background Tool Execution

For tools that take a long time to complete (e.g., AI agent calls, complex API orchestrations), AIAvatarKit supports **background execution**. Instead of blocking the conversation, the avatar immediately acknowledges the request and notifies the user when the result is ready via a callback.

To enable background execution, register an `on_completed` callback on the tool. This is the only requirement — the base `Tool` class handles task management, `task_id` generation, and metadata tracking automatically.

**Registering the callback is the switch.** Without it a tool runs synchronously: the result
goes back to the model, which phrases an answer, and the avatar speaks it — all automatic.
With it, the turn ends immediately after the acknowledgement, and the real result is handed
to your callback instead. Nothing speaks it for you.

That means every background tool needs its `on_completed` to close the loop — typically by
feeding the answer back through the pipeline so the avatar says it:

```python
@tool.on_completed
async def on_completed(result, metadata):
    async for resp in aiavatar_app.sts.invoke(STSRequest(
        session_id=metadata["session_id"],
        user_id=metadata["user_id"],
        context_id=metadata["context_id"],
        text=f"$The background task finished. Relay this to the user:\n\n{result['answer']}",
        wait_in_queue=True,
        skip_quick_response=True,
    )):
        await aiavatar_app.handle_response(resp)
```

`wait_in_queue=True` matters here: the answer arrives unprompted, so it should queue behind
whatever the user is currently saying rather than cutting it off. See
[Invoke Queue](pipeline.md#invoke-queue).

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

### Background Timeout (Hybrid Mode)

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

## Tool Response Formatter (Direct Response)

By default, after a tool executes, the result is passed back to the LLM to generate a human-friendly response (2nd LLM call). However, in some cases you may want to **bypass the LLM and speak the tool result directly**:

- **Accuracy**: Critical information (e.g., order details, reservation IDs) that must not be paraphrased or hallucinated
- **Latency**: Eliminating the 2nd LLM call for faster response times

Use the `@response_formatter` decorator to define a function that converts the tool result into the exact text to speak. The formatted text is spoken as soon as the tool returns, without waiting for the model to rephrase it.

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

### What it saves depends on the service

The accuracy benefit is the same everywhere: the exact text you build is what gets spoken,
with no chance for the model to reword a number. The **latency** benefit is not.

| Service | Second LLM call | Tool chain | `continue_chain` |
| --- | --- | --- | --- |
| `ChatGPTService`, `ClaudeService`, `GeminiService`, `LiteLLMService` | **Skipped** when a formatter fired | Stops by default | Honoured |
| `OpenAIResponsesService`, `OpenAIResponsesWebSocketService` | Still made — the tool result goes back to the model regardless | Continues | Ignored |

On the Responses services the formatted text is yielded immediately and the model's own
follow-up text for that round is suppressed, so the user hears your wording and only your
wording. But the request still goes out, so you do not save the round trip, and the chain
does not stop.

This matters because the built-in application uses `OpenAIResponsesWebSocketService` by
default. If you are reaching for a formatter to cut latency specifically, check which
service you are on first — see [LLM](llm.md#choosing-a-service).

### Continuing Tool Chains with `continue_chain`

On `ChatGPTService`, `ClaudeService`, `GeminiService`, and `LiteLLMService`,
`response_formatter` terminates the tool chain by default. No further LLM call is made,
which maximizes speed. However, if the LLM calls multiple tools in sequence (e.g., check
balance first, then fetch campaign info), a direct-response tool would break the chain and
prevent subsequent tools from being called.

Use `continue_chain=True` to allow the chain to continue after the direct response:

```python
@llm.tools["get_balance"].response_formatter(continue_chain=True)
def format_balance(result, arguments):
    return f"Your balance is {result['balance']:,} {result['currency']}."
```

| Decorator | Behavior on supported services |
|-----------|----------|
| `@tool.response_formatter` | Direct response, **chain stops** (default, fastest) |
| `@tool.response_formatter(continue_chain=True)` | Direct response, **chain continues** (LLM can call more tools) |

`continue_chain` is honoured by `ChatGPTService`, `ClaudeService`, `GeminiService`, and
`LiteLLMService`. The Responses services continue the chain either way, so the flag makes
no difference there.

When `continue_chain=True`, the formatted text is spoken immediately, and the tool result is also sent back to the LLM so it can decide whether to call additional tools. The LLM's text response for this round is suppressed to avoid duplication, but any subsequent tool calls and their responses proceed normally.

## Structured Content (Client-side Data)

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

## Dynamic Tool Call

AIAvatarKit supports **dynamic Tool Calls**.
When many tools are loaded up-front, it becomes harder to make the model behave as intended and your system instructions explode in size. With AIAvatarKit’s **Dynamic Tool Call** mechanism you load **only the tools that are actually needed at the moment**, eliminating that complexity.

The overall flow is illustrated below.

![Dynamic Tool Call Mechanism](images/dynamic_tool_call.png)

**Supported on `ChatGPTService`, `ClaudeService`, `GeminiService`, and `LiteLLMService`
only.** The Responses services — `OpenAIResponsesService` and
`OpenAIResponsesWebSocketService` — do not implement dynamic tool selection, by design: the
Responses API leaves tool selection to the model. They accept `is_dynamic=True` on
`add_tool()` without error, but nothing reads it, so those tools are sent on every turn like
any other. If you need dynamic tool calls, choose one of the four services above — note that
the built-in application defaults to `OpenAIResponsesWebSocketService`.

### 1. Create the tool definitions and implementations
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

### 2. Register the tools as dynamic in the AIAvatarKit LLM service

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


### 3. Tweak the system prompt so the model knows how to use tools

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


### Custom Tool Repository

By default, `get_dynamic_tools_default` sends the names and descriptions of every
registered tool to a separate LLM call, which selects the tools relevant to the current
context. This works for a moderate number of tools, but the selection prompt grows with
the full catalog.

For larger-scale systems, pair AIAvatarKit with a retrieval layer (e.g., a vector-search index) so that, out of thousands of available tools, only the handful that are truly relevant are executed.

AIAvatarKit supports this pattern through the `get_dynamic_tools` hook. Decorate an async
function taking the current messages and metadata; it returns a list of **tool specification
objects** — the specs, not the implementations — for this turn.

```python
@llm.get_dynamic_tools
async def my_get_dynamic_tools(messages: list, metadata: dict = None) -> list:
    # Retrieve candidate tools from your vector database (or any other store)
    tools = await search_tools_from_vector_db(messages, metadata)
    # Extract and return the spec objects (not the implementations)
    return [t.spec for t in tools]
```

The default selector, `get_dynamic_tools_default`, uses every registered tool as a
candidate. Replace it when the full catalog is too large to put in the selection prompt.

## See also

- [Built-in tools](tools-builtin.md) — web search, scraping, image generation, OpenClaw
- [MCP](tools-mcp.md) — tools from MCP servers
- [LLM](llm.md) — the service tools are registered on
- [Guardrail](guardrail.md) — checking what the model says about the results

---

[← Documentation index](../README.md#-documentation)
