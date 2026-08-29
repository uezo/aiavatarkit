# OpenAI Responses API

The Responses API is OpenAI's stateful alternative to Chat Completions: instead of resending
the whole conversation each turn, you send `previous_response_id` and the server keeps the
history. AIAvatarKit offers it over two transports — ordinary HTTP, and a WebSocket
connection that removes per-request connection setup from the critical path.

`OpenAIResponsesWebSocketService` is what the built-in application uses by default, because
first-token latency is the single most visible number in a voice conversation.

**For OpenAI models, prefer this API over [Chat Completions](llm-chat-completions.md).**
Reasoning items persist across turns instead of being discarded, prompt caching hits far
more often because history is referenced by id rather than resent, and — decisive for an
agent — Chat Completions cannot combine tool calling with any reasoning effort above `none`
from GPT-5.4 onward. See
[When to prefer the Responses API](llm-chat-completions.md#when-to-prefer-the-responses-api).

Use `OpenAIResponsesService` to leverage the OpenAI Responses API. Conversation history is managed server-side via `previous_response_id`, eliminating the need for client-side context management.

```python
from aiavatar.sts.llm.openai_responses import OpenAIResponsesService
llm = OpenAIResponsesService(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-5.6-terra",
    system_prompt="You are my cat."
)

aiavatar_app = AIAvatarWebSocketServer(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

The REST service also accepts a pre-configured async client. Azure's v1 Responses endpoint
uses the regular official `AsyncOpenAI` client with an Azure `base_url`; the model remains
the Azure deployment name.

```python
import os
from openai import AsyncOpenAI

azure_client = AsyncOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url=(
        f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}"
        "/openai/v1/"
    ),
)
llm = OpenAIResponsesService(
    openai_client=azure_client,
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
)
```

For lower latency, use the WebSocket variant. It keeps a pool of persistent connections, so
each turn skips the connection setup that a fresh HTTP request pays for.

How much that is worth depends entirely on how many round trips a turn makes. OpenAI reports
alpha users seeing up to 40% lower end-to-end latency, but that figure is for agentic
workflows with 20 or more sequential tool calls — coding assistants, data pipelines. A voice
turn usually makes one call, or one plus a tool, so expect the saving of one connection
setup per round trip rather than anything like 40%. It is still the right default for a voice
avatar, because that saving lands directly on time-to-first-token — just measure your own
workload rather than budgeting from the headline.
See [OpenAI's write-up](https://openai.com/index/speeding-up-agentic-workflows-with-websockets/).

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

**NOTE:** The WebSocket variant does not accept `temperature`. Use `reasoning_effort`
(`"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`) to steer the response
instead. Neither variant supports [dynamic tool calls](#dynamic-tool-calls-are-not-supported).

## Constructor arguments

Both services share most arguments with [Chat Completions](llm-chat-completions.md):
`openai_api_key`, `system_prompt`, `model` (default `gpt-5.6-terra`), `reasoning_effort`,
`extra_body`, `initial_messages`, the splitting arguments, `voice_text_tag`,
`context_manager`, and `shared_context_ids`.

The following arguments select Responses state storage, client configuration, and
transport behavior:

| Argument | Applies to | Notes |
| --- | --- | --- |
| `response_id_store` | both | Where `previous_response_id` values are persisted |
| `openai_client` | REST | Pre-configured async OpenAI-compatible client; caller-owned |
| `base_url` | REST | Standard HTTP endpoint override |
| `ws_url` | WebSocket | WebSocket endpoint override |
| `max_connections` | WebSocket | Pooled connections, so also the parallel request ceiling |
| `max_connection_age` | WebSocket | Seconds before a pooled connection is recycled; defaults to `3300` |

`temperature` is available on the REST service only.

For the REST service, `openai_client` takes precedence over `openai_api_key` and
`base_url`; those client-construction options are ignored when a client is injected.
`await llm.close()` closes the internally constructed REST client or WebSocket pool.
An injected REST client remains the caller's responsibility.

## Dynamic tool calls are not supported

Neither Responses service implements [dynamic tool selection](tools.md#dynamic-tool-call).
Tool selection is left to the model, which is the API's own design, so there is no
`use_dynamic_tools` argument and no `get_dynamic_tools` hook here. Server-side history
through `previous_response_id` also does not fit the pre-flight filtering call that dynamic
selection depends on.

`add_tool(tool, is_dynamic=True)` is accepted and raises nothing, but the flag is never read:
every registered tool is sent on every turn. If your application depends on narrowing a large
tool catalogue before each call, use [Chat Completions](llm-chat-completions.md) or one of
the other native services instead.

## Where the conversation lives

With Chat Completions, history is a list of messages you own, stored by a
[`ContextManager`](database.md). With the Responses API it can instead live on OpenAI's
side, keyed by response id — so what AIAvatarKit persists is the id, through a
`ResponseIdStore`. `SQLiteResponseIdStore` is the default;
`PostgreSQLResponseIdStore` is available for multi-process deployments.

This has a practical consequence: server-side context is subject to OpenAI's retention, not
yours. If you need the transcript for your own audit or evaluation, keep recording it
yourself — see [Pipeline](pipeline.md) for performance and voice recording.

## WebSocket transport

`OpenAIResponsesWebSocketService` keeps a pool of connections open and reuses them across
turns. Connections are recycled once they reach `max_connection_age` (3300 seconds by
default, comfortably under the hour mark) so that a long-lived avatar never accumulates
stale sockets.

```python
from aiavatar.sts.llm.openai_responses_websocket import OpenAIResponsesWebSocketService

llm = OpenAIResponsesWebSocketService(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-5.6-terra",
    system_prompt="You are my cat.",
    max_connections=100,
)
```

`max_connections` bounds how many requests can be in flight at once, so size it against
your expected concurrent sessions rather than leaving it at the default in a busy
deployment.

For Azure OpenAI WebSocket mode, use the Azure WebSocket base URL and the deployment name.
The service appends `/v1/responses` to `ws_url`:

```python
llm = OpenAIResponsesWebSocketService(
    openai_api_key=AZURE_OPENAI_API_KEY,
    ws_url="wss://YOUR_RESOURCE.openai.azure.com/openai",
    model=AZURE_OPENAI_DEPLOYMENT,
)
```

The WebSocket implementation speaks the event protocol directly rather than going through
the OpenAI Python HTTP client. A Langfuse OpenAI client can trace
`OpenAIResponsesService`, but it does not automatically observe the raw WebSocket
transport. WebSocket tracing requires separate manual instrumentation.

From the CLI, `AIAVATAR_LLM_API` selects between two values: `responses-websocket` (the
default) and `chat-completions`. There is no CLI option for the REST Responses service —
anything else raises at startup. Use it from your own application script instead. See
[Getting started](getting-started.md#openai-and-llm-configuration).

## See also

- [Chat Completions](llm-chat-completions.md) — the stateless OpenAI API
- [LLM](llm.md) — the interface and its shared behaviour
- [Database](database.md) — context managers and response id stores

---

[← Documentation index](../README.md#-documentation)
