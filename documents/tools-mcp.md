# MCP

Tools published by an MCP server can be pulled into an `LLMService` and used like any other
tool. Both Streamable HTTP and standard I/O transports are supported.

```sh
pip install fastmcp
```

## The two-step pattern

Registration happens in two steps, and both are required:

1. **Declare what to do with each tool** — set `for_each_tool`, usually to `llm.add_tool`.
2. **Connect and enumerate** — `await mcp.initialize()`. This opens the client, lists the
   server's tools, wraps each one, and only then calls your `for_each_tool` for each.

Setting `for_each_tool` alone registers nothing. It is a callback that `initialize()`
invokes; without that call the server is never contacted and `llm.tools` stays empty.

`initialize()` also returns the list of `Tool` objects, if you would rather handle them
yourself than through the callback.

```python
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.llm.tools.mcp import StreamableHttpMCP

llm = ChatGPTService(openai_api_key=OPENAI_API_KEY)

mcp = StreamableHttpMCP(url=MCP_URL)
mcp.for_each_tool = llm.add_tool

await mcp.initialize()   # Required — this is what actually registers the tools
```

Pair it with `await mcp.close()` when you are finished, so the transport is shut down
cleanly.

## Wiring it into an application

An MCP connection is long-lived, so it belongs in the application lifespan rather than in
request handling. FastAPI's `lifespan` is the natural place: connect on startup, close on
shutdown.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.sts.llm.tools.mcp import StreamableHttpMCP

aiavatar_app = AIAvatarWebSocketServer(openai_api_key=OPENAI_API_KEY)

mcp = StreamableHttpMCP(url=MCP_URL)
mcp.for_each_tool = aiavatar_app.sts.llm.add_tool


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await mcp.initialize()
        yield
    finally:
        await mcp.close()


app = FastAPI(lifespan=lifespan)
app.include_router(aiavatar_app.get_websocket_router())
```

`MCPBase` is also an async context manager, which is tidier when the connection's lifetime
matches a block:

```python
async with StreamableHttpMCP(url=MCP_URL) as mcp:
    mcp.for_each_tool = llm.add_tool
    ...
```

Note the ordering trap: `__aenter__` calls `initialize()` immediately, so a `for_each_tool`
assigned *inside* the block arrives too late for that first enumeration. Set it before
entering, or call `initialize()` yourself:

```python
mcp = StreamableHttpMCP(url=MCP_URL)
mcp.for_each_tool = llm.add_tool

async with mcp:
    ...
```

## Transports

**Streamable HTTP** takes a URL, and optional headers for authentication:

```python
mcp = StreamableHttpMCP(
    url=MCP_URL,
    headers={"Authorization": f"Bearer {MCP_JWT}"},
    sse_read_timeout=60.0,
)
```

**Standard I/O** launches a local script — `.py` and `.js` are both supported:

```python
from aiavatar.sts.llm.tools.mcp import StdioMCP

mcp = StdioMCP(server_script="weather.py")
```

## Filtering and rewriting tools

Because `for_each_tool` receives each `Tool` before it is registered, it is the place to
select a subset, rename things, tighten a schema, or wrap the call:

```python
from aiavatar.sts.llm import Tool

ALLOWED = {"get_weather", "get_forecast"}

@mcp.for_each_tool
def register(tool: Tool):
    if tool.name not in ALLOWED:
        return                      # Skip everything else
    tool.spec["function"]["description"] += " Use only for Japanese cities."
    llm.add_tool(tool)

await mcp.initialize()   # Still required — the callback runs from here
```

Skipping tools you do not need is worth doing. Every registered tool costs prompt space and
gives the model one more thing to choose wrongly — see
[Dynamic tool call](tools.md#dynamic-tool-call) for the alternative when there are many.

Exceptions raised inside `for_each_tool` are logged and the remaining tools still register,
so one bad tool will not stop the rest.

## Several servers at once

Each connection is independent; register them all against the same `LLMService`.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    started_mcps = []
    try:
        for mcp in (weather_mcp, calendar_mcp, docs_mcp):
            started_mcps.append(mcp)
            await mcp.initialize()
        yield
    finally:
        for mcp in reversed(started_mcps):
            await mcp.close()
```

Tool names must be unique across servers, since the model sees one flat list.

## See also

- [Tools](tools.md) — what happens once a tool is registered
- [Built-in tools](tools-builtin.md) — tools that ship with AIAvatarKit
- [LLM](llm.md) — the service tools are registered on

---

[← Documentation index](../README.md#-documentation)
