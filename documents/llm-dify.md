# Dify

`DifyService` connects the pipeline to a Dify application, so the prompt, the model choice,
and any workflow around them live in Dify rather than in your Python code.

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

# Create the adapter with DifyService
aiavatar_app = AIAvatarWebSocketServer(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

## Constructor arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `api_key` | `None` | The Dify application's API key |
| `base_url` | `"http://127.0.0.1"` | Your Dify instance |
| `user` | `None` | Passed through as Dify's end-user identifier |
| `is_agent_mode` | `False` | Set for Dify agent applications |
| `make_inputs` | `None` | Callable building the `inputs` payload per request |
| `timeout` | `10.0` | |

There is no `system_prompt`, `model`, or `context_manager` here, and that is the point:
Dify owns the prompt, the model, and the conversation state. The pipeline supplies speech
and consumes the stream.

`make_inputs` is how request-specific values reach a Dify workflow — return the dict Dify
should receive as `inputs` for that turn.

Note that AIAvatarKit can also *serve* a Dify-compatible endpoint, which is the opposite
direction: a Dify-shaped frontend talking to your avatar. See
[HTTP adapter](adapters-http.md).

## See also

- [LLM](llm.md) — the interface and its shared behaviour
- [HTTP (SSE) adapter](adapters-http.md) — serving a Dify-compatible endpoint

---

[← Documentation index](../README.md#-documentation)
