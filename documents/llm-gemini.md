# Google Gemini

`GeminiService` calls the Gemini API through Google's own SDK, which gives you thinking
configuration as real constructor arguments rather than something smuggled through
`extra_body`.

```sh
pip install google-genai
```

```python
from aiavatar.sts.llm.gemini import GeminiService

llm = GeminiService(
    gemini_api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.0,
    system_prompt="You are my cat."
)

# Create the adapter with GeminiService
aiavatar_app = AIAvatarWebSocketServer(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

NOTE: We support Gemini on Google AI Studio, not Vertex AI for now. Use LiteLLM or other API Proxies.

## Constructor arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `gemini_api_key` | `None` | |
| `model` | `"gemini-2.5-flash"` | |
| `system_prompt` | `None` | Supports `{placeholder}` parameters |
| `temperature` | `0.5` | |
| `thinking_level` | `None` | Thinking depth |
| `thinking_budget` | `-1` | Token budget for thinking; `-1` leaves it to the model |
| `use_dynamic_tools` | `False` | See [Tools](tools.md) |
| `initial_messages` | `None` | Few-shot examples |
| `context_manager` | `None` | See [Database](database.md) |

## Thinking and latency

`thinking_level` and `thinking_budget` are the two levers. When both are set,
`thinking_level` wins; when `thinking_level` is unset and `thinking_budget` is left at
`-1`, no thinking configuration is sent at all and the model decides for itself. For a voice avatar, a thinking
pass is time the user spends in silence, so start conservative and raise it only where the
task actually needs it.

Note that Gemini does not let you turn thinking off everywhere: on Gemini 2.5 Pro and on
Gemini 3 models it cannot be disabled. Budget for it rather than fighting it — a
[quick response](pipeline.md#quick-response) covers the gap while the model thinks.

## Native or compatible?

Gemini is also reachable through `ChatGPTService` with Google's OpenAI-compatible endpoint,
which Google labels beta. Use this class when you want thinking configuration as a supported
argument, and the compatible endpoint when you want one code path across several providers.
See [OpenAI-compatible APIs](llm-openai-compatible.md#google-gemini).

## See also

- [LLM](llm.md) — the interface and its shared behaviour
- [OpenAI-compatible APIs](llm-openai-compatible.md) — Gemini through the compatibility layer
- [Pipeline](pipeline.md) — covering thinking time with a quick response

---

[← Documentation index](../README.md#-documentation)
