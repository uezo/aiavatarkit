# Anthropic Claude

`ClaudeService` calls the Claude API natively, which is the path to use when you want
Claude's own behaviour rather than an OpenAI-shaped approximation of it.

```sh
pip install anthropic
```

Create instance of `ClaudeService` with custom parameters and set it to `AIAvatar`. The default model is `claude-sonnet-4-5`.

```python
# Create ClaudeService
from aiavatar.sts.llm.claude import ClaudeService
llm = ClaudeService(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model="claude-sonnet-4-5",
    temperature=0.0,
    system_prompt="You are my cat."
)

# Create the adapter with ClaudeService
aiavatar_app = AIAvatarWebSocketServer(
    llm=llm,
    openai_api_key=OPENAI_API_KEY   # API Key for STT
)
```

NOTE: We support Claude on Anthropic API, not Amazon Bedrock for now. Use LiteLLM or other API Proxies.

## Constructor arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `anthropic_api_key` | `None` | |
| `base_url` | `None` | For gateways and proxies |
| `model` | `"claude-haiku-4-5"` | |
| `system_prompt` | `None` | Supports `{placeholder}` parameters |
| `temperature` | `0.5` | |
| `max_tokens` | `1024` | Raise it for long answers — this is a hard cap |
| `use_dynamic_tools` | `False` | See [Tools](tools.md) |
| `initial_messages` | `None` | Few-shot examples |
| `context_manager` | `None` | See [Database](database.md) |

`max_tokens` is worth a second look. Unlike the OpenAI services it has a real default here,
so a long answer will be truncated at 1024 tokens unless you raise it.

## Native or compatible?

Claude is also reachable through `ChatGPTService` with Anthropic's OpenAI-compatible
endpoint. Use this class instead when you need thinking configuration that actually takes
effect, strict tool schemas, or prompt caching — none of which survive the compatibility
layer. See [OpenAI-compatible APIs](llm-openai-compatible.md#anthropic-claude) for the full
list of what is lost.

Amazon Bedrock is not supported directly. Reach Bedrock-hosted Claude through
[`LiteLLMService`](llm-litellm.md) or an API proxy.

## See also

- [LLM](llm.md) — the interface and its shared behaviour
- [OpenAI-compatible APIs](llm-openai-compatible.md) — Claude through the compatibility layer
- [Tools](tools.md) — tool calling across providers

---

[← Documentation index](../README.md#-documentation)
