# LiteLLM

`LiteLLMService` routes through [LiteLLM](https://github.com/BerriAI/litellm), which speaks
to a hundred-plus model providers behind one interface. It is the escape hatch: when a
provider has no native class and no OpenAI-compatible endpoint, this is how you reach it.

```sh
pip install litellm
```

You can use other LLMs by using `LiteLLMService` or implementing `LLMService` interface.

See the details of LiteLLM here: https://github.com/BerriAI/litellm

```python
from aiavatar.sts.llm.litellm import LiteLLMService

llm = LiteLLMService(
    api_key=API_KEY,
    model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    system_prompt="You are my cat.",
)
```

## Constructor arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `api_key` | `None` | |
| `base_url` | `None` | For self-hosted gateways and LiteLLM proxy |
| `model` | `None` | A LiteLLM model string, usually `provider/model` |
| `system_prompt` | `None` | Supports `{placeholder}` parameters |
| `system_prompt_by_user_prompt` | `False` | Send the system prompt as a user message for models with no system role |
| `temperature` | `None` | |
| `reasoning_effort` | `None` | Passed through where the provider supports it |
| `use_dynamic_tools` | `False` | See [Tools](tools.md) |
| `context_manager` | `None` | See [Database](database.md) |

`system_prompt_by_user_prompt` exists for models that have no system role at all. Turning it
on prepends the system prompt as a user message instead of dropping it.

## When to reach for it

LiteLLM adds a dependency and a layer of indirection, so prefer a native class or an
OpenAI-compatible endpoint when one exists. It earns its place for Amazon Bedrock, Vertex AI,
and self-hosted or regional providers that speak neither.

Because behaviour varies by the provider underneath, treat tool calling, streaming, and
reasoning support as properties of the target model and verify them before relying on them.

## See also

- [LLM](llm.md) — the interface and its shared behaviour
- [OpenAI-compatible APIs](llm-openai-compatible.md) — the lighter option when it applies
- [Claude](llm-claude.md) — reaching Bedrock-hosted Claude

---

[← Documentation index](../README.md#-documentation)
