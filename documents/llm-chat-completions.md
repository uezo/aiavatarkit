# OpenAI Chat Completions

`ChatGPTService` speaks the OpenAI Chat Completions API. It is the most widely supported
shape in the industry, which makes this one class the entry point for OpenAI itself, for
Azure OpenAI, and — through `base_url` — for a long list of other providers.

```python
from aiavatar.sts.llm.chatgpt import ChatGPTService

llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-5.6-terra",
    system_prompt="You are my cat.",
)
```

> **If your provider is OpenAI itself, use the [Responses API](llm-responses.md) instead.**
> It is faster in practice, it keeps reasoning state across turns, and on GPT-5.4 and later
> Chat Completions cannot combine tool calling with any reasoning effort above `none`. This
> page still applies to Azure OpenAI and to every
> [OpenAI-compatible provider](llm-openai-compatible.md) — see
> [When to prefer the Responses API](#when-to-prefer-the-responses-api) for the details.

## Constructor arguments

| Argument | Default | Notes |
| --- | --- | --- |
| `openai_client` | `None` | Pre-configured async OpenAI-compatible client; caller-owned |
| `openai_api_key` | `None` | |
| `base_url` | `None` | Point at any OpenAI-compatible endpoint |
| `model` | `"gpt-5.6-terra"` | Actual model ID or Azure deployment name |
| `system_prompt` | `None` | Supports `{placeholder}` parameters |
| `temperature` | `None` | Omitted from the request when `None` |
| `reasoning_effort` | `None` | Sent as a top-level parameter when set |
| `extra_body` | `None` | Merged into the request body when non-empty |
| `enable_tool_filtering` | `True` | |
| `use_dynamic_tools` | `False` | See [Tools](tools.md) |
| `initial_messages` | `None` | Few-shot examples prepended to every context |
| `voice_text_tag` | `None` | See [LLM](llm.md#voice-text-tag-think-before-answering) |
| `context_manager` | `None` | See [Database](database.md) |
| `custom_openai_module` | `None` | Deprecated module-level injection retained for compatibility |

Splitting arguments (`split_chars`, `option_split_chars`, `option_split_threshold`,
`split_on_control_tags`) control how the stream is cut into speakable units.

When `openai_client` is provided, it takes precedence over `openai_api_key`, `base_url`,
and `custom_openai_module`; those client-construction options are ignored. A client
constructed inside `ChatGPTService` is closed by `await llm.close()`. An injected client
is caller-owned, so the caller remains responsible for closing it.

## Reasoning effort

`reasoning_effort` is sent as a top-level request parameter whenever it is not `None`, and
omitted entirely when it is. That distinction matters more than it looks: several providers
reject the parameter outright rather than ignoring it.

```python
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-5.6-terra",
    reasoning_effort="none",   # Fastest first token
)
```

For a voice avatar, minimal reasoning is usually the right default — the latency of a
thinking pass is very visible when someone is waiting to be spoken to. The built-in
application reflects that: it sends `reasoning_effort="none"` unless you configure
otherwise. See [Getting started](getting-started.md#openai-and-llm-configuration).

## When to prefer the Responses API

For OpenAI, `OpenAIResponsesService` is the better default and this class is the fallback.
Three reasons, in the order they are likely to bite an avatar:

**Tool calling and reasoning are mutually exclusive here.** OpenAI states that *"starting
with GPT-5.4, Chat Completions does not support tool calling with `reasoning_effort` values
other than `none`."* AIAvatarKit is agent-native and defaults to `gpt-5.6-terra`, whose
reasoning default is `medium`. Registering a function tool therefore requires
`reasoning_effort="none"` explicitly; omitting the parameter still selects the model default
and the API rejects the request. If you want an avatar that both uses tools and thinks about
them, that combination only exists on the Responses API.

**Reasoning state survives the turn there, and is lost here.** Chat Completions is
stateless: you resend the message history each turn, and reasoning items cannot be sent
back, so whatever the model worked out last turn is discarded. The Responses API persists
them, letting the model *"continue its reasoning process to produce better results in the
most token-efficient manner"*. OpenAI puts it plainly: using reasoning models with Responses
*"will result in better model intelligence when compared to Chat Completions."*

**It is usually faster and cheaper.** Because history lives server-side and is referenced by
id rather than resent, prompt caching hits far more often — OpenAI reports a *"40% to 80%
improvement"* in cache utilisation against Chat Completions in internal tests. Less resent
context also means less time before the first token, which is the number a voice avatar
feels most.

Keep using `ChatGPTService` when you are talking to Azure OpenAI, to a compatible provider,
or to an OpenAI model old enough that none of the above applies.

All three quotations above are from OpenAI's
[migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses); the
cache figure is described there as coming from internal tests.

## Sampling parameters

`temperature` follows the same rule as `reasoning_effort`: it is sent only when it is not
`None`, and its default *is* `None`.

Sampling support is endpoint- and model-specific. OpenAI documents `temperature`, `top_p`,
and `logprobs` for GPT-5.4 only with reasoning effort set to `none`; GPT-5.6 instead defaults
to `medium`. AIAvatarKit deliberately does not infer a supported combination or rewrite one
parameter based on the other.

Add sampling parameters only after confirming that the selected endpoint, model, and
reasoning setting support them:

```python
# Example for a model that supports temperature with reasoning disabled
llm = ChatGPTService(openai_api_key=OPENAI_API_KEY, reasoning_effort="none", temperature=0.0)

# Commonly unsupported on reasoning models
llm = ChatGPTService(openai_api_key=OPENAI_API_KEY, reasoning_effort="low", temperature=0.0)
```

The safe library default is to leave `temperature=None`, which omits the field. If you need
thinking and a specific temperature, verify that combination for the exact model or drop the
temperature.

Watch for the same collision when parameters are injected later: `inline_llm_params` and
`edit_chat_completion_params` both add fields to the outgoing request, so a `temperature`
added there meets the same rule.

Other providers behind this class disagree about sampling in their own ways; Anthropic
clamps `temperature` to 0–1, for instance. See
[OpenAI-compatible APIs](llm-openai-compatible.md).

→ [GPT-5.6 guide (OpenAI)](https://developers.openai.com/api/docs/guides/latest-model)

## Extra body

Anything the provider accepts that has no dedicated argument goes in `extra_body`, which is
merged into the request when non-empty:

```python
llm = ChatGPTService(
    openai_api_key=ANTHROPIC_API_KEY,
    base_url="https://api.anthropic.com/v1/",
    model="claude-haiku-4-5",
    extra_body={"thinking": {"type": "disabled"}},
)
```

This is how providers that expose reasoning under a different name are configured. See
[OpenAI-compatible APIs](llm-openai-compatible.md) for what each one expects.

## Editing request parameters per call

For anything that has to be decided at request time, register a hook. It receives the
assembled parameter dict along with the current `context_id` and `user_id`, and **mutates the
dict in place** — the return value is discarded.

```python
@llm.edit_chat_completion_params
def edit_params(params: dict, context_id: str, user_id: str):
    if len(params["messages"]) > 40:
        params["model"] = "gpt-5.6-luna"        # Mutate; do not return
    if user_id in PREMIUM_USERS:
        params["reasoning_effort"] = "low"
```

All three arguments are positional, so the hook must accept them even if it ignores the
identifiers. Returning a new dict has no effect: assign onto `params` instead.

The hook runs last, after `inline_llm_params` has been merged, so it has the final say over
every field. That also means it can reintroduce a parameter the provider will reject — see
[Sampling parameters](#sampling-parameters).

`inline_llm_params` covers the simpler case of overriding parameters for one call — see
[LLM](llm.md#inline-llm-parameters).

## Azure OpenAI

Construct the official `AsyncAzureOpenAI` client and inject it. Provider selection,
authentication, API version, endpoint, retries, and custom HTTP settings then stay on the
client, while `model` has one meaning: the Azure deployment name sent with the request.

```python
import os
from openai import AsyncAzureOpenAI

azure_client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

llm = ChatGPTService(
    openai_client=azure_client,
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    system_prompt="You are my cat.",
)
```

The old configuration remains functional during the deprecation period:

```python
llm = ChatGPTService(
    openai_api_key=AZURE_OPENAI_API_KEY,
    base_url=(
        "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/"
        "YOUR_DEPLOYMENT?api-version=2024-10-21"
    ),
    model="azure",
)
```

This legacy path emits `DeprecationWarning`. It still selects `AsyncAzureOpenAI` whenever
`model` contains `azure` and parses `api-version` from `base_url`; new code must not depend
on either behavior. `custom_openai_module` is deprecated for the same reason: construct its
client explicitly and pass `openai_client`.

Azure content filtering surfaces as an API error with `code == "content_filter"`, which is
worth handling explicitly so the avatar says something sensible instead of failing silently.
See [LLM error handling](llm.md#llm-error-handling).

## Observability

Because the client instance is injectable, tracing configuration stays outside the LLM
service:

```sh
pip install langfuse
```

```python
from langfuse.openai import AsyncOpenAI

langfuse_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

llm = ChatGPTService(
    openai_client=langfuse_client,
    model="gpt-5.6-terra",
)
```

See [Administration](admin.md#observability).

## See also

- [LLM](llm.md) — the interface and its shared behaviour
- [OpenAI-compatible APIs](llm-openai-compatible.md) — Anthropic, Gemini, xAI, OpenRouter, LM Studio
- [Responses API](llm-responses.md) — OpenAI's stateful API and its WebSocket transport
- [Tools](tools.md) — tool calling, dynamic tools, and MCP

---

[← Documentation index](../README.md#-documentation)
