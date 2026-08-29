# OpenAI-compatible APIs

Most providers now expose a Chat Completions-shaped endpoint, so `ChatGPTService` reaches
them all with nothing but a `base_url`, an API key, and a model name.

```python
from aiavatar.sts.llm.chatgpt import ChatGPTService

llm = ChatGPTService(
    openai_api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
)
```

The part that is not uniform is reasoning. Every provider has landed on a different way to
say "think less" or "do not think", and the disagreement is not cosmetic — some ignore
`reasoning_effort`, and at least one rejects the request outright. For a voice avatar this
matters more than usual, because a thinking pass is time the user spends listening to
silence.

| Provider | `base_url` | Reasoning control | `temperature` |
| --- | --- | --- | --- |
| [Anthropic Claude](#anthropic-claude) | `https://api.anthropic.com/v1/` | `reasoning_effort` **ignored**; use `extra_body={"thinking": ...}` | Clamped to 0–1 |
| [Google Gemini](#google-gemini) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `reasoning_effort`, **or** `extra_body={"thinking_config": ...}` — not both | Supported |
| [xAI Grok](#xai-grok) | `https://api.x.ai/v1` | `reasoning_effort`; **rejected by models that do not support it** | Model-dependent |
| [OpenRouter](#openrouter) | `https://openrouter.ai/api/v1` | `reasoning_effort`, or the richer `reasoning` map | Depends on the slug |
| [LM Studio](#lm-studio) | `http://localhost:1234/v1` | `extra_body={"reasoning": {"effort": ...}}` on models that support it | Depends on the model |

Remember that AIAvatarKit only sends `reasoning_effort` when it is not `None`, and only
merges `extra_body` when it is non-empty. Leaving both unset sends neither, which is the
safe starting point on an unfamiliar provider.

`temperature` behaves the same way — sent only when it is not `None`. That default matters,
because sampling parameters are the second thing providers disagree about. Support depends
on the endpoint, model, and reasoning setting; Anthropic silently clamps the value into 0–1,
and on aggregators the answer depends entirely on which model the slug resolves to. On an
unfamiliar provider, send nothing and add parameters one at a time.

When a provider has a native service class in AIAvatarKit, prefer it. The compatibility
layers below are convenient, not complete.

## Anthropic Claude

```python
llm = ChatGPTService(
    openai_api_key=ANTHROPIC_API_KEY,
    base_url="https://api.anthropic.com/v1/",
    model="claude-haiku-4-5",
    extra_body={"thinking": {"type": "disabled"}},
)
```

**`reasoning_effort` is ignored.** Anthropic documents it as an ignored field on this
endpoint, so setting it changes nothing — it does not error, it simply has no effect.
Thinking is configured through the native `thinking` object instead, passed via
`extra_body`: `{"type": "disabled"}` to turn it off, or
`{"type": "enabled", "budget_tokens": 2000}` to bound it.

Other differences worth knowing before you ship this:

- Anthropic describes the compatibility layer as intended for testing and comparing model
  capabilities, **not as a production solution**. Use [`ClaudeService`](llm-claude.md) for
  real deployments.
- Claude's reasoning output is not returned through this endpoint at all, even when thinking
  is enabled.
- Multiple system messages are hoisted to the front and concatenated with newlines, because
  Anthropic accepts only one initial system message.
- `strict` on tool definitions is ignored, so tool arguments are not guaranteed to match
  your schema. Validate them yourself, or use `ClaudeService`.
- `response_format`, `seed`, `logprobs`, `presence_penalty`, and `frequency_penalty` are all
  ignored, and `n` must be 1.
- `temperature` is silently clamped into 0–1 rather than rejected, so a value tuned for
  OpenAI's 0–2 range quietly means something different here.
- Prompt caching is unavailable here.

→ [OpenAI SDK compatibility (Anthropic)](https://platform.claude.com/docs/en/api/openai-sdk)

## Google Gemini

```python
llm = ChatGPTService(
    openai_api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.5-flash",
    reasoning_effort="none",
)
```

`reasoning_effort` works here and maps onto Gemini's internal thinking levels:
`"low"`, `"medium"`, `"high"`, and `"none"` — but `"none"` applies to 2.5 models only.
**Reasoning cannot be disabled on Gemini 2.5 Pro or on Gemini 3 models**, so plan for a
thinking pass in your latency budget when using them.

For finer control there is `thinking_config`, carrying `thinking_level` and
`include_thoughts`. Google's own example nests it two levels deep, under an inner
`extra_body` and a `google` namespace — pass exactly that shape:

```python
llm = ChatGPTService(
    openai_api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.5-flash",
    extra_body={
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinking_level": "low",
                    "include_thoughts": False,
                }
            }
        }
    },
)
```

The repetition is not a typo. The outer `extra_body` is AIAvatarKit's argument, which becomes
the OpenAI SDK's `extra_body` and is merged into the request body; the inner one is the field
Gemini actually reads from that body. Flattening it — putting `thinking_config` at the top —
sends a field Gemini ignores, and the setting silently does nothing.

**Do not set both.** The two mechanisms overlap and are not meant to be combined. Since
AIAvatarKit sends `reasoning_effort` whenever it is not `None`, leave it unset when you use
`extra_body`.

Google labels the OpenAI compatibility layer as beta while feature support is extended.
[`GeminiService`](llm-gemini.md) is the fuller path, with `thinking_level` and
`thinking_budget` as first-class arguments.

→ [OpenAI compatibility (Gemini API)](https://ai.google.dev/gemini-api/docs/openai)

## xAI Grok

```python
llm = ChatGPTService(
    openai_api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    model="grok-4.6",
    reasoning_effort="low",
)
```

xAI accepts `reasoning_effort` with `"low"`, `"medium"`, `"high"`, and `"xhigh"`. `"high"`
is the default. `"xhigh"` needs `grok-4.6` or later; `grok-4.5` accepts it but treats it as
`"high"`.

**This is the provider most likely to hard-fail.** Models that do not support reasoning
effort return HTTP 400 rather than ignoring the parameter — reports describe
`grok-4-1-fast` and its non-reasoning variant rejecting it with a "does not support
parameter" error, with support arriving from Grok 4.3 onward. Note also that xAI's
documented values do not include `"none"`.

Sampling parameters are similarly model-dependent on xAI; tooling in the ecosystem strips
unsupported sampling and reasoning-effort parameters per model rather than assuming they
are accepted. Treat `temperature` as something to verify against the specific Grok model,
not as a given.

Two consequences for AIAvatarKit:

- On a non-reasoning Grok model, leave `reasoning_effort` unset. It is `None` by default, so
  this means simply not passing it.
- From the CLI, the built-in application defaults to `reasoning_effort="none"`, which xAI
  will not accept. Set `AIAVATAR_LLM_REASONING_EFFORT=omit` to suppress the parameter, or
  give it a value xAI supports. See
  [Getting started](getting-started.md#openai-and-llm-configuration).

→ [Reasoning (xAI docs)](https://docs.x.ai/docs/guides/reasoning)

## OpenRouter

```python
llm = ChatGPTService(
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model=OPENROUTER_MODEL,
    reasoning_effort="none",
)
```

One key reaches every model OpenRouter fronts, which makes it the quickest way to try a
model you do not otherwise have an account for. Model IDs are `provider/model` slugs;
aliases such as `~openai/gpt-latest` resolve to whatever that vendor's current flagship is.

OpenRouter accepts `reasoning_effort` in OpenAI style, from `"xhigh"` down to `"none"`, and
also a richer `reasoning` map for models with thinking tokens:

```python
llm = ChatGPTService(
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-haiku-4-5",
    extra_body={"reasoning": {"enabled": False}},
)
```

Because a slug can point at any vendor, what a reasoning setting actually does varies with
the model behind it. Treat reasoning behaviour as a property of the chosen slug, not of
OpenRouter.

Optional attribution headers (`HTTP-Referer`, `X-OpenRouter-Title`) let your application
appear on OpenRouter's leaderboards. They are not required; supply them through a custom
client module if you want them.

→ [OpenRouter API reference](https://openrouter.ai/docs/api-reference/parameters)

## LM Studio

```python
llm = ChatGPTService(
    openai_api_key="lm-studio",   # Any non-empty string
    base_url="http://localhost:1234/v1",
    model="openai/gpt-oss-20b",
)
```

LM Studio serves local models behind an OpenAI-compatible endpoint on port 1234 by default.
It runs without authentication, but the OpenAI client requires *some* key, so pass any
placeholder string.

Reasoning, where the model supports it, is configured with a `reasoning` object rather than
`reasoning_effort`:

```python
llm = ChatGPTService(
    openai_api_key="lm-studio",
    base_url="http://localhost:1234/v1",
    model="openai/gpt-oss-20b",
    extra_body={"reasoning": {"effort": "low"}},
)
```

Streaming and tool calling both work over `/v1/chat/completions`, which is what the pipeline
needs. Beyond that, LM Studio's compatibility is close but not identical in every edge case,
and behaviour depends as much on the loaded model as on the server — verify tool calling
with your actual model before relying on it.

Local models also change the latency picture completely. A first token that arrives in 200 ms
on a hosted API may take considerably longer on consumer hardware, so measure before
assuming the pipeline's timing defaults still fit. See [Pipeline](pipeline.md).

→ [OpenAI compatibility API (LM Studio)](https://lmstudio.ai/docs/app/api/endpoints/openai)

## Choosing between compatible and native

| You want | Use |
| --- | --- |
| One code path across many providers | `ChatGPTService` with `base_url` |
| To try a model without a new account | [OpenRouter](#openrouter) |
| Claude's thinking, prompt caching, strict tool schemas | [`ClaudeService`](llm-claude.md) |
| Gemini's thinking level and budget as real arguments | [`GeminiService`](llm-gemini.md) |
| A provider none of the above covers | [`LiteLLMService`](llm-litellm.md) |

## See also

- [Chat Completions](llm-chat-completions.md) — the class these all use, and Azure OpenAI
- [LLM](llm.md) — the interface and its shared behaviour
- [Getting started](getting-started.md) — pointing the built-in application at a provider

---

[← Documentation index](../README.md#-documentation)
