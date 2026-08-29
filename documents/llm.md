# LLM

`LLMService` is the interface every model provider implements. It hides the differences that
would otherwise leak into your application: tool specs are converted per provider, streaming
is normalised into the same chunk shape, and conversation context is managed identically
everywhere. A tool you define once works on GPT, Gemini, and Claude alike.

## Choosing a service

| Provider | Class | Guide | Extra install |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `ChatGPTService` | [Chat Completions](llm-chat-completions.md) | — |
| Azure OpenAI | `ChatGPTService` | [Chat Completions](llm-chat-completions.md#azure-openai) | — |
| OpenAI Responses API | `OpenAIResponsesService` | [Responses API](llm-responses.md) | — |
| OpenAI Responses over WebSocket | `OpenAIResponsesWebSocketService` | [Responses API](llm-responses.md#websocket-transport) | — |
| Anthropic Claude | `ClaudeService` | [Claude](llm-claude.md) | `anthropic` |
| Google Gemini | `GeminiService` | [Gemini](llm-gemini.md) | `google-genai` |
| Dify | `DifyService` | [Dify](llm-dify.md) | — |
| LiteLLM | `LiteLLMService` | [LiteLLM](llm-litellm.md) | `litellm` |
| Anthropic, Gemini, xAI, OpenRouter, LM Studio | `ChatGPTService` with `base_url` | [OpenAI-compatible APIs](llm-openai-compatible.md) | — |

**Using OpenAI?** Prefer [`OpenAIResponsesService`](llm-responses.md) over
`ChatGPTService`. It is faster, it keeps reasoning state between turns, and it is the only
one of the two that can use tools and reasoning together on current models.

**Native class or compatible endpoint?** Several providers appear twice. Prefer the native
class — `ClaudeService`, `GeminiService` — when you want that provider's own features:
Claude's thinking blocks, Gemini's thinking budget, provider-specific tool semantics. Reach
for `ChatGPTService` with a `base_url` when you want one code path across many providers, or
when the provider only offers an OpenAI-shaped endpoint. The compatibility layers are
lossy in ways worth knowing about before you commit — see
[OpenAI-compatible APIs](llm-openai-compatible.md).

## Setting a service on the pipeline

Build the service, then hand it to the adapter or the pipeline:

```python
from aiavatar.sts.llm.claude import ClaudeService

llm = ClaudeService(
    anthropic_api_key=ANTHROPIC_API_KEY,
    system_prompt="You are my cat.",
)

aiavatar_app = AIAvatarWebSocketServer(
    llm=llm,
    openai_api_key=OPENAI_API_KEY,   # still used for STT
)
```

For the simplest OpenAI setup you can skip the service entirely and pass
`openai_api_key`, `llm_model`, and `system_prompt` straight to the adapter.

## Parameters that vary by provider

The interface is uniform; the providers behind it are not. These are the arguments that mean
different things — or fail differently — depending on who you point at.

| Parameter | What varies | Where to look |
| --- | --- | --- |
| `reasoning_effort` | Accepted, silently ignored, or rejected with HTTP 400 depending on the provider and the model | [Chat Completions](llm-chat-completions.md#reasoning-effort), [OpenAI-compatible](llm-openai-compatible.md) |
| `reasoning_effort` **with tools** | On OpenAI Chat Completions from GPT-5.4, only `none` is supported; GPT-5.6 defaults to `medium`, so tools require an explicit `none` | [Responses API](llm-responses.md) |
| `temperature` | Support varies by endpoint, model, and reasoning setting; clamped to 0–1 by Anthropic; model-dependent elsewhere | [Chat Completions](llm-chat-completions.md#sampling-parameters) |
| `max_tokens` | Exposed and defaulted to `1024` on Claude; not exposed at all on the OpenAI services | [Claude](llm-claude.md) |
| Thinking configuration | `reasoning_effort`, Anthropic's `thinking` object, Gemini's doubly-nested `thinking_config`, or the `thinking_level` / `thinking_budget` arguments | [Gemini](llm-gemini.md), [OpenAI-compatible](llm-openai-compatible.md) |
| System prompt handling | Anthropic hoists and concatenates all system messages; some models have no system role at all | [OpenAI-compatible](llm-openai-compatible.md#anthropic-claude), [LiteLLM](llm-litellm.md) |
| Tool schema strictness | `strict` is honoured by OpenAI and ignored by Anthropic's compatibility layer | [OpenAI-compatible](llm-openai-compatible.md#anthropic-claude) |
| Dynamic tool calls | Available on Chat Completions, Claude, Gemini, and LiteLLM; the Responses services leave tool selection to the model | [Tools](tools.md#dynamic-tool-call) |

Two behaviours make this manageable. AIAvatarKit **omits `reasoning_effort` and
`temperature` entirely when they are `None`**, and both default to `None` on the OpenAI
services. That avoids inventing provider rules, but omission is not a promise that every
model-and-tool combination will work: GPT-5.6 defaults to `medium`, so Chat Completions with
function tools needs `reasoning_effort="none"` explicitly. `extra_body` is merged only when
non-empty, so provider-specific settings stay out of requests to providers that would reject
them.

The practical rule when moving to an unfamiliar provider: start with nothing set, confirm a
plain turn works, then add one parameter at a time.

## Shared behaviour

Everything below applies to every service, whichever provider is behind it.

### Voice Text Tag (Think Before Answering)

By setting `voice_text_tag`, you can have the LLM "think before answering" (Chain-of-Thought) while vocalizing only the answer portion. You can specify a single tag or a list of tags.

```python
# Single tag: vocalize only <answer> content
llm = ChatGPTService(
    system_prompt="Think within <think> tags. Write your answer within <answer> tags.",
    voice_text_tag="answer"
)

# Multiple tags: vocalize both <ack> and <answer>, skip <think>
llm = ChatGPTService(
    system_prompt="Output <ack>first reaction</ack><think>reasoning</think><answer>full response</answer>",
    voice_text_tag=["ack", "answer"]
)
```

### LLM Error Handling

You can handle errors that occur during LLM API calls by using the `on_error` decorator. This is useful for customizing avatar responses when content filters are triggered or when API errors occur.

```python
from aiavatar.sts.llm import LLMResponse

@llm.on_error
async def on_error(llm_response: LLMResponse):
    ex = llm_response.error_info.get("exception")   # Get exception
    error_json = llm_response.error_info.get("response_json", {})   # Get response JSON from OpenAI

    # Make response
    if error_json.get("error", {}).get("code") == "content_filter":
        llm_response.text = '<face name="angry" />You shouldn\'t say that!'
        llm_response.voice_text = "You shouldn't say that!"
    else:
        llm_response.text = '<face name="sorrow" />An error occurred'
        llm_response.voice_text = "An error occurred"
```

**NOTE**: When an error occurs, the conversation context is not updated. This is intentional because including the programmatically overwritten response in the context may cause unexpected LLM behavior in subsequent conversations.

### Custom Chat Logging

Use the `print_chat` decorator to customize how user/AI conversation turns are logged.

```python
@llm.print_chat
def print_chat(role, context_id, user_id, text, files):
    if role == "user":
        logger.info(f"\033[1;32mUser:\033[0m {text}")
    else:
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if think_match or answer_match:
            if think_match:
                logger.info(f"\033[3;38;5;246mThinking: {think_match.group(1).strip()}\033[0m")
            logger.info(f"\033[1;35mAI:\033[0m {answer_match.group(1).strip() if answer_match else text}")
        else:
            logger.info(f"\033[1;35mAI:\033[0m {text}")
```

**NOTE**: This example uses ANSI escape sequences optimized for console output. These escape codes will appear as noise in log files.

### System Prompt Parameters

You can embed parameters into your system prompt dynamically.

First, define the adapter with a system prompt containing placeholders:

```python
aiavatar_app = AIAvatarWebSocketServer(
    openai_api_key="YOUR_OPENAI_API_KEY",
    system_prompt="User's name is {name}."
)
```

Then pass the values with each request as `system_prompt_params`. `invoke()` is an async
generator, so it has to be iterated — calling it alone does nothing:

```python
from aiavatar.sts.models import STSRequest

async for response in aiavatar_app.sts.invoke(STSRequest(
    # (other fields omitted)
    system_prompt_params={"name": "Nekochan"}
)):
    ...
```

In a normal application the adapter iterates the stream for you; you only call `invoke()`
directly when driving the pipeline yourself.

Placeholders in the system prompt, such as `{name}`, will be replaced with the corresponding values at runtime.

### Inline LLM Parameters

When calling `LLMService.chat_stream` directly (outside the Speech-to-Speech pipeline), you can override model-specific parameters on a per-request basis using `inline_llm_params`.

```python
# Override provider-supported generation parameters for a single call
async for chunk in llm.chat_stream(
    context_id="ctx_001",
    user_id="user_001",
    text="Hello!",
    inline_llm_params={"reasoning_effort": "none", "temperature": 0.0}
):
    print(chunk.text, end="", flush=True)
```

The key-value pairs in `inline_llm_params` are merged into the underlying API call parameters, so any parameter accepted by the provider's API can be specified. AIAvatarKit does not validate combinations such as `temperature` plus `reasoning_effort`; the selected endpoint and model must support them. The exact keys depend on the LLM service:

| Service | Example keys |
|---|---|
| ChatGPTService | `model`, `temperature`, `reasoning_effort`, ... |
| ClaudeService | `model`, `temperature`, `max_tokens`, ... |
| GeminiService | `model`, `config`, ... |
| LiteLLMService | `model`, `temperature`, ... |

For a practical example, see [Quick Response](pipeline.md#quick-response) — `QuickResponder` uses `inline_llm_params` to disable tool calls and reasoning for fast first-response generation.

## See also

- [Tools](tools.md) — giving the model something to do
- [Guardrail](guardrail.md) — blocking and correcting responses
- [Pipeline](pipeline.md) — where generation sits in a turn
- [Database](database.md) — where conversation context is stored
- [Administration](admin.md) — tracing requests with Langfuse

---

[← Documentation index](../README.md#-documentation)
