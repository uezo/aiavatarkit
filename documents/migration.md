# Migration guide

Breaking changes between versions, and what to do about them.

In version **v0.7.0**, the internal Speech-to-Speech pipeline previously provided by the external `LiteSTS` library has been fully integrated into AIAvatarKit.

## What Changed?

- The functionality remains the same — **no API behavior changes**.
- However, **import paths have been updated**.

## Required Changes

All imports from `litests` should now be updated to `aiavatar.sts`.

For example:

```python
# Before
from litests import STSRequest, STSResponse
from litests.llm.chatgpt import ChatGPTService

# After
from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.llm.chatgpt import ChatGPTService
```

This change ensures compatibility with the new internal structure and removes the need for `LiteSTS` as a separate dependency.

## OpenAI client injection

`ChatGPTService` and `OpenAIResponsesService` now accept a pre-configured async client
through `openai_client`. Prefer this for Azure OpenAI, Langfuse, custom HTTP clients,
token providers, retry settings, and default headers.

```python
# Before: the model string selected Azure implicitly.
llm = ChatGPTService(
    openai_api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_BASE_URL,
    model="azure",
)

# After: client type/configuration selects the provider; model is the deployment.
from openai import AsyncAzureOpenAI

client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)
llm = ChatGPTService(
    openai_client=client,
    model=AZURE_OPENAI_DEPLOYMENT,
)
```

The old model-based Azure selection and `custom_openai_module` remain functional but emit
`DeprecationWarning`. Injected clients are caller-owned; close them in the application
shutdown path. The Responses WebSocket service continues to use `openai_api_key`, `ws_url`,
and `model` directly because it speaks the WebSocket event protocol rather than using the
OpenAI HTTP client.

## Silero VAD ONNX models

`SileroSpeechDetector` and `SileroStreamSpeechDetector` now use ONNX Runtime on
CPU. ONNX Runtime is included in the base package dependencies, and the default
model requires no configuration changes.

If you set `model_path`, replace the custom `.jit` model with a compatible
Silero `.onnx` model, or omit the option to use the default model. JIT files are
no longer accepted. For example, with a local hub directory:

```python
from aiavatar.sts.vad.silero import SileroSpeechDetector

vad = SileroSpeechDetector(
    hub_cache_path="/models/silero-vad",
    model_path="/models/custom_silero_vad.onnx",
)
```

The hub directory must include its bundled ONNX model and compatible utilities,
even when a custom model is supplied. See [Local models](vad.md#local-models)
for the loading behavior. Probability values can differ between JIT and ONNX;
check any tuned speech probability threshold with representative audio.

The default `model_pool_size=1` supports multiple sessions with separate
inference histories. Existing pool settings remain supported.

## See also

- [Getting started](getting-started.md) — installation and the current setup
- [Adapters](adapters.md) — the adapter layer introduced in v0.7

---

[← Documentation index](../README.md#-documentation)
