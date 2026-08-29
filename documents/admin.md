# Administration

The Admin Panel is a FastAPI-embedded interface for monitoring a running AIAvatarKit
application and changing its configuration without restarting it. `setup_admin_panel()`
installs the UI, its static assets, and the management API under `/admin`.

Configuration changes apply to the live pipeline, its components, and the adapter. They are
deliberately volatile — discarded when the process exits — because component composition
belongs in your Python code, not in a database somewhere.

AIAvatarKit provides a built-in admin panel for monitoring, configuring, and evaluating your AI avatar from a web browser.

## Admin Panel

Set up the Admin Panel with a single function call. Once configured, access it at `/admin/` on your server.

```python
import os

from aiavatar.admin import BasicAdminAuthenticator, setup_admin_panel

setup_admin_panel(
    app,
    adapter=aiavatar_app,
    authenticator=BasicAdminAuthenticator(
        os.environ["ADMIN_USERNAME"],
        os.environ["ADMIN_PASSWORD"],
    ),
)
```

The Admin Panel includes:

- **Metrics** — First-response statistics and a detailed latency breakdown measured from the end of the user's speech
- **Logs** — Searchable conversation messages grouped by context, with session filtering, voice playback, and per-turn timing details
- **Config** — Adjust pipeline, VAD, STT, LLM, TTS, and adapter settings at runtime
- **Evaluation** — Run dialog evaluation scenarios when an evaluator is available
- **Light/Dark themes** — Follow the operating system theme or switch it manually

Evaluation is configured automatically when the pipeline uses `ChatGPTService`. For other LLM services, pass a `DialogEvaluator` through the optional `evaluator` argument.

The same authenticator protects the HTML, static assets, and `/admin/api` endpoints. The frontend does not use a separate API key. In addition to `BasicAdminAuthenticator`, `authenticator` accepts any synchronous or asynchronous callable that receives a FastAPI `Request`, allowing integration with an SSO session or an authenticated reverse proxy.

Passing `authenticator=None` disables authentication and should be limited to local development. Use HTTPS when using Basic authentication in production.

Character and Control features are not part of the new Admin Panel. The previous UI and APIs remain available as an independent legacy package when an existing application still needs them:

```python
from aiavatar.admin_legacy import setup_admin_panel
```

See the [Admin Panel documentation](../aiavatar/admin/README.md) for authentication examples, screen and API specifications, component responsibilities, time semantics, and frontend development instructions.

## REST API

Admin Panel operations are available under `/admin/api` and use the same authentication as the UI. See the interactive API documentation at `/docs` for request and response schemas, or the [Admin Panel API summary](../aiavatar/admin/README.md#api) for an overview.

## Observability

You can monitor the entire sequence - what requests are sent to the LLM, how they are interpreted, which tools are invoked, and what responses are generated from specific results or data - to support AIAvatar quality improvements and governance.

AIAvatarKit accepts a pre-configured OpenAI-compatible client, so you can construct
a [Langfuse](https://langfuse.com) wrapper first and inject that instance.

```sh
pip install langfuse
```

```sh
export LANGFUSE_SECRET_KEY=sk-lf-XXXXXXXX
export LANGFUSE_PUBLIC_KEY=pk-lf-XXXXXXXX
export LANGFUSE_BASE_URL=http://localhost:3000
```

```python
from langfuse.openai import AsyncOpenAI

langfuse_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
llm = ChatGPTService(
    openai_client=langfuse_client,
    system_prompt="You are a helpful assistant.",
)
```

The same pattern applies to `OpenAIResponsesService`. The Responses WebSocket
implementation speaks the event protocol directly and is not traced by this HTTP client.

## See also

- [Pipeline](pipeline.md) — the performance records the panel displays
- [Evaluation](evaluation.md) — scoring conversations
- [Guardrail](guardrail.md) — constraining what gets said

---

[← Documentation index](../README.md#-documentation)
