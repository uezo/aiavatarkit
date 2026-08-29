# OpenAI-compatible endpoint

`AIAvatarChatCompletionsServer` exposes the pipeline behind an OpenAI-compatible Chat
Completions endpoint, so any client that already speaks that API can talk to your avatar.
This adapter is experimental.

```sh
pip install sse-starlette
```

`AIAvatarChatCompletionsServer` exposes an experimental OpenAI-compatible Chat Completions endpoint. Its default channel ID is `"chatcompletions"`.

```python
from aiavatar.adapter.chatcompletions.server import AIAvatarChatCompletionsServer

chat_completions_adapter = AIAvatarChatCompletionsServer(
    sts=sts,
    channel_id="chatcompletions",
)
app.include_router(chat_completions_adapter.get_api_router())
```

## The bearer token is an identifier, not authentication

Every request must carry a bearer token, but **the adapter does not authenticate it**. Its
check is only that a non-empty `Bearer` credential is present — it is never compared against
a configured key or a signature. The string is then used directly as the channel-specific
user key for context mapping.

So the token is an opaque user ID that the caller chooses, not a credential. Anyone who can
reach the endpoint can invent one and start a conversation; supplying a different token
simply starts a different conversation.

**This endpoint must sit behind your own authentication.** Put a reverse proxy, an API
gateway, or a FastAPI dependency in front of it, and treat the adapter's token purely as
identity once the caller has already been authorised.

When choosing what to put in the token:

- Use a **stable** value per user, so their conversation continues across requests.
- Never share one token between users — they would share a conversation context.
- Never reuse a secret or a shared API key as the token. It is an identifier, it is stored
  and logged as one, and it grants nothing on its own.

## See also

- [Adapters](adapters.md) — choosing a channel
- [HTTP (SSE) adapter](adapters-http.md) — the native streaming API

---

[← Documentation index](../README.md#-documentation)
