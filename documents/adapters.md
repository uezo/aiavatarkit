# Adapters

An adapter wraps the pipeline for one channel. It owns transport concerns only — accepting
audio, framing responses, authenticating callers — and delegates everything about the
conversation to `STSPipeline`.

Because the pipeline is a separate object, several adapters can attach to the same one. That
is how a single avatar answers on the web, on the phone, and on LINE without three copies of
your configuration.

| Channel | Class | Module | Default channel name |
| --- | --- | --- | --- |
| WebSocket | `AIAvatarWebSocketServer` | `aiavatar.adapter.websocket.server` | `websocket` |
| HTTP (SSE) | `AIAvatarHttpServer` | `aiavatar.adapter.http.server` | `http` |
| LINE Bot | `AIAvatarLineBotServer` | `aiavatar.adapter.linebot.server` | `linebot` |
| Twilio Voice | `AIAvatarTwilioServer` | `aiavatar.adapter.twilio.server` | `phone` |
| Twilio SMS | `AIAvatarTwilioSMSServer` | `aiavatar.adapter.twilio.server` | `sms` |
| Asterisk | `AIAvatarAsteriskServer` | `aiavatar.adapter.asterisk.server` | `phone` |
| OpenAI-compatible endpoint | `AIAvatarChatCompletionsServer` | `aiavatar.adapter.chatcompletions.server` | `chatcompletions` |
| Speech recognition only | `StreamSpeechRecognitionServer` | `aiavatar.adapter.stt.server` | — |

Two adapters need a security decision before they are exposed. The
[Twilio routers](adapters-twilio.md#-protect-these-routers-before-exposing-them) publish
unauthenticated endpoints that place calls and send SMS, and the
[LINE Bot router](adapters-linebot.md#protecting-the-management-endpoints) leaves its
management endpoints open unless you set `api_key`.

Each has its own page: [WebSocket](adapters-websocket.md), [HTTP](adapters-http.md),
[LINE Bot](adapters-linebot.md),
[Twilio](adapters-twilio.md), [Asterisk](adapters-asterisk.md),
[OpenAI-compatible endpoint](adapters-chatcompletions.md), and
[Speech recognition server](adapters-stt-server.md).

A channel adapter connects a client or an external messaging service to an `STSPipeline`. An adapter can create its own pipeline from its convenience parameters, or attach to an existing pipeline through the `sts` parameter. Attaching multiple adapters to one pipeline lets all channels share the same VAD, STT, LLM, TTS, conversation store, and pipeline hooks.

Each adapter registers itself as a response handler when it is created. No additional registration is required. A `session_id` must identify only one active adapter session within a shared pipeline so that responses can be routed to the correct channel.

## Attaching an adapter to a pipeline

The following examples assume that `app` is a FastAPI application and `sts` is an existing `STSPipeline`:

```python
import os
from fastapi import FastAPI
from aiavatar.sts import STSPipeline
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

app = FastAPI()
sts = STSPipeline(
    stt=OpenAISpeechRecognizer(
        openai_api_key=os.environ["OPENAI_API_KEY"],
    ),
    llm_openai_api_key=os.environ["OPENAI_API_KEY"],
    llm_system_prompt="You are a helpful assistant.",
)
```

When `sts` is supplied to an adapter, configure pipeline-level behavior on that shared `STSPipeline`. The adapter's other arguments configure only its transport and channel-specific behavior.

## Connecting Multiple Channels

Create the pipeline once, then pass it to every additional adapter. For example, the following application creates the pipeline through the WebSocket adapter and then attaches a LINE Bot adapter to it:

```python
import os
from fastapi import FastAPI
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer

app = FastAPI()

# The first adapter creates and owns the shared pipeline configuration.
websocket_adapter = AIAvatarWebSocketServer(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    system_prompt="You are a helpful assistant.",
    channel="websocket",
)

# The second adapter attaches to the exact same pipeline instance.
line_adapter = AIAvatarLineBotServer(
    sts=websocket_adapter.sts,
    channel_access_token=os.environ["LINEBOT_CHANNEL_ACCESS_TOKEN"],
    channel_secret=os.environ["LINEBOT_CHANNEL_SECRET"],
    api_key=os.environ["LINEBOT_ADMIN_API_KEY"],
    channel="linebot",
)

app.include_router(websocket_adapter.get_websocket_router(path="/ws"))
app.include_router(line_adapter.get_api_router(), prefix="/line")
```

This shares pipeline components and conversation storage, but it does not by itself establish that a WebSocket user and a LINE user are the same person. Use a channel context bridge when conversation continuity must follow a user across channels.

## Sharing Context Across Channels

`ChannelContextBridge` maps each `(channel_id, channel_user_id)` pair to an application-level `user_id`, then stores the latest `context_id` for that application user. Adapters mapped to the same application user therefore resume the same conversation until the bridge timeout expires.

Use one bridge instance for all adapters. LINE Bot and Chat Completions use their supplied bridge internally. Bind the bridge to the WebSocket adapter:

```python
import os
from aiavatar.adapter.channel_context_bridge import SQLiteChannelContextBridge
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer

bridge = SQLiteChannelContextBridge(
    db_path="channel_context_bridge.db",
    timeout=3600,
)

websocket_adapter = AIAvatarWebSocketServer(
    sts=sts,
    channel="websocket",
)
bridge.bind(websocket_adapter, channel_id="websocket")

line_adapter = AIAvatarLineBotServer(
    sts=sts,
    channel_access_token=os.environ["LINEBOT_CHANNEL_ACCESS_TOKEN"],
    channel_secret=os.environ["LINEBOT_CHANNEL_SECRET"],
    api_key=os.environ["LINEBOT_ADMIN_API_KEY"],
    channel="linebot",
    channel_context_bridge=bridge,
)
```

The bridge's `channel_id` is the identity namespace used to look up a channel user. It should normally match the adapter's actual channel name—for example, bind an adapter created with `channel="websocket_m5"` using `channel_id="websocket_m5"`. A renamed tag such as `"desktop_robot"` in `insert_channel_tag` is only an LLM-facing label and is not used by the bridge.

By default, an automatically created mapping uses the channel user ID as the application user ID. If both channels provide the same stable user ID, they therefore share context without an explicit link. When the channel user IDs differ, link both identities to one application user from a trusted account-linking or startup flow:

```python
await bridge.link_channel_user(
    channel_id="websocket",
    channel_user_id="web-user-123",
    user_id="user-123",
)
await bridge.link_channel_user(
    channel_id="linebot",
    channel_user_id="U0123456789abcdef",
    user_id="user-123",
)
```

The WebSocket client should continue to send its channel-specific ID (`"web-user-123"`) as `user_id`; the bridge resolves it to `"user-123"` before the request reaches the pipeline. After `timeout` seconds without a context update, a request that does not supply its own `context_id` starts a new context.

For PostgreSQL storage and custom user ID generation, see [Database](database.md).

## Channel-aware Processing

Every adapter assigns a channel to its requests. The adapters described above default to `"websocket"`, `"http"`, `"phone"`, `"sms"`, `"linebot"`, and `"chatcompletions"`. Override the adapter's `channel` (or `channel_id` for Chat Completions) when the application needs a more specific name, such as `"websocket_m5"`.

Set `insert_channel_tag` on the shared pipeline to expose that channel to the LLM:

```python
sts.insert_channel_tag = [
    "phone",                                  # Keep the channel name.
    "sms",
    ("websocket_m5", "desktop_robot"),       # Rename it for the LLM.
]
```

The pipeline then transforms requests as follows before invoking the LLM:

```text
# phone
<channel name='phone' />Hello

# websocket_m5
<channel name='desktop_robot' />Hello
```

Channels not included in the list receive no tag. Set `insert_channel_tag=True` to insert every request's channel without filtering, or `False` to disable insertion.

Describe the desired behavior in the system prompt, for example:

```text
When <channel name='sms' /> is present, respond briefly without speech-oriented phrasing.
When <channel name='phone' /> is present, use natural spoken language.
When <channel name='desktop_robot' /> is present, you may refer to the robot's body and surroundings.
```

The LINE Bot, Twilio SMS, and Chat Completions adapters automatically add their channel names to the shared pipeline's `skip_tts_channels`. For another text-only adapter, add its channel explicitly:

```python
sts.skip_tts_channels.append("your_text_channel")
```



## See also

- [Pipeline](pipeline.md) — what every adapter shares
- [Avatar control](avatar.md) — the control tags adapters resolve
- [Database](database.md) — the channel context bridge

---

[← Documentation index](../README.md#-documentation)
