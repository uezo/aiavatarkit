# LINE Bot adapter

`AIAvatarLineBotServer` receives LINE Messaging API webhooks and replies through the same
pipeline as every other channel. Because LINE identifies users stably, it pairs well with
the channel context bridge: the same person continues the conversation they started
elsewhere.

```sh
pip install line-bot-sdk
```

You can build a LINE Bot using the LINE Messaging API.

```python
# NOTE: Register https://{your.domain}/webhook as the "Webhook URL" in LINE Developers Console

# Create LINE Bot adapter
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer
aiavatar_app = AIAvatarLineBotServer(
    system_prompt="You are a cat.",
    openai_api_key=OPENAI_API_KEY,
    channel_access_token=LINEBOT_CHANNEL_ACCESS_TOKEN,
    channel_secret=LINEBOT_CHANNEL_SECRET,
    api_key=LINEBOT_ADMIN_API_KEY,
    image_download_url_base="https://{your.domain}",
    debug=True
)

# Create FastAPI app
from fastapi import FastAPI
app = FastAPI()

# Set adapter endpoints
router = aiavatar_app.get_api_router()
app.include_router(router)
```

Note: `image_download_url_base` is optional. If omitted, images from users are embedded as base64 data URLs directly in the LLM context, eliminating the need for the LLM to fetch images from an external URL.

By default, the LINE Messaging API user ID is used as the AIAvatarKit user ID. To map channel user IDs to your own app-level user IDs, use `ChannelContextBridge`. See [Adapters](adapters.md#sharing-context-across-channels) for details.

Customization hooks:

```python
@aiavatar_app.preprocess_request
async def preprocess_request(request: STSRequest):
    # Pre-process request before sending to LLM
    # e.g. edit request text
    request.text = "Pre-processed: " + request.text

@aiavatar_app.preprocess_response
async def preprocess_response(response: STSResponse):
    # Pre-process response before sending to LINE API
    # e.g. edit response voice_text (not text)
    response.voice_text = "Pre-processed: " + response.voice_text

@aiavatar_app.process_avatar_control_request
async def process_avatar_control_request(avatar_control_request: AvatarControlRequest, reply_message_request: ReplyMessageRequest):
    # Process facial expression
    # e.g. set `sender` to the message in reply_message_request to change icon
    face = avatar_control_request.face_name
    if face:
        reply_message_request.messages[0].sender = Sender(iconUrl=f"https://your_domain/path/to/icon/{face}.png")

@aiavatar_app.on_send_error_message
async def on_send_error_message(reply_message_request: ReplyMessageRequest, event: Event, ex: Exception):
    # Pre-process error message
    # e.g. edit error response
    text = make_user_friendly_error_message(event, ex)
    reply_message_request.messages[0] = TextMessage(text=text)

@aiavatar_app.event("postback")
async def handle_postback_event(event: Event, user_id: str, context_id: Optional[str]):
    # Process event
    # e.g. Register postback data
    await register_data(user_id, event.postback.data)
```


Context data is stored in `aiavatar.db` via SQLite by default. To use PostgreSQL, create a `PostgreSQLChannelContextBridge` and pass it to `AIAvatarLineBotServer` as `channel_context_bridge`. See [Adapters](adapters.md#sharing-context-across-channels) for details.

```python
from aiavatar.adapter.channel_context_bridge.postgres import PostgreSQLChannelContextBridge
bridge = PostgreSQLChannelContextBridge(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

aiavatar_app = AIAvatarLineBotServer(
    system_prompt="You are a cat.",
    openai_api_key=OPENAI_API_KEY,
    channel_access_token=LINEBOT_CHANNEL_ACCESS_TOKEN,
    channel_secret=LINEBOT_CHANNEL_SECRET,
    api_key=LINEBOT_ADMIN_API_KEY,
    image_download_url_base="https://{your.domain}",
    channel_context_bridge=bridge,    # <- Set PostgreSQL context bridge
    debug=True
)
```

## Attaching to an existing pipeline

The example above lets the adapter build its own pipeline. Pass `sts=` instead to attach to
one that already exists, which is how LINE shares a conversation with the web or the phone.
The default channel name is `"linebot"`.

```python
import os
from aiavatar.adapter.linebot.server import AIAvatarLineBotServer

line_adapter = AIAvatarLineBotServer(
    sts=sts,
    channel_access_token=os.environ["LINEBOT_CHANNEL_ACCESS_TOKEN"],
    channel_secret=os.environ["LINEBOT_CHANNEL_SECRET"],
    api_key=os.environ["LINEBOT_ADMIN_API_KEY"],   # Protects the management endpoints
    channel="linebot",
)
app.include_router(line_adapter.get_api_router(), prefix="/line")
```

Configure `https://your-domain.example/line/webhook` as the webhook URL in the LINE Developers Console. Supported messages and customization hooks are documented above.

## Protecting the management endpoints

The router exposes more than the webhook. Alongside `POST /webhook` it serves:

| Route | What it does |
| --- | --- |
| `GET /channel_user/{line_user_id}` | Returns the mapped application user id and current `context_id` |
| `DELETE /channel_user/{line_user_id}` | Deletes the mapping |
| `POST /push/{user_id}` | Makes the bot send a message to that user |
| `GET /image/{image_id}` | Serves an uploaded image |

The first three are guarded by bearer-token authentication — **but only when `api_key` is
set**. Leave it unset and they are open to anyone who can reach the router, which for a LINE
bot means the public internet, since the webhook has to be reachable by LINE.

```python
line_adapter = AIAvatarLineBotServer(
    sts=sts,
    channel_access_token=os.environ["LINEBOT_CHANNEL_ACCESS_TOKEN"],
    channel_secret=os.environ["LINEBOT_CHANNEL_SECRET"],
    api_key=os.environ["LINEBOT_ADMIN_API_KEY"],
)
```

Callers then present it as a bearer token:

```sh
curl -X POST https://your-domain.example/line/push/user_001 \
  -H "Authorization: Bearer $LINEBOT_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your order has shipped."}'
```

A missing or mismatched token returns `401`. Use a long random value, keep it in the
environment rather than in source, and rotate it if it leaks.

`POST /webhook` is not covered by `api_key` and does not need to be: it verifies LINE's
`X-Line-Signature` against your `channel_secret`, so requests that did not come from LINE
are rejected. `GET /image/{image_id}` is unauthenticated by design. When
`image_download_url_base` is set, its URL is passed to the LLM provider so it can retrieve
an image received from a LINE user. Treat anything in the upload directory as public.

## See also

- [Adapters](adapters.md) — sharing context across channels
- [Database](database.md) — the channel context bridge

---

[← Documentation index](../README.md#-documentation)
