# Twilio adapter

Two adapters cover Twilio. `AIAvatarTwilioServer` bridges Twilio Media Streams to the
pipeline for voice calls, inbound and outbound. `AIAvatarTwilioSMSServer` handles messaging.
They are separate because sessions and response delivery work differently for a live call
and for SMS, but they can share one Twilio client and one pipeline.

```sh
pip install twilio
```

## ⚠️ Protect these routers before exposing them

Both routers mix Twilio's inbound webhooks with outbound *action* endpoints, and the action
endpoints have **no authentication of any kind**:

| Route | What an anonymous caller can do |
| --- | --- |
| `POST /call/make` | Place a call from your Twilio number to any number they choose, speaking text they supply |
| `POST /sms/send` | Send an SMS from your Twilio number to any number |

Unlike the [LINE Bot adapter](adapters-linebot.md#protecting-the-management-endpoints), these
classes take no `api_key` argument, so there is nothing to switch on. The webhook routes also
do **not** verify Twilio's `X-Twilio-Signature`, so inbound requests are not checked for
authenticity either.

A Twilio webhook has to be reachable from the internet. Mounting these routers as shown below
therefore publishes your telephony billing to anyone who finds the path.

### What has to be public, and what must not be

The router mixes two kinds of route, and they need opposite treatment.

| Route | Who calls it | Exposure |
| --- | --- | --- |
| `POST /voice` | Twilio, as the voice webhook | **Public** |
| `WS /ws` | Twilio Media Streams | **Public** |
| `POST /sms` | Twilio, as the messaging webhook | Public, if you use SMS |
| `POST /call/make` | Your application | **Must not be public** |
| `POST /sms/send` | Your application | **Must not be public** |

`/ws` is easy to miss. The `/voice` handler answers with TwiML whose `<Stream>` points at
`{webhook_base_url}/ws`, so Twilio connects there for the audio itself — blocking it leaves
calls that connect and then hear nothing.

### Protecting the webhooks

These have to accept requests from the internet, so authenticate the *caller* instead of
hiding the path. Validate `X-Twilio-Signature` with your Twilio Auth Token and the Twilio SDK
before the request reaches `/voice`, `/sms`, or the WebSocket upgrade. Do **not** compare the
header directly with the Auth Token: Twilio signs the exact public request URL and the request
parameters, using the Auth Token as the signing key. Make sure a reverse proxy does not hide
the original scheme or host from validation, and validate `/ws` against its public `wss://`
URL. The adapter does not do this for you. See
[Twilio's request-validation guide](https://www.twilio.com/docs/usage/security#validating-requests-are-coming-from-twilio).

### Protecting the action endpoints

These should never be reachable from outside. In rough order of how little they trust the
network:

- **Keep them off the public router.** Mount the adapter's router on an internal-only app or
  port, and expose only `/voice`, `/ws`, and `/sms` through your public proxy.
- **Block them at the edge.** Deny `/call/make` and `/sms/send` at the reverse proxy, load
  balancer, or API gateway, and reach them from inside the network only.
- **Put your own authentication in front.** Declare those two paths on your own router
  behind a dependency and call `make_call()` / `send_sms()` yourself, rather than mounting
  the adapter's versions at all.

Treat the examples below as wiring diagrams, not as deployable configuration.

## Twilio Voice Adapter

`AIAvatarTwilioServer` connects Twilio Media Streams to the pipeline. Its default channel name is `"phone"`. If the router is mounted at `/twilio`, include that prefix in `webhook_base_url` so the generated WebSocket URL points to `/twilio/ws`.

```python
import os
from aiavatar.adapter.twilio.server import AIAvatarTwilioServer

twilio_voice_adapter = AIAvatarTwilioServer(
    sts=sts,
    account_sid=os.environ["TWILIO_ACCOUNT_SID"],
    auth_token=os.environ["TWILIO_AUTH_TOKEN"],
    phone_number=os.environ["TWILIO_PHONE_NUMBER"],
    webhook_base_url="https://your-domain.example/twilio",
    channel="phone",
)
app.include_router(twilio_voice_adapter.get_router(), prefix="/twilio")
```

Configure `https://your-domain.example/twilio/voice` as the Twilio voice webhook.

## Twilio SMS Adapter

Voice and SMS use separate adapters because they have different session and response-delivery mechanisms. `AIAvatarTwilioSMSServer` requires an existing pipeline and defaults to the `"sms"` channel. It can reuse the Twilio client created by the voice adapter.

```python
from aiavatar.adapter.twilio.server import AIAvatarTwilioSMSServer

twilio_sms_adapter = AIAvatarTwilioSMSServer(
    sts=sts,
    twilio_client=twilio_voice_adapter.twilio_client,
    phone_number=os.environ["TWILIO_PHONE_NUMBER"],
    channel="sms",
)
app.include_router(twilio_sms_adapter.get_router(path="/sms"), prefix="/twilio")
```

Configure `https://your-domain.example/twilio/sms` as the Twilio messaging webhook. If voice is not enabled, pass `account_sid` and `auth_token` directly instead of `twilio_client`.

## See also

- [Adapters](adapters.md) — sharing one pipeline across channels
- [Asterisk adapter](adapters-asterisk.md) — self-hosted telephony

---

[← Documentation index](../README.md#-documentation)
