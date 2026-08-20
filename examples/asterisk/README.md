# Asterisk Media WebSocket / Avaya adapter example

This example terminates Avaya SIP/RTP on Asterisk and connects the call to
AIAvatarKit over Asterisk's Media WebSocket protocol. A separate ARI manager
owns bridges, call teardown, SIP REFER, and the Asterisk bridge fallback.

## Requirements

- Python 3.11+
- Asterisk 20.18+, 22.8+, 23.2+, or a later branch with `chan_websocket`,
  PJSIP, ARI, JSON Media WebSocket control, `transport_data`, and HTTPS/WSS
- An Avaya-to-Asterisk PJSIP trunk
- A reachable VOICEVOX-compatible server for the default AIAvatarKit pipeline
- An OpenAI API key for the default pipeline

Install the repository in editable mode and set the application credentials:

```sh
python -m pip install -e .
export OPENAI_API_KEY=CHANGE_ME
export AIAVATAR_MEDIA_USERNAME=aiavatarkit
export AIAVATAR_MEDIA_PASSWORD=CHANGE_ME
export ASTERISK_ARI_BASE_URL=https://asterisk.internal:8089/ari
export ASTERISK_ARI_USERNAME=aiavatar
export ASTERISK_ARI_PASSWORD=CHANGE_ME
export ASTERISK_TRANSFER_STRATEGY=refer
export ASTERISK_REFER_TIMEOUT=30
export ASTERISK_MEDIA_START_TIMEOUT=10
```

Run from the repository root:

```sh
python -m uvicorn examples.asterisk.run:app --host 0.0.0.0 --port 8000
```

The application entry point is `run.py`.

The Media WebSocket endpoint is `/asterisk/media`. Terminate TLS at the app or
at a trusted reverse proxy so Asterisk connects with `wss://`.

## Asterisk configuration

Copy and adapt these examples rather than installing them verbatim:

- `websocket_client.conf.example`: outbound Media WebSocket connection. Its
  section name must equal `ASTERISK_MEDIA_CONNECTION`.
- `http.conf.example`: private-network ARI HTTP/HTTPS listener.
- `ari.conf.example`: ARI account used only on the private management network.
- `extensions.conf.example`: inbound Stasis entry and allowlisted REFER
  destinations.
- `pjsip.conf.example`: conceptual trusted Avaya trunk and caller identity
  settings.

The manager creates External Media with `transport=websocket`,
`encapsulation=none`, `format=slin16`, and `transport_data=f(json)`. Asterisk
therefore sends 16 kHz signed linear PCM as BINARY frames and JSON control
events as TEXT frames using the `media` subprotocol.

A media channel establishes its WebSocket once. Do not reconnect the same
channel after a disconnect. A transfer recovery creates and registers a new
External Media channel instead.

## Transfer behavior

An AI response can request only an application-owned alias:

```xml
<operation name="transfer" destination="operator" />
```

The adapter waits for `MEDIA_MARK_PROCESSED` so the final transfer announcement
finishes before call control starts. `operator` is resolved through
`transfer_destinations`; arbitrary numbers and SIP URIs are rejected.

Use `on_transfer_prepare` to attach application-owned handoff keys as Asterisk
channel variables without asking the LLM to produce them:

```python
from aiavatar.adapter.asterisk import (
    AsteriskSessionData,
    AsteriskTransferRequest,
)


@asterisk.on_transfer_prepare
async def prepare_handoff(
    request: AsteriskTransferRequest,
    session: AsteriskSessionData,
) -> None:
    request.variables["AIAVATAR_USER_ID"] = request.user_id
    if request.context_id:
        request.variables["AIAVATAR_CONTEXT_ID"] = request.context_id
```

The callback mutates `request.variables` in place and returns nothing. Assigning
an existing key overwrites its current value. `request` and `session` are mutable
dataclasses, but only `request.variables` is consumed as transfer output;
`destination` and `transfer_strategy` are informational snapshots. Use
`session.data` for application-owned per-call state instead of changing
manager-owned lifecycle fields.

The variables are placed on the caller channel for REFER and on the outbound
channel for bridge transfer. They are not SIP headers by themselves; map them
to the agreed SIP header, UUI, or Refer-To parameter in trusted Asterisk/Avaya
configuration. A failed hook aborts the transfer with
`transfer_prepare_failed`.

With the `refer_then_bridge` strategy, the caller leaves Stasis at the
allowlisted extension in `aiavatar-transfer`. The dialplan runs `Transfer()` and
returns to Stasis as `transfer-failed` when REFER is rejected or unsupported.
The manager then holds the caller, originates a new Avaya channel with the
saved caller identity, and bridges both parties after answer. A REFER may
destroy the original Asterisk channel before an explicit result. Because the
same event can be an ordinary caller hangup, the manager reports it through
`on_transfer_unknown` and does not claim a completed handoff.

While a REFER result is pending, the per-call manager arms a watchdog using
`refer_timeout` (30 seconds by default). If the terminal Stasis event was
missed, it reconciles the caller channel through ARI and completes the handoff
or restores AI media. If `Transfer()` is still running outside Stasis, the
manager waits and rechecks instead of racing Asterisk with an unsafe media
restore. The watchdog is inactive for the `bridge` strategy.

After a failed transfer, the call remains in `restoring_ai` until the
replacement Media WebSocket reports `MEDIA_START`. `media_start_timeout` (10
seconds by default) terminates the call if media cannot be restored, preventing
an immediate failure response from being sent before the socket is ready.

Do not derive P-Asserted-Identity from an untrusted endpoint. The example reads
the original identity only in the Avaya-bound ingress context, preserves caller
presentation, and relies on the trusted Avaya PJSIP endpoint's identity policy
for the fallback INVITE. Validate anonymous calls, PAI, Refer-To adaptation,
UCID/UUI, and terminal display behavior with Avaya SIP traces before production.
