# Asterisk adapter

`AIAvatarAsteriskServer` connects Asterisk ARI call control and a bidirectional media
WebSocket to an existing `STSPipeline`, for deployments that run their own telephony rather
than going through a cloud provider.

The Asterisk adapter connects ARI call control and a bidirectional Media
WebSocket to an existing `STSPipeline`. Setup, transfer strategies, lifecycle
behavior, Asterisk configuration examples, and operational constraints are
documented in the [Asterisk adapter guide](https://github.com/uezo/aiavatarkit/blob/main/aiavatar/adapter/asterisk/README.md).

Setup, transfer strategies, call lifecycle, Asterisk configuration examples, and
operational constraints are documented in detail in the adapter's own guide:

- [`aiavatar/adapter/asterisk/README.md`](../aiavatar/adapter/asterisk/README.md)

The adapter defaults to the `phone` channel, the same as the Twilio voice adapter, so
channel-aware behaviour written for one applies to the other.

## See also

- [Adapters](adapters.md) — sharing one pipeline across channels
- [Twilio adapter](adapters-twilio.md) — cloud telephony

---

[← Documentation index](../README.md#-documentation)
