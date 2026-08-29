# Adapter Agent Guide

This guide applies to `aiavatar/adapter/`. Also follow the repository-root
`AGENTS.md`, the nearest more-specific guide, and `tests/AGENTS.md` before
selecting or running tests.

## Scope and sources of truth

- Adapters own transport framing, authentication, encoding, provider callbacks,
  connection state, and transport cleanup.
- VAD, STT, LLM, TTS, queueing, and persistence policy belong in
  `aiavatar/sts/`.
- Treat adapter models, transport-specific models, and serialized event names as
  wire contracts.
- When translating requests or responses, inspect `aiavatar/sts/models.py` and
  preserve all applicable identifiers, metadata, structured content, control
  tags, tool calls, text, and audio fields.
- Read both sides of a transport together: server, client, shared models,
  maintained browser/example consumers, and focused tests.
- For Asterisk work, also follow `aiavatar/adapter/asterisk/AGENTS.md`.

## Boundary invariants

- `Adapter` registers a response handler on its supplied `STSPipeline`.
  Multiple adapters may share a pipeline only when their `can_handle()` rules
  assign every session to the intended owner without overlap.
- Do not duplicate pipeline state or processing policy inside an adapter.
- Keep mutable connection and request state scoped by the correct session,
  user, context, channel, and transaction identifiers.
- The identifier passed to STS/VAD identifies its live pipeline session;
  transport-facing identifiers may be translated at the adapter boundary only
  as documented by a nearer guide. `user_id` identifies an application user,
  and `context_id` identifies conversation continuity. Never substitute one for another.
- Pair every socket, stream, task, callback, mapping, and provider session with
  deterministic disconnect and error cleanup.
- Session teardown must run the applicable disconnect hooks, finalize captured
  pipeline/VAD state, and remove only mappings owned by that connection.
- Re-check ownership after an `await` whenever disconnect, replacement, or a
  newer transaction could have changed the active connection.
- Session-start hooks may rewrite a request before session values are stored.
  Request hooks run before invocation; response hooks run before delivery.
  Preserve the established ordering when adding transport-specific callbacks.
- Provider callback authentication is adapter-specific. Do not assume that
  authentication on one route or adapter protects another provider entrypoint.

## Runtime configuration

- If an adapter is exposed in the current Admin Panel, configuration candidates
  are derived from public constructor parameters that also exist as safe,
  JSON-compatible attributes on the running instance.
- Admin updates those attributes directly with `setattr` and does not use
  `get_config()` or `set_config()` as its configuration contract.
- Constructor-backed secret fields may be editable, but never return or log the
  existing value. Preserve masking and the convention that an empty submission
  leaves the current value unchanged.
- Keep clients, pools, locks, tasks, callbacks, derived state, and resource
  handles out of Admin-editable constructor fields.
- Runtime Admin changes are process-local. Do not add persistence or component
  replacement implicitly.

## AIAvatar WebSocket invariants

- These rules apply to `aiavatar/adapter/websocket/`, not to the Asterisk Media
  WebSocket or streaming-STT protocols.
- Treat `start` as client protocol initialization: clients send it to establish
  socket and session values before `data`, `invoke`, `config`, or `stop`.
  Adding server-side enforcement is a compatibility change and requires focused tests.
- Track one active response transaction per session. A newer accepted
  transaction invalidates stale output from the previous one.
- Preserve the established interruption notification before activating a new
  transaction.
- Responses carrying a stale transaction ID must not be delivered. Events that
  intentionally have no transaction ID remain valid according to the protocol.
- Hold the per-session send lock around each complete WebSocket write.
- During streamed audio, re-check the active transaction before sending further
  chunks so interrupted audio cannot leak into the next turn.
- Preserve both native-client and browser-compatible authentication paths.
- Keep complete-audio and chunked-audio modes compatible. Chunked audio must
  communicate its PCM format before raw PCM chunks are consumed.
- The browser example is a maintained protocol consumer; update and verify it
  when changing messages or audio behavior.

## Other transports

- HTTP chat streams JSON events over SSE; verify route and payload details in
  the current server and client rather than copying older documentation.
- OpenAI-shaped endpoints implement only their modeled compatibility surface.
  Preserve streaming termination and non-streaming response shapes.
- Streaming STT has its own request/response contract and audio-format
  assumptions; keep them aligned with the selected VAD and recognizer.
- Telephony and messaging adapters must preserve provider media conversion,
  callback ordering, identity mapping, interruption, and cleanup semantics.
- `ChannelContextBridge` owns channel identity mapping and conversation
  continuity. Preserve its timeout and context persistence triggered by the STS
  `start` response.

## Validation

- Update every maintained side of a protocol change: models, server, client,
  browser/example consumer, documentation, and focused tests as applicable.
- Test malformed input, missing or invalid identifiers, authentication failure,
  disconnect during send, concurrent writes, stale transactions, callback
  failure, cancellation, and finalization—not only the happy path.
- Start with the narrowest relevant test described by `tests/AGENTS.md`.
- For documentation-only changes, validate links, route names, examples, and
  consistency without invoking live integrations.
