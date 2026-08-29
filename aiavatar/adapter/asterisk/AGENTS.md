# Asterisk Adapter Agent Guide

This guide applies to `aiavatar/adapter/asterisk/`. Also follow the
repository-root and `aiavatar/adapter/AGENTS.md` guides, and read
`tests/AGENTS.md` before selecting or running tests.

## Ownership model

- The per-call actor coordinated by `AsteriskCallManager` is the sole owner of
  call lifecycle state.
- `AsteriskARIClient` owns ARI transport, the call registry owns indexes, the
  event handler owns raw-event routing, pre-registration setup, and cleanup of
  callers that never completed registration, and the call service owns
  registered-call ARI topology mutations.
- Do not introduce a second lifecycle state owner for transfers, media recovery,
  playback, or cleanup.
- Registry indexes must remain mutually consistent. Do not leave a partially
  updated index visible across an `await`.
- The public call `session_id` remains stable across media recovery.
- Each Media WebSocket channel has a distinct private STS/VAD session key.
  Translate between the private key and stable public call ID only at the
  adapter boundary.

## Connection and transaction concurrency

- Claim a new response transaction before the first operation that can yield
  control.
- Revoke stale playback and operation ownership before issuing an interrupting
  media command.
- After every awaited callback, manager notification, media command, or setup
  operation, re-check that the session mapping, socket, transaction, connection
  generation, and cleanup state are still owned by the current operation.
- Deliver transactionless responses only while the session has no active
  transaction.
- Keep Media WebSocket connection ownership separate from playback
  cancellation. Use connection generation for socket and callback ownership,
  and playback generation only for audio/playback invalidation.
- Scope callback tasks to the session and connection generation that created
  them. An old disconnect must not cancel callbacks owned by a replacement
  connection.
- A manager-authorized media channel accepts exactly one Media WebSocket.
  Recovery creates and registers a new media channel identity rather than
  attaching a second socket to the old one.

## Cleanup and recovery

- Media and call cleanup must run in shared, shielded cleanup tasks.
  Cancellation of one waiter must not cancel underlying resource release.
- Capture the private STS/VAD key owned by the connection being cleaned up.
  Remove its routing entry before awaiting unrelated cleanup work, then finalize
  only that captured key.
- Cleanup that waits for an older connection must re-evaluate current ownership
  afterward and continue until no manager-authorized replacement resource still
  requires cleanup.
- A stale connection may remove only its own socket, tasks, mappings, and media
  state.
- Discard adapter-owned synthesis or callback results if connection or playback
  ownership changes while awaiting them.
- During graceful shutdown, cancel pending setup work and release the remote
  caller even when cancellation happens before local session registration.
- Keep close and cancellation paths idempotent so disconnect, manager shutdown,
  and error recovery may converge safely.

## Transfers and uncertain outcomes

- Transfer preparation is the veto-capable phase. Started, completed, failed,
  and unknown callbacks are notifications whose exceptions must be isolated
  from call control.
- Notification callbacks must not recursively transfer or hang up the same call.
- Distinguish confirmed success, confirmed failure, and unknown ARI outcomes.
  Timeouts, transport failures, and ambiguous server failures may occur after
  the remote mutation has started.
- Reconcile unknown outcomes before attempting a fallback that could duplicate
  or conflict with the original topology mutation.
- Unknown or empty provider transfer status is not a confirmed failure; fail
  closed unless ownership can be established safely.

## Validation

- Add focused concurrency coverage for disconnect during setup, replacement
  Media WebSockets, stale responses, callback cancellation, cleanup waiter
  cancellation, shutdown before registration, and transfer outcomes.
- Assert both the stable public call identity and private STS/VAD routing
  identity after recovery and cleanup.
- Use fakes for ARI and Media WebSocket behavior unless the user explicitly
  authorizes a live Asterisk integration.
