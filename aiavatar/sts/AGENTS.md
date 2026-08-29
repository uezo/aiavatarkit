# Speech-to-Speech Agent Guide

This guide applies to `aiavatar/sts/`. Follow the repository-root `AGENTS.md`;
before selecting or running tests, also follow `tests/AGENTS.md`. This file
contains only speech-pipeline-specific rules.

## Boundaries and public surface

- `pipeline.py` owns VAD -> STT -> LLM/tools -> TTS orchestration. Keep transport
  encoding, connection management, client protocols, and wire models in
  `aiavatar/adapter/`.
- `models.py` contains pipeline messages; adapters translate them into public
  transport models.
- Treat package `__init__.py` files as the authority for supported imports.
  Concrete providers normally remain imported from their implementation modules.
- Treat exported spellings and event strings as compatibility surfaces. In
  particular, do not incidentally normalize `GuardrailRespose`, direct
  `"canceled"`, or queued `"cancelled"`.

## Pipeline and event invariants

- Construct `STSRequest` and `STSResponse` by keyword and preserve each event's
  established identifier shape; some control and queued events intentionally
  omit otherwise available identifiers.
- Preserve the request `transaction_id` across every response path, including
  cancellation, timeout, validation failure, and exceptions.
- `accepted` is dispatched out of band before processing. A normal invocation
  streams `start`, zero or more `chunk` and `tool_call` events, then `final`;
  validation or no-speech exits use `canceled`, and failures use `error`.
- Do not reorder or rename events without checking adapters and clients. Transaction
  filtering, barge-in, and stream completion depend on this lifecycle.
- A newer active transaction must suppress stale LLM and synthesis output for the
  same session. Preserve active-transaction checks on both streaming boundaries.
- Invocation queues are per session. Preserve the distinction between clearing
  pending work with `wait_in_queue=False` and processing it sequentially with
  `wait_in_queue=True`.
- Treat changes to queue mode while requests are active as lifecycle changes;
  balance workers, response queues, timeouts, cancellation, and idle cleanup.
- Hooks may intentionally mutate the current request. Do not move request-specific
  values into shared component configuration or module globals.

## Session and VAD invariants

- Batch recognition overrides belong to `STSPipeline`; streaming recognition
  overrides belong to the stream VAD. Keep both override paths scoped by session.
- Do not mutate a shared recognizer to switch one session. Concurrent sessions may
  require different engines, and reset behavior must preserve session isolation.
- Use explicit audio-state reset when a boundary must discard buffered audio;
  the streaming implementation also uses it to cancel pending recognition.
  Preserve the normal per-turn reset behavior that retains pre-roll/VAD buffering
  for continuous recording.
- Keep session deletion and finalization teardown idempotent, and preserve their
  detector-specific gate, filter, and session cleanup. Do not assume that teardown
  cancels recognition work already in flight; inspect that lifecycle before changing it.
- Treat turn-end gate ordering, wait state, timeouts, cleanup, and failure fallback
  as observable policy. Change them only with focused concurrency coverage.
- Preserve the VAD callback contract
  `(audio, text, metadata, recorded_duration, session_id)`.
- When VAD performance metadata is present, preserve the timing fields in
  `metadata["vad_performance"]` across recording, persistence, queries, and Admin
  consumers. Keep recorder and query `time_origin` settings aligned.
- Keep audio assumptions explicit and consistent across detectors, recognizers,
  synthesizers, recorders, and adapters: sample rate, channels, sample width,
  raw PCM versus WAV, preprocessing, and chunk boundaries.

## Component contracts

- STT providers implement `transcribe()` and retain the base `recognize()`
  preprocessing, postprocessing, and `SpeechRecognitionResult` flow.
- LLM providers keep shared guardrail, tool-call, context-update, voice-text, and
  streaming behavior in `LLMService.chat_stream()`. Preserve shared response and
  structured-content shapes.
- TTS providers implement `generate()` without bypassing the common
  `SpeechSynthesizer.synthesize()` flow for empty input, preprocessing, caching,
  and postprocessing. Provider methods receive already-preprocessed text.
- A synthesis cache key must include every request field that can change generated
  audio; override the base key when provider-specific model, speaker, speed, or
  equivalent settings affect output.
- Register resources constructed by `STSPipeline` with its lifecycle so shutdown
  closes them. Caller-injected resources remain caller-owned; lifecycle changes
  must be explicit and covered by cleanup and failure-path tests.
- For persistent STS state, preserve schema migration, SQLite/PostgreSQL behavior,
  pool ownership, and idempotent cleanup.

## Change routing and validation

- Start from the owning base class and compare sibling implementations before
  changing a provider contract.
- For pipeline message or event changes, inspect adapters, client filtering,
  examples, and focused adapter tests.
- For VAD timing or persistence changes, inspect recorders, query semantics,
  schema migrations, both database backends, and Admin consumers.
- For public import changes, update the owning `__init__.py` only when a new public
  API is intended; test that exact import and update user-facing documentation.
- Use deterministic dummies, fakes, and monkeypatches for concurrency and error
  paths. Do not substitute live providers for focused regression coverage.
- Select tests through `tests/AGENTS.md`; provider, PostgreSQL, model-backed,
  local-service, and hardware tests require their stated prerequisites and authorization.
