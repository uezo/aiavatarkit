# AIAvatarKit Agent Guide

## Scope and instruction routing

This guide applies to the whole repository. Before changing a scoped area, read
its nearest guide:

- `aiavatar/sts/AGENTS.md` for the speech-to-speech pipeline and providers.
- `aiavatar/adapter/AGENTS.md` for transports, protocols, and clients.
- `aiavatar/admin/AGENTS.md` for the current Admin Panel.
- `aiavatar/admin_legacy/` is an independently preserved implementation, not a
  compatibility layer for the current Admin Panel; change it only when explicitly scoped.
- `tests/AGENTS.md` before selecting or running tests.

Scoped guides add local rules; they do not replace this guide.

## Sources of truth and compatibility

- Treat implementation, package `__init__.py` exports, and focused tests as the
  authority for public names, signatures, defaults, and behavior.
- Treat `README.md`, `FEATURES.md`, `FEATURES_TABLE.md`, and component READMEs as
  orientation. Re-check their claims against source before relying on them.
- Keep package versions, Python requirements, dependencies, and package-data
  declarations in the active packaging metadata; do not copy volatile inventories here.
- Treat established public names, defaults, protocol fields, and event strings as
  compatibility surfaces. Change them deliberately and with focused regression coverage.

## Working-tree boundaries

- This checkout may contain ignored databases, recordings, credentials, character
  data, generated media, caches, and root-level experiments. They are user data,
  not disposable fixtures.
- Run `git status --short` and use `git ls-files` to establish canonical scope
  before broad searches or edits.
- Preserve unrelated tracked, untracked, and ignored files. Do not use `git clean`,
  destructive resets, or broad deletion commands.
- Treat build outputs, caches, recordings, generated media, and local databases as
  outputs rather than implementation sources unless the task explicitly targets them.
- Do not inspect ignored local scripts, databases, logs, credentials, or character
  data unless the user places them in scope.
- Never print, copy, or commit secrets. Use environment-variable names and obvious
  placeholders in code, tests, examples, and documentation.

## Repository-wide change invariants

- Keep provider-, hardware-, database-, and platform-specific dependencies optional
  unless core code imports them unconditionally.
- Keep optional imports lazy or guarded so unrelated package imports work without
  every provider extra installed.
- When changing dependencies or package data, update the active packaging metadata
  and any compatibility shim that still exists.
- Do not add blocking network, database, model, audio, or filesystem work to an
  asyncio event loop.
- Keep mutable request and conversation state scoped by the relevant session, user,
  context, and transaction identifiers; do not use process globals for per-session state.
- Pair tasks, streams, clients, pools, queues, threads, and background workers with
  cancellation or close paths, including failure and disconnect paths.
- Treat changes to shared models, callbacks, config fields, and protocol values as
  cross-boundary changes; inspect every producer and consumer they affect.
- Where SQLite and PostgreSQL implement the same abstraction, preserve observable
  parity or document and test an intentional difference.
- Do not silently change public defaults. Add focused regression tests when behavior changes.

## Examples and runnable applications

- Match implementation size to the request; keep minimal examples minimal.
- Put maintained repository examples under `examples/`. Root-level scripts may be
  ignored local experiments and must not be treated as canonical sources.
- Do not change library internals solely to make an example work. Verify the current
  public API and document only the prerequisites needed by that example.

## Validation and documentation

- Follow `tests/AGENTS.md`. Start with the narrowest relevant test after inspecting
  its imports, fixtures, external services, credentials, and filesystem effects.
- Do not run the full pytest suite by default or use discovered live credentials.
  Report missing optional dependencies or services instead of starting unrelated systems.
- For documentation-only changes, validate imports, paths, commands, links, and
  consistency; runtime tests are unnecessary unless executable behavior changed.
- Update user-facing documentation when a public feature, setup step, default, or
  behavior changes, and verify exact claims against source.
- Add to an `AGENTS.md` only when a rule is durable, non-obvious, and operationally
  important. Prefer links to owning documentation over copied feature descriptions.
