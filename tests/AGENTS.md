# Test Guide

These instructions apply to everything under `tests/`. Also follow the
repository-root `AGENTS.md`. Run test commands from the repository root.

## Instruction routing

- Read this guide before selecting, collecting, or running any test.
- For tests that mirror a scoped source area, also read its source guide even
  when only test files are changing:
  - `tests/sts/**` -> `aiavatar/sts/AGENTS.md`
  - `tests/adapter/**` -> `aiavatar/adapter/AGENTS.md`
  - `tests/adapter/asterisk/**` -> `aiavatar/adapter/asterisk/AGENTS.md`
  - `tests/admin/**` -> `aiavatar/admin/AGENTS.md`
- If a more specific guide exists for the source under test, read it too.

## Excluded suite

- Exclude `tests/character/**` from routine validation, collection, and broad
  test selection. It is slow and covers `aiavatar/character`, which is planned
  for deprecation and is outside the supported test scope.
- Run it only when the user explicitly scopes the task to `aiavatar/character`
  or `tests/character`.
- When an authorized broader selection could include it, pass
  `--ignore=tests/character`.

## Safe execution

- Read the target test, its fixtures, imported helpers, and subject imports
  before executing it. Collection imports modules, so `--collect-only` is not a
  substitute for inspection.
- Run the narrowest relevant target: one test node, one file, or a small
  confirmed-local group.
- Do not run the full pytest suite by default. It mixes hermetic tests with paid
  APIs, databases, local services, model downloads, servers, and hardware.
- Use the isolated invocation:

  ```sh
  python -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
    tests/path/test_file.py::test_name -q
  ```

- Replace the target only after confirming its dependencies and side effects.
- Run async tests through pytest so async fixtures and teardown execute normally.
- Check `git status --short` before and after testing when filesystem effects are
  possible.
- Report the exact target, pass/fail/skip counts, warnings, and unavailable
  prerequisites.

## External dependencies and secrets

- Never open, print, copy, edit, or reuse secrets from `.env`, the ignored root
  `pytest.ini`, or other ignored local configuration.
- Some legacy helpers read `pytest.ini` directly. The isolated pytest command
  prevents pytest itself from loading that file, but cannot prevent test code
  from opening it.
- Inspect selected tests and helpers for environment access, localhost URLs,
  client SDKs, subprocesses, databases, model loading, and device access.
- A skip marker—or the absence of one—is not proof that a test is hermetic.
- Run tests that contact a real provider, PostgreSQL database, network or local
  service, model runtime, or hardware only after explicit authorization for the
  named dependency, credentials, possible cost, and side effects.
- If a prerequisite is unavailable, report it. Do not borrow local credentials,
  install unrelated dependencies, or start services merely to make a test pass.
- Supply authorized credentials through the process environment only.
- Do not add tests that read ignored local configuration.

## Isolation and cleanup

- Prefer `tmp_path`, fakes, deterministic dummy services, and monkeypatching.
- Pass explicit temporary paths for databases, caches, recordings, media, and
  generated output; do not rely on repository-local defaults.
- Keep mutable test state scoped to the test or fixture.
- Use yielding fixtures or `try/finally` so cleanup also runs after failure.
- Await client and pool shutdown, cancel and join tasks, and close streams,
  queues, recorder workers, and subprocesses.
- Database tests must use unique records and remove only records they created.
  Never assume a local database is disposable.
- Remove only artifacts created by the current test. Do not use `git clean` or
  delete ignored user data.

## Adding or changing tests

- Prefer deterministic unit tests over live provider calls.
- Keep integration tests visibly separate and document their prerequisites,
  network use, cost, and cleanup behavior.
- Skip integrations safely when their explicit configuration is absent.
- Keep tests order-independent and free of state left by earlier tests.
- Add focused regression coverage for changed behavior and failure paths.
- Match nearby test style and avoid unrelated rewrites.
