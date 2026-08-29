# Admin Panel Agent Guide

This guide applies to `aiavatar/admin/`. Also follow the repository-root
`AGENTS.md` and read `tests/AGENTS.md` before selecting or running tests.

## Scope and boundaries

- This is the current FastAPI-embedded Admin Panel.
  `aiavatar/admin_legacy/` is a separately preserved implementation; change it
  only when the task explicitly includes it.
- Read `aiavatar/admin/README.md`, then verify routes, models, defaults, and UI
  behavior against current source and focused tests.
- Keep Admin HTTP handlers thin. Database filtering, grouping, time bucketing,
  and latency calculations belong in their domain query services.
- Evaluation is optional. Routes and UI capabilities must reflect whether an
  evaluator is actually available.
- Preserve `event_at = speech_end_at ?? created_at` consistently across metrics,
  logs, ordering, and time buckets.

## Runtime configuration

- Config forms are derived from public constructor parameters that also exist
  as safe, JSON-compatible attributes on the running component.
- Updates use direct `setattr`, affect the current process only, and do not use
  component `get_config()` or `set_config()` methods.
- Do not persist Admin changes or replace component instances implicitly.
- Secret constructor fields may be editable, but never return or log their
  existing values. Preserve masking and the convention that an empty submission
  leaves the current value unchanged.
- Exclude clients, pools, tasks, callbacks, locks, resource handles, derived
  state, and initialization-only parameters.
- The application owns the object graph. Display the active component class
  without offering unsafe class replacement.
- When changing a configurable constructor field, inspect its Admin form type,
  serialization, validation, and runtime assignment behavior together.

## Authentication and API invariants

- Router-level authentication must protect the page, static assets, and every
  Admin API route uniformly.
- Keep custom authentication behind the callable boundary in `auth.py`; do not
  embed provider-specific SSO logic in individual endpoints.
- Never hard-code, return, log, or expose credentials.
- Preserve static-path containment so requests cannot escape the packaged Admin
  asset directory.
- Keep endpoint request and response shapes synchronized with the browser
  modules and Admin documentation.
- Test both authenticated and rejected requests whenever authentication or
  routing changes.

## Frontend and assets

- Keep Python route modules focused on their domain services.
- The frontend uses same-origin, page-relative Admin API requests; preserve that
  resolution beneath the fixed `/admin` mount.
- Do not introduce a Node.js runtime or frontend framework without an explicit
  architectural request.
- When adding or renaming a static module, update its imports, HTML entrypoint,
  package-data inclusion, and asset-serving tests together.
- `static/admin.tailwind.css` is the editable CSS source.
  `static/admin.css` is the tracked generated package asset.
- Regenerate `admin.css` with the Tailwind workflow documented in
  `aiavatar/admin/README.md`; do not hand-edit only the generated file.
- If the required generator is unavailable, report the ungenerated source
  change clearly rather than claiming the packaged CSS is current.

## Validation

- Start with the focused Admin test using the isolated invocation prescribed by
  `tests/AGENTS.md`.
- For API changes, verify authentication, route prefixes, response shapes,
  invalid input, empty results, and unavailable capabilities.
- For browser changes, verify module loading, API failures, empty/loading states,
  light and dark themes, and relevant viewport sizes.
- Update `aiavatar/admin/README.md` when public routes, screens, time semantics,
  authentication, configuration behavior, or the CSS workflow changes.
