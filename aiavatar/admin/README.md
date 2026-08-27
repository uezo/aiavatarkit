# Admin Panel

The Admin Panel is a FastAPI-embedded interface for monitoring AIAvatarKit and changing its runtime configuration. Calling `setup_admin_panel()` installs the UI, static assets, and management APIs under `/admin`.

This package is not a compatibility implementation of the previous Admin Panel. Applications that still require the previous UI and API can use the independently preserved `aiavatar.admin_legacy` package.

## Setup

```python
import os

from fastapi import FastAPI

from aiavatar.admin import BasicAdminAuthenticator, setup_admin_panel

app = FastAPI()

# adapter is the Adapter used by the application.
admin = setup_admin_panel(
    app,
    adapter=adapter,
    title="AIAvatarKit Admin Panel",
    authenticator=BasicAdminAuthenticator(
        os.environ["ADMIN_USERNAME"],
        os.environ["ADMIN_PASSWORD"],
    ),
)
```

The Admin Panel is available at `/admin/`. Requests to `/admin` are redirected to `/admin/`.

Use the returned `AdminPanel` handle to add another Adapter to the Config view.

```python
admin.add_adapter(http_adapter, name="http")
```

When `name` is omitted, a short name is derived from the Adapter class name. For example, `AIAvatarWebSocketServer` becomes `websocket`.

## UI

### Metrics

The Metrics view groups latency by the `channel` recorded on each Pipeline request. It does not combine latency averages across channels, so text-only and voice traffic do not distort each other's statistics.

- Period: `1h`, `6h`, `24h`, `7d`, or `30d`
- Interval: `1m`, `5m`, `15m`, `1h`, or `1d`
- Overall summary: request, success, error, and channel counts
- Per-channel summary: request count, error count, average, median, and P95 displayed latency
- Per-channel chart: Speech latency when speech timing is available; otherwise Pipeline latency

The UI shows one latency chart per channel. A channel with measurable speech timing uses latency from speech end to first content output. A text-only channel instead uses latency from Pipeline invocation to first content output. The API returns both metric sets so other consumers can choose explicitly.

The Pipeline form of the chart contains six phases:

1. Input / STT
2. Stop current response
3. Before-LLM handlers
4. LLM
5. Processing
6. TTS

The Speech form prepends three speech phases to the same Pipeline phases:

1. Silence detection
2. Streaming STT finalization
3. Turn-end gate

The first content output endpoint is selected from existing timing fields in this order:

1. Quick Response: `before_llm_time`
2. Voice response: `tts_first_chunk_time`
3. Text response fallback: `llm_first_chunk_time`

No additional Pipeline start or first-output timestamps are persisted. When an intermediate timing point is absent, its unknown interval is included in the next known phase so that the stack remains equal to the measured latency.

The request count includes every record in the selected period. Pipeline coverage includes successful records with one of the endpoints above, including records without `speech_end_at`. Speech coverage is the subset that also has usable speech-end and VAD timing. Failed requests and records without a usable endpoint remain in request/error counts but not in latency averages.

New requests receive their channel from `STSRequest.channel`. Historical records without a channel are grouped as `Unclassified`.

### Logs

The Logs view groups conversation messages by `context_id`. The table displays Context ID, message count, period, and User ID. Selecting a row opens the turn details in a drawer on the right.

The following filters are available. All specified conditions are combined with AND.

- User ID: exact match
- Session ID: exact match
- Context ID: exact match
- Channel: exact transaction match; returns contexts containing at least one matching transaction
- Keyword: searches request, response, voice response, quick response, error, and tool-call data
- Error presence
- Limit: 1–10,000; default 200

The limit applies to the number of messages retrieved. The messages are then grouped by `context_id`.

The drawer displays:

- Timestamp, Channel, Session ID, User ID, Context ID, and Transaction ID
- Request, Response, Error, and Tool calls
- First-response time and the detailed nine-phase breakdown for each turn
- Request and Response audio playback

Audio controls are shown only when the Pipeline Voice Recorder is enabled. When a Quick Response is present, its audio is played before the regular Response audio.

### Config

The Config view reads and updates the currently running objects:

- Pipeline
- VAD
- STT
- LLM
- TTS
- Registered Adapters

Input fields are generated from constructor metadata for parameters that also exist as JSON-compatible members on the running object. Applying a form updates those members directly. `get_config()` and `set_config()` are not used.

Card titles include the active class as static text because the object graph is owned by application code, for example `VAD: SileroStreamSpeechDetector`. Parameters that initialize resources, derived state, or registered hooks are omitted when they cannot be changed safely by member assignment.

When TTS is a `SpeechSynthesizerRouter`, the Config view shows one section for each registered route, such as `TTS · ja` and `TTS · multi`. Each section updates only that route's active synthesizer. The Router, route-selection function, provider classes, preprocessors, and postprocessors remain owned by application code and cannot be changed from Admin.

All changes are volatile. They take effect in the current process and are discarded when it exits. The Admin Panel does not select component classes, write application configuration files, or recreate components. Applications that need persistent configuration must express it in their own code or configuration layer.

Leave a nullable field blank to set it to `None`. Blank secret fields are the exception: they keep the currently configured value. Values cached when a live session is created, such as Silero's speech probability threshold, apply to sessions created after the change; existing sessions continue with their current value.

### Evaluation

The Evaluation view accepts a JSON array of scenarios and runs Dialog Evaluation in the background. Results are saved to `evaluation_results/<evaluation_id>.json`, relative to the process working directory.

The Evaluation tab is shown only when an Evaluator is available. An Evaluator is created automatically when the Adapter uses `ChatGPTService` as its LLM. For other LLMs, pass a `DialogEvaluator` through `setup_admin_panel(..., evaluator=...)`.

The automatic Evaluator reuses the source `ChatGPTService` client, preserving Azure,
custom transport, and tracing configuration. It does not take ownership of that client.

Character and Control UI and APIs are not included in this Admin Panel.

## Time Semantics

Metrics period selection and bucketing, as well as Logs display and ordering, use the same event timestamp:

```text
event_at = speech_end_at ?? created_at
```

`speech_end_at` is the time at which the user's speech ended. `created_at`, which represents record persistence time, is used as the event timestamp for text requests and as a fallback for older records without `speech_end_at`.

For compatibility, the Logs API still returns this value in a field named `created_at`, but its value is the `event_at` defined above. Records without `speech_end_at` can appear in the channel-specific Pipeline metrics and the log list, but not in the Speech breakdown. The existing aggregate Metrics endpoints and the per-turn Logs breakdown remain speech-based for compatibility.

## Authentication

### Authentication Boundary

`setup_admin_panel()` attaches one authentication dependency to the entire `/admin` router. The same authentication mechanism therefore protects:

- HTML (`/admin/`)
- JavaScript and CSS (`/admin/assets/...`)
- APIs (`/admin/api/...`)

The frontend fetches the same-origin `/admin/api/` endpoints and does not store or send a separate API key. Basic Authorization headers and SSO cookies are applied to both the page and its APIs through the browser's same-origin authentication behavior.

### Basic Authentication

`BasicAdminAuthenticator` provides HTTP Basic authentication. It compares usernames and passwords with `secrets.compare_digest` and returns `401` with a `WWW-Authenticate` header when authentication fails. Always use it with HTTPS in production, and do not hard-code credentials.

### Replacing the Authentication Mechanism

`authenticator` accepts a synchronous or asynchronous callable that takes one `Request`. Return any value after successful authentication. To reject a request, raise `HTTPException` or return `False`.

```python
from fastapi import HTTPException, Request


async def authenticate_admin(request: Request):
    # In a real application, validate an SSO SDK session, a session store,
    # or headers supplied by an authenticated reverse proxy.
    user = await sso_session.get_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


setup_admin_panel(
    app,
    adapter=adapter,
    authenticator=authenticate_admin,
)
```

SSO login initiation and callback handling should normally live in application middleware, dedicated routes, or an authentication proxy. The Admin authenticator should validate the established session and authorization. This boundary allows the authentication mechanism to change without modifying the Admin UI or individual APIs.

Passing `authenticator=None` exposes the Admin Panel without authentication. Always configure an authenticator outside local development.

## API

All endpoints are under `/admin/api`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/capabilities` | Return available optional features |
| GET | `/metrics/by-channel?period=24h&interval=1h` | Return Pipeline and Speech metrics grouped by channel |
| GET | `/metrics/summary?period=24h` | Return the Metrics summary |
| GET | `/metrics/timeline?period=24h&interval=1h` | Return the detailed phase timeline |
| GET | `/logs` | Return conversation messages matching the filters |
| GET | `/logs/voice/{transaction_id}/{voice_type}` | Return WAV audio or a Response audio count |
| GET | `/config/runtime` | Describe safe editable members of the running object graph |
| POST | `/config/runtime/{section}` | Apply volatile member changes to one component |
| POST | `/evaluate` | Start an Evaluation |
| GET | `/evaluate/{evaluation_id}` | Retrieve Evaluation results |

Config POST requests use the common body shape `{"config": {...}}`.

Metrics and Logs queries currently support `SQLitePerformanceRecorder` and `PostgreSQLPerformanceRecorder`.

## Component Responsibilities

### Python

| File | Responsibility |
| --- | --- |
| `__init__.py` | Build the `/admin` router, serve the page and static assets, register APIs, and provide `AdminPanel` |
| `auth.py` | Provide the replaceable authentication boundary and Basic authentication |
| `metrics.py` | Define Metrics API request and response models and invoke the query layer |
| `logs.py` | Provide the Logs and audio APIs and their response models |
| `config/` | Describe and update safe members of the running object graph |
| `evaluation.py` | Start Evaluations in the background and retrieve their results |

Database-specific behavior, timestamp selection, latency calculation, log filtering, and grouping are centralized in `aiavatar.sts.performance_recorder.query`. The Admin API remains a thin HTTP layer over those query results.

### Frontend

| File | Responsibility |
| --- | --- |
| `static/index.html` | Provide the shared layout and load Chart.js and the application entry point |
| `static/admin-app.js` | Load capabilities and manage navigation, view lifecycles, and global status |
| `static/admin-api.js` | Provide the same-origin HTTP client for `/admin/api/` |
| `static/metrics-view.js` | Render overall counts and per-channel Pipeline and Speech charts |
| `static/logs-view.js` | Render filters, the log table, details drawer, audio playback, and per-turn charts |
| `static/config-view.js` | Load and arrange Config components |
| `static/config-panel.js` | Generate configuration fields and save their values |
| `static/evaluation-view.js` | Start Evaluations and poll for results |
| `static/theme.js` | Initialize, switch, and save the Light/Dark theme and notify charts of changes |
| `static/admin.tailwind.css` | Contain the Tailwind CSS source and Admin-specific styles |
| `static/admin.css` | Contain the generated CSS used at runtime |

The frontend uses Vanilla JavaScript ES modules. TypeScript, Node.js, and a frontend build environment are not required at runtime. Chart.js is loaded from a CDN by `index.html`, so the browser must be able to reach that CDN to display the Metrics chart.

## Theme and CSS Updates

The initial Light/Dark theme follows the operating system setting. A manual selection is saved in `localStorage` under `aiavatar-admin-theme`. Changing the theme also updates the Chart.js colors on the Metrics view.

The distributed package already contains the generated `admin.css`; no CSS build step is required at runtime. When developing the Admin Panel from a repository checkout, regenerate the CSS after changing `admin.tailwind.css` by running the following command from the repository root. Tailwind CSS standalone CLI v4.3.3 is required, but Node.js and `node_modules` are not.

```bash
tailwindcss -i aiavatar/admin/static/admin.tailwind.css -o aiavatar/admin/static/admin.css --minify
```

If the CLI is not on `PATH`, invoke it by its path:

```bash
/path/to/tailwindcss -i aiavatar/admin/static/admin.tailwind.css -o aiavatar/admin/static/admin.css --minify
```

## Legacy Admin Panel

The previous implementation is preserved with its UI and APIs in `aiavatar.admin_legacy`. Select either the new or legacy package; an application would normally install only one of them.

```python
from aiavatar.admin_legacy import setup_admin_panel
```

The new Admin Panel does not retain the legacy API paths, API-key authentication, Character, or Control features. `admin_legacy` is a preserved implementation for applications that need to continue using the previous Admin Panel, not a compatibility layer for migrating to the new one.
