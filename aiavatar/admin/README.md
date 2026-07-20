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

The Metrics view shows the latency from the end of the user's speech to the first response.

- Period: `1h`, `6h`, `24h`, `7d`, or `30d`
- Interval: `1m`, `5m`, `15m`, `1h`, or `1d`
- Summary: request count, error count, average, median, and P95
- Chart: stacked average latency by phase for each time bucket

The detailed breakdown contains nine phases:

1. Silence detection
2. Streaming STT finalization
3. Turn-end gate
4. STT
5. Stop current response
6. Before-LLM handlers
7. LLM
8. Processing
9. TTS

The endpoint for a normal response is the first TTS audio chunk. The endpoint for a Quick Response is `before_llm_time`. When an intermediate timing point is absent in an older record, its unknown interval is included in the next known phase so that the stack remains equal to the measured first-response time.

The request count includes every record in the selected period. The detailed breakdown uses only records that meet all of the following conditions, and the UI reports this subset as coverage:

- `speech_end_at` is present
- The request has no error
- A Quick Response or TTS first-chunk endpoint is present
- The VAD and timing data can be interpreted

### Logs

The Logs view groups conversation messages by `context_id`. The table displays Context ID, message count, period, and User ID. Selecting a row opens the turn details in a drawer on the right.

The following filters are available. All specified conditions are combined with AND.

- User ID: exact match
- Session ID: exact match
- Context ID: exact match
- Keyword: searches request, response, voice response, quick response, error, and tool-call data
- Error presence
- Limit: 1–10,000; default 200

The limit applies to the number of messages retrieved. The messages are then grouped by `context_id`.

The drawer displays:

- Timestamp, Session ID, User ID, Context ID, and Transaction ID
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

Input fields are generated dynamically from the values returned by each component's `get_config()`. Saving calls its `set_config()`. These operations update live objects and do not persist values to configuration files. Applications that need persistence must manage it separately.

### Evaluation

The Evaluation view accepts a JSON array of scenarios and runs Dialog Evaluation in the background. Results are saved to `evaluation_results/<evaluation_id>.json`, relative to the process working directory.

The Evaluation tab is shown only when an Evaluator is available. An Evaluator is created automatically when the Adapter uses `ChatGPTService` as its LLM. For other LLMs, pass a `DialogEvaluator` through `setup_admin_panel(..., evaluator=...)`.

Character and Control UI and APIs are not included in this Admin Panel.

## Time Semantics

Metrics period selection and bucketing, as well as Logs display and ordering, use the same event timestamp:

```text
event_at = speech_end_at ?? created_at
```

`speech_end_at` is the time at which the user's speech ended. `created_at`, which represents record persistence time, is used only as a fallback for older records without `speech_end_at`.

For compatibility, the Logs API still returns this value in a field named `created_at`, but its value is the `event_at` defined above. Records without `speech_end_at` can appear in the log list, but they cannot be included in a detailed breakdown measured from the end of the user's speech.

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
| GET | `/metrics/summary?period=24h` | Return the Metrics summary |
| GET | `/metrics/timeline?period=24h&interval=1h` | Return the detailed phase timeline |
| GET | `/logs` | Return conversation messages matching the filters |
| GET | `/logs/voice/{transaction_id}/{voice_type}` | Return WAV audio or a Response audio count |
| GET/POST | `/config/pipeline` | Read or update Pipeline configuration |
| GET/POST | `/config/vad` | Read or update VAD configuration |
| GET/POST | `/config/stt` | Read or update STT configuration |
| GET/POST | `/config/llm` | Read or update LLM configuration |
| GET/POST | `/config/tts` | Read or update TTS configuration |
| GET | `/config/adapters` | List Adapters and their configuration |
| GET/POST | `/config/adapter/{name}` | Read or update one Adapter's configuration |
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
| `config/` | Provide configuration APIs for Pipeline, Adapter, VAD, STT, LLM, and TTS |
| `evaluation.py` | Start Evaluations in the background and retrieve their results |

Database-specific behavior, timestamp selection, latency calculation, log filtering, and grouping are centralized in `aiavatar.sts.performance_recorder.query`. The Admin API remains a thin HTTP layer over those query results.

### Frontend

| File | Responsibility |
| --- | --- |
| `static/index.html` | Provide the shared layout and load Chart.js and the application entry point |
| `static/admin-app.js` | Load capabilities and manage navigation, view lifecycles, and global status |
| `static/admin-api.js` | Provide the same-origin HTTP client for `/admin/api/` |
| `static/metrics-view.js` | Render Metrics summary cards and the Chart.js chart |
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
