# Database

AIAvatarKit keeps several kinds of state outside the process: conversation history, session
state, per-turn performance records, registered speakers, and the mapping that lets one
person be recognised across channels. Most database-backed state has both SQLite and
PostgreSQL implementations. The speaker registry is different: it uses `InMemoryStore`,
optionally persisted to files, or the PostgreSQL-backed `PGVectorStore`.

SQLite is the default for database-backed components and needs no setup. Most use
`aiavatar.db` unless you say otherwise; `SQLiteChannelContextBridge` defaults to
`channel_context_bridge.db`, and speaker data is not stored in SQLite. Move to PostgreSQL
when you run multiple processes, or want the records somewhere your other tooling can reach.

## What is stored where

| Concern | Interface | SQLite | PostgreSQL |
| --- | --- | --- | --- |
| Conversation context | `ContextManager` | `SQLiteContextManager` | `PostgreSQLContextManager` |
| Session state | `SessionStateManager` | `SQLiteSessionStateManager` | `PostgreSQLSessionStateManager` |
| Performance records | `PerformanceRecorder` | `SQLitePerformanceRecorder` | `PostgreSQLPerformanceRecorder` |
| Character data | `CharacterRepositoryBase` | `SQLiteCharacterRepository` | `PostgreSQLCharacterRepository` |
| Character activity | `ActivityRepositoryBase` | `SQLiteActivityRepository` | `PostgreSQLActivityRepository` |
| Users | `UserRepository` | `SQLiteUserRepository` | `PostgreSQLUserRepository` |
| Speaker registry | `BaseSpeakerStore` | `InMemoryStore` with optional file persistence | `PGVectorStore` (requires `pgvector`) |
| Channel context bridge | `ChannelContextBridge` | `SQLiteChannelContextBridge` | `PostgreSQLChannelContextBridge` |
| Responses API response ids | `ResponseIdStore` | `SQLiteResponseIdStore` | `PostgreSQLResponseIdStore` |

### Conversation context

`SQLiteContextManager(db_path="aiavatar.db", context_timeout=3600)` stores the message
history the LLM sees. `context_timeout` is how long a `context_id` stays usable before a new
conversation is started; raise it for assistants that should remember across a working day.

The OpenAI Responses API keeps its context on the server through `previous_response_id`.
AIAvatarKit's Responses services persist those ids in a `ResponseIdStore` and also save
conversation history through a `ContextManager` for local reference.

### Session state

`SQLiteSessionStateManager(db_path="aiavatar.db", session_timeout=3600, cache_ttl=60)` holds
per-session values that must survive across turns but do not belong in the conversation. It
keeps an in-memory cache in front of the database, refreshed every `cache_ttl` seconds.

### Speaker registry

`SpeakerRegistry(match_threshold=0.72, store=...)` matches a voice embedding against
registered speakers. Use the default `InMemoryStore`, which persists to a file via
`data_path`. See [Speech-to-Text](stt.md) for how matching is wired into recognition.

`InMemoryStore` and `PGVectorStore` both implement the asynchronous `BaseSpeakerStore`
interface. `PGVectorStore` requires PostgreSQL's `vector` extension and can use a shared
connection pool through `get_pool`.

### Channel context bridge

`SQLiteChannelContextBridge(db_path="channel_context_bridge.db", timeout=3600)` maps each
`(channel_id, channel_user_id)` pair to an application-level `user_id`, and remembers the
latest `context_id` for that person. That is what lets someone hang up the phone, open LINE,
and continue the same conversation. `timeout` bounds how long the continuation is offered.
See [Adapters](adapters.md) for the wiring.

## Changing the SQLite database path

When you create the LLM separately, pass the same `db_connection_str` to the LLM service and `STSPipeline`:
the LLM creates its conversation context manager, and the pipeline creates its session
state manager and performance recorder. You do not need to instantiate these database
components individually.

With your existing `vad`, `stt`, and `tts` components:

```python
from aiavatar.sts import STSPipeline
from aiavatar.sts.llm.chatgpt import ChatGPTService

DB_CONNECTION_STR = "myapp.db"

llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    db_connection_str=DB_CONNECTION_STR,
    # With OpenAIResponsesService or OpenAIResponsesWebSocketService, this also
    # configures the automatically created ResponseIdStore to use the same database.
)

sts = STSPipeline(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    db_connection_str=DB_CONNECTION_STR,
)
```

If the pipeline creates the default LLM for you, setting `db_connection_str` on the
pipeline alone is enough. An adapter that creates its pipeline internally also accepts
this argument and forwards it to the pipeline. When you supply an existing `llm`, `sts`,
or database component, its database settings are preserved; configure it when you create it.

If you also use a channel context bridge, set its path separately:

```python
from aiavatar.adapter.channel_context_bridge import SQLiteChannelContextBridge

bridge = SQLiteChannelContextBridge(db_path=DB_CONNECTION_STR)
```

Pass or bind the bridge to your adapters as shown in [Adapters](adapters.md#sharing-context-across-channels).
It can use the same database file or a separate one. Relative paths are resolved from the
process's working directory; create any parent directory before constructing the components.
Changing the path selects a different file and does not move existing data.

## Using PostgreSQL

Install `asyncpg` to use PostgreSQL instead of SQLite:

```sh
pip install asyncpg
```

You can pass a `postgresql://...` connection string as `db_connection_str` to the LLM and
pipeline in the same way as the SQLite example above. They automatically create the
PostgreSQL components and connection pools. However, independently created components do
not share pools just because their connection strings match. For Responses services, the
automatically created context manager and response ID store also have separate pools.

To share a pool and control its connection limits, create a `PostgreSQLPoolProvider` and
pass its `get_pool` to the LLM's context manager, and the provider itself to the pipeline
or an adapter that creates the pipeline:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.database.postgres import PostgreSQLPoolProvider
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.llm.context_manager.postgres import PostgreSQLContextManager

# DB_CONNECTION_STR = "postgresql://{user}:{password}@{host}:{port}/{databasename}"
DB_CONNECTION_STR = "postgresql://postgres:postgres@127.0.0.1:5432/aiavatar"

# PoolProvider
pool_provider = PostgreSQLPoolProvider(
    connection_str=DB_CONNECTION_STR,
    # max_size=20,  # Max connection count (default: 20)
    # min_size=5    # Min connection count (default: 5)
)

# LLM
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt=SYSTEM_PROMPT,
    context_manager=PostgreSQLContextManager(
        get_pool=pool_provider.get_pool # Set `get_pool` to PostgreSQLContextManager
    )
)

# Adapter (Create pipeline internally)
ws_app = AIAvatarWebSocketServer(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    db_pool_provider=pool_provider,     # Creates PostgreSQLSessionStateManager and PostgreSQLPerformanceRecorder internally
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await pool_provider.close()

app = FastAPI(lifespan=lifespan)
app.include_router(ws_app.get_websocket_router())
```

For a Responses service, also pass
`response_id_store=PostgreSQLResponseIdStore(get_pool=pool_provider.get_pool)` to the LLM;
import it from `aiavatar.sts.llm.response_id_store.postgres`. Supplying only the context
manager does not configure the response ID store. An optional
`PostgreSQLChannelContextBridge` from `aiavatar.adapter.channel_context_bridge.postgres`
can use the same `get_pool=pool_provider.get_pool`.

**NOTE**: `PerformanceRecorder` runs in a separate thread from the main thread, so it does not use the shared connection pool. Instead, it retrieves only the connection information from the PoolProvider and creates its own dedicated connection pool. It writes performance information serially as it receives it through a queue, so it basically uses only a single connection. We recommend not changing this unless you have a specific reason.

## Choosing a backend

Use SQLite for development, single-process deployments, and edge devices. Move to
PostgreSQL when any of the following is true:

- More than one worker process serves the same conversations. SQLite's locking will bite.
- Performance records are consumed by dashboards or queries outside the application.
- Your deployment already treats the filesystem as ephemeral.

For components that provide both SQLite and PostgreSQL implementations, switching is usually
a constructor change plus `PoolProvider` wiring. The speaker registry instead switches
between `InMemoryStore` and `PGVectorStore`.

## See also

- [Pipeline](pipeline.md) — what produces performance records and session state
- [Speech-to-Text](stt.md) — the speaker registry in use
- [Adapters](adapters.md) — sharing context across channels
- [Character](character.md) — character data storage

---

[← Documentation index](../README.md#-documentation)
