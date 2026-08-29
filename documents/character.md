# Character

`CharacterService` manages an AI character's settings and generates the content that makes
it feel like it exists in real time: a weekly schedule, today's schedule, and diaries
written as if by the character. Feed those into the conversation daily and the character
stops being a stateless chatbot.

For a lighter setup with no database, see CharacterLoader at the end of this page.

Schedules and diaries are generated as if by the character's own will. By updating these
daily and feeding them into the conversation, you can make the character feel like it is
actually living in real-world time.

Storage follows the same rule as everything else: **SQLite by default**, in `aiavatar.db`.
Pass a `db_pool_provider`, or a `db_connection_str` beginning with `postgresql://`, to use
PostgreSQL instead. See [Database](database.md).


## Get started

Register a new character using a character setting prompt. At this time, both the weekly schedule and today's schedule are also generated.

```python
from datetime import date
from aiavatar.character import CharacterService

# Initialize service
character_service = CharacterService(
    openai_api_key="YOUR_API_KEY"
)

# Initialize a new character with weekly and daily schedules
character, weekly, daily = await character_service.initialize_character(
    name="Alice",
    character_prompt="You are Alice, a cheerful high school student who loves reading..."
)

print(f"Character ID: {character.id}")
```

The character reaches the model through **two** hooks, and they carry different things.

`get_system_prompt` supplies the character settings — the `character_prompt`, expression and
language rules, output constraints:

```python
@llm.get_system_prompt
async def get_system_prompt(context_id: str, user_id: str, system_prompt_params: dict):
    return await character_service.get_system_prompt(
        character_id="YOUR_CHARACTER_ID",
        system_prompt_params=system_prompt_params
    )
```

`get_initial_messages` supplies everything that changes: the user's name, episodes,
attributes, conversation examples, and **today's schedule**. These are injected as a
few-shot exchange at the head of the context rather than as system text.

```python
@llm.get_initial_messages
async def get_initial_messages(context_id: str, user_id: str, system_prompt_params: dict):
    return await character_service.get_initial_messages(
        character_id="YOUR_CHARACTER_ID",
        user_id=user_id,
        system_prompt_params=system_prompt_params
    )
```

Wiring only `get_system_prompt` gives you a character with the right personality that has no
idea what it is supposed to be doing today. If the schedule matters to you, register both.

Note that `get_initial_messages` needs the `user_id` — it looks up the user's name to build
the self-introduction turn — so it cannot be built once and cached like the system prompt.

It also generates today's schedule on demand when none exists yet. Pass
`generate_schedule=False` to skip that and fall back to "no schedule information" instead.

[`bind_character`](#binding-to-adapter) registers both hooks for you, along with the
character tools; the pair above is what it does internally.


## Updating Diaries

Diaries can be automatically generated using `create_diary_with_generation`. The following information is used:

- Character settings
- Today's schedule
- Today's news (retrieved via web search)
- Previous day's diary

```python
# Generate diary from daily activities
diary = await character_service.create_diary_with_generation(
    character_id=character.id,
    diary_date=date.today()
)
```

The generated diary can be used as context for the LLM using `GetDiaryTool`. By setting `include_schedule=True`, the schedule information for the day is also retrieved (default is `True`).

```python
from aiavatar.character.tools import GetDiaryTool
llm.add_tool(
    GetDiaryTool(
        character_service=character_service,
        character_id=YOUR_CHARACTER_ID,
        include_schedule=True
    )
)
```


## Updating Schedules

Daily schedules can be automatically generated using `create_daily_schedule_with_generation`. The following information is used:

- Character settings
- Weekly schedule
- Previous day's schedule

```python
daily_schedule = await character_service.create_daily_schedule_with_generation(
    character_id=character.id,
    schedule_date=date.today()
)
```

## Automated Daily Updates

For a more realistic character experience, use a scheduler service (such as cron) to automatically update schedules and diaries:

- **Daily schedule**: Generate at the beginning of each day (e.g., 0:00 or 6:00)
- **Diary**: Generate at the end of each day (e.g., 23:00)

Example cron configuration:

```
# Generate daily schedule at 6:00 AM
0 6 * * * /usr/bin/python3 /path/to/generate_schedule.py

# Generate diary at 11:00 PM
0 23 * * * /usr/bin/python3 /path/to/generate_diary.py
```

Example script for `generate_schedule.py`:

```python
import asyncio
from datetime import date
from aiavatar.character import CharacterService

async def main():
    character_service = CharacterService(
        openai_api_key="YOUR_API_KEY"
    )
    await character_service.create_daily_schedule_with_generation(
        character_id="YOUR_CHARACTER_ID",
        schedule_date=date.today()
    )

asyncio.run(main())
```

## Batch Generation

You can batch generate daily schedules and diaries for a date range using `create_activity_range_with_generation`.

```python
await character_service.create_activity_range_with_generation(
    character_id=YOUR_CHARACTER_ID,
    start_date=date(2026, 1, 8),
    end_date=date(2026, 1, 16),  # Defaults to today if omitted
    overwrite=False,
)
```

This is useful for recovering data when automatic updates were stopped, or for building up initial data when creating a new character.

## Long-term Memory

This feature is **optional**. If you want to make diaries searchable as long-term memory, you can integrate with an external memory service by configuring `MemoryClient`:

```python
from aiavatar.character import CharacterService, MemoryClient

memory_client = MemoryClient(base_url="http://memory-service:8000")

character_service = CharacterService(
    openai_api_key="YOUR_API_KEY",
    memory_client=memory_client
)
```

Registered diaries can be included in search results using the `search` method.

```python
# In addition to diaries, conversation history with users and other knowledge are searched comprehensively
result = await character_service.memory.search(
    character_id="YOUR_CHARACTER_ID",
    user_id="YOUR_USER_ID",
    query="travel summer 2026"
)
```

The default `MemoryClient` uses [ChatMemory](https://github.com/uezo/chatmemory) as its backend, but you can also use other long-term memory services by inheriting from `MemoryClientBase`.


## Binding to Adapter

The `bind_character` function provides a convenient way to integrate character management with your AIAvatar application. It automatically configures the system prompt, user management, and character-related tools in a single call.

```python
from aiavatar.character import CharacterService
from aiavatar.character.binding import bind_character

character_service = CharacterService(
    openai_api_key="YOUR_API_KEY"
)

bind_character(
    adapter=aiavatar_app,
    character_service=character_service,
    character_id="YOUR_CHARACTER_ID",
    default_user_name="You"
)
```

This single function call sets up:

- **System prompt**: Automatically retrieves the character's system prompt with user-specific parameters
- **User management**: Creates a new user with `default_user_name` if the user doesn't exist
- **Username sync**: Sends the username and character name to the client on connection, and updates when changed
- **Tools**: Registers the following tools automatically:
  - `UpdateUsernameTool`: Allows the character to update the user's name during conversation
  - `GetDiaryTool`: Retrieves the character's diary and schedule
  - `MemorySearchTool`: Searches long-term memory (only if `memory_client` is configured)


## CharacterLoader (Lightweight Alternative)

`CharacterLoader` is a lightweight alternative to `CharacterService` that loads character settings from local files instead of a database. No database or external API is required — just plain markdown and JSON files.

This is ideal when you want to quickly set up a character without infrastructure, or when you prefer to manage character definitions as files.

### Single file mode

The simplest usage is to point to a single markdown file containing the system prompt:

```python
from aiavatar.character.loader import CharacterLoader

loader = CharacterLoader("system_prompt.md")

# Bind to LLM service
loader.bind(adapter.sts.llm)
```

### Directory mode

For richer character definitions, use directory mode with `split_initial_messages=True`. Initial messages are prepended to the conversation history as pseudo user/assistant turns, allowing you to inject character knowledge (episodes, attributes, conversation examples) without overloading the system prompt. Point to a directory containing:

```
my_character/
├── character.md                # Character settings (required with split_initial_messages)
├── response_instructions.md    # Response rules (optional, appended to system prompt)
├── message_templates.json      # Template definitions for initial messages
├── episode.md                  # Character's past experiences (optional)
├── attribute.md                # Likes, dislikes, personality traits (optional)
└── conversation_example.md     # Example dialogues for tone reference (optional)
```

```python
loader = CharacterLoader(
    "my_character",
    split_initial_messages=True,
    lang="ja",
    user_names={"user_001": "Alice"},
    default_user_name="You"
)

loader.bind(adapter.sts.llm)
```

The `message_templates.json` defines how initial messages and self-introduction are structured:

```json
{
    "initial_message_defs": {
        "ja": {
            "self_intro": "わかりました。{username}さんですね。",
            "episode": "わかりました。",
            "attribute": "わかりました。",
            "conversation_example": "わかりました。"
        }
    },
    "prefixes": {
        "ja": {
            "episode": "以下はあなたの過去の経験です。\n\n",
            "attribute": "以下はあなたの属性情報です。\n\n",
            "conversation_example": "以下は会話例です。口調やトーンの参考にしてください。\n\n"
        }
    },
    "self_intro_template": {
        "ja": "$ユーザーの名前は{username}です。"
    }
}
```

`CharacterLoader` loads `<key>.md` for each key in `initial_message_defs` except
`self_intro`. Therefore, a file such as `conversation_example.md` must also have a
`conversation_example` entry in `initial_message_defs`.

### Hot reload

All files are cached with mtime-based invalidation. Edit any file while the application is running, and changes will be reflected on the next request — no restart needed.

### Custom user name resolution

Use the `@loader.get_user_name` decorator to resolve user names dynamically (e.g., from a database or external service):

```python
@loader.get_user_name
def get_user_name(user_id: str):
    return db.get_username(user_id)
```

### Custom message formatting

Use the `@loader.format_messages` decorator to post-process initial messages before they are sent to the LLM:

```python
@loader.format_messages
def format_messages(messages):
    # Add timestamps, filter messages, etc.
    return messages
```

### Comparison with CharacterService

| | CharacterLoader | CharacterService |
|---|---|---|
| Data source | Local files (`.md`, `.json`) | Database (SQLite / PostgreSQL) |
| Dependencies | None (standard library only) | `openai`, database libraries |
| Schedule / Diary generation | Not supported | Auto-generated via LLM |
| Long-term memory | Not supported | Supported via MemoryClient |
| Character tools | Not included | username update, diary, memory search |
| Hot reload | Supported (mtime-based) | Not supported |

## See also

- [Long-term memory](memory.md) — recalling past conversations
- [LLM](llm.md) — supplying the system prompt
- [Database](database.md) — where character data is stored

---

[← Documentation index](../README.md#-documentation)
