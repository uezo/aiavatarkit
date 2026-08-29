# Long-term memory

Conversation context covers the current conversation. Long-term memory covers everything
before it — past conversations, facts about the user, knowledge you have loaded — and shared
context covers messages that should be visible to every session of a character at once.

## Recalling past conversations

To recall information from past conversations across different contexts, a long-term memory service is used.

To store conversation history, define a function decorated with `@aiavatar_app.sts.on_finish`. To retrieve memories from the conversation history, call the search function of the long-term memory service as a tool.

Below is an example using [ChatMemory](https://github.com/uezo/chatmemory).

```python
# Create client for ChatMemory
from aiavatar.character.memory import MemoryClient
memory_client = MemoryClient(
    base_url="http://localhost:8000"
)

# Add messages to ChatMemory service
@aiavatar_app.sts.on_finish
async def on_finish(request, response):
    await memory_client.add_messages(
        character_id=YOUR_CHARACTER_ID,  # Character ID registered via CharacterService, or any value to separate memory spaces
        request=request,
        response=response
    )

# Add MemorySearchTool to recall past events, conversations, or information about the user.
from aiavatar.character.tools import MemorySearchTool
llm.add_tool(
    MemorySearchTool(
        memory_client=memory_client,
        character_id=YOUR_CHARACTER_ID,
        debug=True
    )
)
```

## Shared Context

Context is typically shared only between an individual user and the AI character. With AIAvatarKit, you can manage histories that define how broadly the context is shared, for example, making it common to every user.

This lets you inject context with general events that are independent of any single user interaction, such as public news or actions the AI character has taken.

```python
# Add character-wide shared messages identified by context_id="shared_context_id"
now = datetime.now(ZoneInfo(self.timezone))
await self.llm.context_manager.add_histories(
    context_id="shared_context_id",
    data_list=[
        {
            "role": "user",
            "content": f"$Current datetime: {now.strftime('%Y/%m/%d %H:%M:%S')}\nToday's news: {news}"
        },
        {
            "role": "assistant",
            "content": "I recognized current datetime and today's news."
        },
    ],
    context_schema="chatgpt"
)
```

```python
# Pass "shared_context_id" via `shared_context_ids` to load the shared history
llm = ChatGPTService(
    openai_api_key=OPENAI_API_KEY,
    system_prompt="You are a helpful virtual assistant.",
    shared_context_ids=["shared_context_id"]
)
```

## See also

- [Character](character.md) — diaries as a source of memory
- [Tools](tools.md) — how memory search reaches the model
- [Database](database.md) — persistence

---

[← Documentation index](../README.md#-documentation)
