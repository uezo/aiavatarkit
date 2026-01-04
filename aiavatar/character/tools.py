import logging
from aiavatar.sts.llm import Tool
from .memory import MemoryClientBase

logger = logging.getLogger(__name__)


class MemorySearchTool(Tool):
    def __init__(
        self,
        *,
        memory_client: MemoryClientBase,
        character_id: str,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.memory_client = memory_client
        self.character_id = character_id
        self.debug = debug
        super().__init__(
            name=name or "search_memory",
            spec=spec or {
                "type": "function",
                "function": {
                    "name": name or "search_memory",
                    "description": "Search long-term memory when you need to recall past events, conversations, or information about the user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                    },
                }
            },
            func=self.search_memory,
            instruction=instruction,
            is_dynamic=is_dynamic
        )

    async def search_memory(self, query: str, metadata: dict = None):
        if self.debug:
            logger.info(f"Query for search_memory: {query}")

        result = await self.memory_client.search(
            character_id=self.character_id,
            user_id=metadata["user_id"],
            query=query
        )

        if self.debug:
            logger.info(f"Result from search_memory: {result}")

        return result.model_dump()
