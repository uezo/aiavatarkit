import logging
from datetime import date
from aiavatar.sts.llm import Tool
from .service import CharacterService
from .memory import MemoryClientBase

logger = logging.getLogger(__name__)


class GetDiaryTool(Tool):
    def __init__(
        self,
        *,
        character_service: CharacterService,
        character_id: str,
        include_schedule: bool = True,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.character_service = character_service
        self.character_id = character_id
        self.include_schedule = include_schedule
        self.debug = debug
        super().__init__(
            name=name or "get_diary",
            spec=spec or {
                "type": "function",
                "function": {
                    "name": name or "get_diary",
                    "description": "Retrieve a diary entry by specifying a date",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diary_date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                        },
                    },
                    "required": ["diary_date"]
                }
            },
            func=self.get_diary,
            instruction=instruction,
            is_dynamic=is_dynamic
        )

    async def get_diary(self, diary_date: str, metadata: dict = None):
        if not self.character_service.activity:
            await self.character_service.get_pool()

        contents = {}

        diary_date_obj = date.fromisoformat(diary_date)

        diary = await self.character_service.activity.get_diary(
            character_id=self.character_id,
            diary_date=diary_date_obj
        )
        if diary:
            contents["diary"] = diary.content

        if self.include_schedule:
            daily_schedule = await self.character_service.activity.get_daily_schedule(
                character_id=self.character_id,
                schedule_date=diary_date_obj
            )
            if daily_schedule:
                contents["schedule"] = daily_schedule.content

        if not contents:
            contents["message"] = "No entries."

        return contents


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


class UpdateUsernameTool(Tool):
    def __init__(
        self,
        *,
        character_service: CharacterService,
        name: str = None,
        spec: str = None,
        instruction: str = None,
        is_dynamic: bool = False,
        debug: bool = False
    ):
        self.character_service = character_service
        self.debug = debug

        super().__init__(
            name or "update_username",
            spec or {
                "type": "function",
                "function": {
                    "name": name or "update_username",
                    "description": "Update username for the specific user. Use this tool when the user says 'Remember my name' or something like that.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                        },
                        "required": ["username"]
                    },
                }
            },
            self.update_username,
            instruction,
            is_dynamic
        )

    async def update_username(self, username: str, metadata: dict):
        try:
            user_id = metadata["user_id"]
            await self.character_service.user.update(user_id=user_id, name=username)
            if self.debug:
                logger.info(f"Set username '{username}' for user_id: {user_id}")
            return {"username": username}
        except Exception as ex:
            logger.error(f"Error at updating username: {ex}")
            return {"result": "error"}
