from typing import Any
from ..adapter import Adapter, AIAvatarRequest, AIAvatarResponse
from ..sts.models import STSResponse
from .tools import UpdateUsernameTool, GetDiaryTool, MemorySearchTool
from . import CharacterService


def bind_character(
    *,
    adapter: Adapter,
    character_service: CharacterService,
    character_id: str,
    default_user_name: str
):
    @adapter.on_session_start
    async def on_session_start(request: AIAvatarRequest, data: Any):
        if not await character_service.user.get(user_id=request.user_id):
            user = await character_service.user.create(name=default_user_name)
            request.user_id = user.id

    @adapter.on_response
    async def on_response(a_resp: AIAvatarResponse, sts_resp: STSResponse):
        if a_resp.type == "connected":
            # Send username and character name on connected
            user = await character_service.user.get(user_id=a_resp.user_id)
            a_resp.metadata["username"] = user.name
            character = await character_service.character.get(character_id=character_id)
            a_resp.metadata["charactername"] = character.name
        elif a_resp.type == "tool_call" and sts_resp.tool_call.name == "update_username":
            # Also send username on changed
            if username := sts_resp.tool_call.result.data.get("username"):
                a_resp.metadata["username"] = username

    @adapter.sts.llm.get_system_prompt
    async def get_system_prompt(context_id: str, user_id: str, system_prompt_params: dict):
        if not system_prompt_params:
            system_prompt_params = {}
        user = await character_service.user.get(user_id=user_id)
        system_prompt_params["username"] = user.name
        return await character_service.get_system_prompt(
            character_id=character_id,
            system_prompt_params=system_prompt_params
        )

    adapter.sts.llm.add_tool(
        UpdateUsernameTool(
            character_service=character_service,
            debug=character_service.debug
        )
    )

    adapter.sts.llm.add_tool(
        GetDiaryTool(
            character_service=character_service,
            character_id=character_id,
            include_schedule=True,
            debug=character_service.debug
        )
    )

    if character_service.memory:
        adapter.sts.llm.add_tool(
            MemorySearchTool(
                memory_client=character_service.memory,
                character_id=character_id,
                debug=True
            )
        )
