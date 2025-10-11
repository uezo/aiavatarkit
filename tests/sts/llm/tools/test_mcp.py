import pytest
import json
import os
from uuid import uuid4
from fastmcp import Client
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.llm.tools.mcp import StdioMCP

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@pytest.mark.asyncio
async def test_mcp():
    async with StdioMCP(server_script="tests/sts/llm/tools/mcpserver.py") as mcp:
        assert isinstance(mcp.client, Client)

        tools = await mcp.initialize()
        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "get_alert"

        result_text = await tools[0].func(location="Tokyo")
        result = json.loads(result_text)
        assert result["location"] == "Tokyo"
        assert result["weahter"] == "clear"
        assert result["temperature"] == 32.1

        result_text = await tools[1].func(location="Tokyo")
        result = json.loads(result_text)
        assert result["location"] == "Tokyo"
        assert result["alert"] == "Thunderstorm"

@pytest.mark.asyncio
async def test_mcp_llm():
    async with StdioMCP(server_script="tests/sts/llm/tools/mcpserver.py") as mcp:
        llm = ChatGPTService(openai_api_key=OPENAI_API_KEY)
        mcp.for_each_tool = llm.add_tool
        assert len(llm.tools) == 0

        await mcp.initialize()
        assert len(llm.tools) == 2

        assert llm.tools["get_weather"].name == "get_weather"
        assert llm.tools["get_alert"].name == "get_alert"

        resp = ""
        async for chunk in llm.chat_stream(
            context_id=f"context_mcp_tool_{uuid4()}",
            user_id="user01", text="What is the weather and temperture in New York?"
        ):
            if chunk.text:
                resp += chunk.text

        assert "clear" in resp.lower()
        assert "32" in resp
