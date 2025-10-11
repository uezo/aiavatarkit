import asyncio
import logging
from typing import List, Dict
from fastmcp import Client  # pip install fastmcp
from fastmcp.client.transports import ClientTransport, PythonStdioTransport, NodeStdioTransport, StreamableHttpTransport
from .. import Tool

logger = logging.getLogger(__name__)


class MCPBase:
    def __init__(self, *, client: Client = None, transport: ClientTransport = None, debug: bool = False):
        if not client and not transport:
            raise ValueError("Either 'client' or 'transport' must be provided")

        self.client = client or Client(transport=transport)
        self.tools: List[Tool] = None
        self._for_each_tool= None
        self.debug = debug

    @property
    def for_each_tool(self):
        # Usase(decorator): @mcp.for_each_tool
        def _decorator(func):
            self._for_each_tool = func
            return func
        return _decorator

    @for_each_tool.setter
    def for_each_tool(self, func):
        # Usase(set directly): mcp.for_each_tool = func
        self._for_each_tool = func

    async def initialize(self) -> List[Tool]:
        await self.client.__aenter__()

        self.tools = []
        mcp_tools = await self.client.list_tools()
        if self.debug:
            logger.info(f"MCP Tools: {mcp_tools}")

        for mcp_tool in mcp_tools:
            async def call_tool(tool_name=mcp_tool.name, **kwargs):
                results = await self.client.call_tool(tool_name, kwargs)
                dumped_results = [r.text if r.type == "text" else r.model_dump() for r in results]
                if len(results) == 1:
                    return dumped_results[0]
                else:
                    return dumped_results

            self.tools.append(Tool(
                name=mcp_tool.name,
                spec={
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description,
                        "parameters": mcp_tool.inputSchema
                    }
                },
                func=call_tool
            ))

        if self._for_each_tool:
            for t in self.tools:
                try:
                    self._for_each_tool(t)
                except Exception as ex:
                    logger.exception(f"Error at handling tool {t.name}: {ex}")

        return self.tools

    async def close(self):
        try:
            await self.client.__aexit__(None, None, None)
        except (RuntimeError, asyncio.CancelledError) as e:
            if isinstance(e, RuntimeError) and "cancel scope" not in str(e):
                raise
        except Exception as e:
            if self.debug:
                logger.warning(f"Error closing {self.__class__.__name__} client: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


class StdioMCP(MCPBase):
    def __init__(self, *, server_script: str = None, client: Client = None, debug: bool = False):
        if client:
            transport = None
        else:
            if server_script.endswith(".py"):
                transport = PythonStdioTransport(script_path=server_script)
            elif server_script.endswith(".js"):
                transport = NodeStdioTransport(script_path=server_script)
        super().__init__(client=client, transport=transport, debug=debug)


class StreamableHttpMCP(MCPBase):
    def __init__(self, *, url: str = None, headers: Dict[str, str] = None, sse_read_timeout: float = None, client: Client = None, debug: bool = False):
        if client:
            transport = None
        else:
            transport = StreamableHttpTransport(url=url, headers=headers, sse_read_timeout=sse_read_timeout)
        super().__init__(client=client, transport=transport, debug=debug)
