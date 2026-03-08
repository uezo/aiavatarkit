import asyncio
import logging
from typing import Callable
import openai
from .. import Tool

logger = logging.getLogger(__name__)


class OpenClawTool(Tool):
    def __init__(
        self,
        *,
        openclaw_api_key: str,
        openclaw_base_url: str = None,
        openclaw_session_key: str = None,
        on_openclaw_response: Callable = None,
        immediate_message: str = "Accepted. You will be notified when the response is ready.",
        timeout: int = 30000,
        name=None,
        spec=None,
        instruction=None,
        is_dynamic=False,
        debug: bool = False,
    ):
        self.openai_client = openai.AsyncClient(
            api_key=openclaw_api_key,
            base_url=openclaw_base_url,
            timeout=timeout
        )
        self.openclaw_session_key = openclaw_session_key
        self._on_openclaw_response = on_openclaw_response
        self.immediate_message = immediate_message
        self.debug = debug
        # Hold references to background tasks to prevent GC from cancelling them mid-execution
        self._background_tasks: set = set()

        super().__init__(
            name or "send_query_to_openclaw",
            spec or {
                "type": "function",
                "function": {
                    "name": name or "send_query_to_openclaw",
                    "description": "Invoke OpenClaw, a versatile AI agent capable of performing a wide range of tasks autonomously, including web search, information retrieval, data analysis, code execution, file operations, and complex multi-step reasoning. Use this tool when the user's request requires actions beyond your built-in capabilities, such as accessing real-time information, interacting with external services, or executing tasks that need autonomous planning and execution. Pass the user's request or a clear description of the task as the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    },
                }
            },
            self.invoke_openclaw,
            instruction,
            is_dynamic
        )

    def on_openclaw_response(self, func: Callable):
        self._on_openclaw_response = func
        return func

    async def _call_openclaw_api(self, query: str) -> str:
        if self.debug:
            logger.info(f"Request to OpenClaw: {query}")
        resp = await self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="openclaw",
            extra_headers={
                "x-openclaw-session-key": self.openclaw_session_key or "agent:main:main"
            }
        )
        answer = resp.choices[0].message.content
        if self.debug:
            logger.info(f"Response from OpenClaw: {answer}")
        return answer

    async def _invoke_in_background(self, query: str, metadata: dict):
        try:
            answer = await self._call_openclaw_api(query)
            await self._on_openclaw_response(query, answer, metadata)
        except Exception:
            logger.exception("Error at invoke_openclaw (background)")
            await self._on_openclaw_response(query, "Error", metadata)

    async def invoke_openclaw(self, query: str, metadata: dict = None):
        if self._on_openclaw_response:
            task = asyncio.create_task(self._invoke_in_background(query, metadata or {}))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return {"answer": self.immediate_message}
        else:
            try:
                answer = await self._call_openclaw_api(query)
                return {"answer": answer}
            except Exception:
                logger.exception("Error at invoke_openclaw")
                return {"answer": "Error"}
