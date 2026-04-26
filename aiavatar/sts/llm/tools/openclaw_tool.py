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
        openclaw_session_key_key: str = "x-openclaw-session-key",   # set "X-Hermes-Session-Id" for Hermes
        stream: bool = False,
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
        self.openclaw_session_key_key = openclaw_session_key_key
        self.stream = stream
        self.debug = debug
        self._on_stream_chunk: Callable = None

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
            is_dynamic,
            immediate_message=immediate_message
        )

    def on_stream_chunk(self, func: Callable):
        self._on_stream_chunk = func
        return func

    async def _call_openclaw_api(self, query: str, context_id: str) -> str:
        if self.debug:
            logger.info(f"Request to OpenClaw: {query}")

        if self.stream:
            stream = await self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": query}],
                model="hermes-agent",
                stream=True,
                extra_headers={
                    self.openclaw_session_key_key: context_id or self.openclaw_session_key
                }
            )
            chunks = []
            async for chunk in stream:
                if self._on_stream_chunk:
                    await self._on_stream_chunk(chunk)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
            answer = "".join(chunks)

        else:
            resp = await self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": query}],
                model="openclaw",
                extra_headers={
                    self.openclaw_session_key_key: context_id or self.openclaw_session_key
                }
            )
            answer = resp.choices[0].message.content

        if self.debug:
            logger.info(f"Response from OpenClaw: {answer}")
        return answer

    async def invoke_openclaw(self, query: str, metadata: dict = None):
        try:
            context_id = metadata["context_id"]
            answer = await self._call_openclaw_api(query, context_id)
            return {"answer": answer}
        except Exception:
            logger.exception("Error at invoke_openclaw")
            return {"answer": "Error"}
