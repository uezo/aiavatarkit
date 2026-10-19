import asyncio
from datetime import datetime, timezone
import logging
from uuid import uuid4

import httpx

from .. import Tool

logger = logging.getLogger(__name__)


class GrokBotTool(Tool):
    """Send a request to Grok Bot and wait for its webhook response."""

    def __init__(
        self,
        *,
        webhook_url: str,
        webhook_key: str,
        timeout: float = 300.0,
        http_timeout: float = 30.0,
        immediate_message: str = "Accepted. You will be notified when the response is ready.",
        name=None,
        spec=None,
        instruction=None,
        is_dynamic=False,
    ):
        self.webhook_url = webhook_url
        self.webhook_key = webhook_key
        self.timeout = timeout
        self.http_timeout = http_timeout
        self._running_tasks: dict[str, dict] = {}

        name = name or "send_query_to_grokbot"
        super().__init__(
            name=name,
            spec=spec or {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Send a request or task to Grok Bot and receive its answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The request or task to send to Grok Bot.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            func=self.invoke_grokbot,
            instruction=instruction,
            is_dynamic=is_dynamic,
            immediate_message=immediate_message,
        )

    async def invoke_grokbot(self, query: str, metadata: dict = None):
        metadata = metadata or {}
        request_id = metadata.get("task_id") or str(uuid4())
        submitted_at = metadata.get("submitted_at")
        requested_at = (
            datetime.fromtimestamp(submitted_at, timezone.utc)
            if submitted_at is not None else datetime.now(timezone.utc)
        ).isoformat()
        request_info = {
            "request_id": request_id,
            "requested_at": requested_at,
            "request": query,
        }
        future = asyncio.get_running_loop().create_future()
        # A callback can arrive before the POST response has returned.
        running_task = {
            **request_info,
            "context_id": metadata.get("context_id"),
            "user_id": metadata.get("user_id"),
            "progress": "",
            "task": asyncio.current_task(),
            "future": future,
            "response_timeout": None,
        }
        self._running_tasks[request_id] = running_task
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.post(
                    self.webhook_url,
                    headers={"Authorization": f"Bearer {self.webhook_key}"},
                    json={"text": query, "request_id": request_id, "requested_at": requested_at},
                )
                response.raise_for_status()
            async with asyncio.timeout(self.timeout) as response_timeout:
                running_task["response_timeout"] = response_timeout
                answer = await future
            return {**request_info, "answer": answer}
        except asyncio.TimeoutError:
            return {**request_info, "answer": "Timed out waiting for Grok Bot's response."}
        except httpx.HTTPError:
            return {**request_info, "answer": "Failed to send the request to Grok Bot."}
        finally:
            self._running_tasks.pop(request_id, None)
            if not future.done():
                future.cancel()

    def receive_response(self, request_id: str, text: str = None, status: str = "done") -> bool:
        """Record progress, complete, or silently cancel an authenticated request."""
        running = self._running_tasks.get(request_id)
        if running is None or running["future"].done():
            return False
        future = running["future"]
        if status == "canceled":
            future.cancel()
            running["task"].cancel()
            logger.info("Grok Bot request canceled: request_id=%s", request_id)
            return True
        if text is None:
            return False
        if status == "progress":
            running["progress"] += text + "\n"
            response_timeout = running["response_timeout"]
            if response_timeout is not None and self.timeout is not None:
                response_timeout.reschedule(asyncio.get_running_loop().time() + self.timeout)
            return True
        if status != "done":
            return False
        future.set_result(text)
        return True

    def get_running_tasks(self, context_id: str = None, user_id: str = None) -> list[dict]:
        if not context_id and not user_id:
            return []
        return [
            {key: task[key] for key in ("request_id", "requested_at", "request", "progress")}
            for task in self._running_tasks.values()
            if (not context_id or task["context_id"] == context_id)
            and (not user_id or task["user_id"] == user_id)
        ]

    def create_check_tool(self, name: str = "check_running_grokbot_tasks", description: str = None) -> Tool:
        async def check(metadata: dict = None):
            metadata = metadata or {}
            tasks = self.get_running_tasks(
                context_id=metadata.get("context_id"),
                user_id=metadata.get("user_id"),
            )
            if tasks:
                return {"running_tasks": tasks}
            return {"running_tasks": [], "message": "No running tasks."}

        return Tool(
            name=name,
            spec={
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or "Check the requests and progress of running Grok Bot tasks.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            func=check,
        )

    async def aclose(self):
        """Cancel outstanding requests and completion callbacks on shutdown."""
        tasks = {running["task"] for running in self._running_tasks.values()} | self._background_tasks
        tasks.discard(asyncio.current_task())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
