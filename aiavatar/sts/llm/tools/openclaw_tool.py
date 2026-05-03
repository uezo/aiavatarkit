import logging
from typing import Callable, Dict, List
from uuid import uuid4
import openai
from .. import Tool

logger = logging.getLogger(__name__)


class OpenClawConfig:
    def __init__(
        self,
        *,
        openclaw_api_key: str = None,
        openclaw_base_url: str = None,
        openclaw_session_key: str = None,
        openclaw_session_key_key: str = None,
        openclaw_model: str = None
    ):
        self.openclaw_api_key = openclaw_api_key
        self.openclaw_base_url = openclaw_base_url
        self.openclaw_session_key = openclaw_session_key
        self.openclaw_session_key_key = openclaw_session_key_key
        self.openclaw_model = openclaw_model


class OpenClawTool(Tool):
    def __init__(
        self,
        *,
        openclaw_api_key: str = None,
        openclaw_base_url: str = None,
        openclaw_session_key: str = None,
        openclaw_session_key_key: str = "x-openclaw-session-key",   # set "X-Hermes-Session-Id" for Hermes
        openclaw_model: str = "openclaw",   # set "hermes-agent" for Hermes
        openclaw_configs: Dict[str, OpenClawConfig] = None,
        stream: bool = False,
        immediate_message: str = "Accepted. You will be notified when the response is ready.",
        timeout: int = 30000,
        name=None,
        spec=None,
        instruction=None,
        is_dynamic=False,
        debug: bool = False,
    ):
        self.openclaw_api_key = openclaw_api_key
        self.openclaw_base_url = openclaw_base_url
        self.openclaw_session_key = openclaw_session_key
        self.openclaw_session_key_key = openclaw_session_key_key
        self.openclaw_model = openclaw_model
        self.openclaw_configs = openclaw_configs or {}
        self.timeout = timeout
        self.stream = stream
        self.debug = debug
        self._on_stream_chunk: Callable = None
        self._running_tasks: Dict[str, dict] = {}

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
                            "query": {"type": "string", "description": "The user's request or task description to send to OpenClaw for processing."},
                            "report_channel": {"type": "string", "description": "The channel where the task results should be reported back."},
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

    # Running tasks management

    def add_running_task(self, request: str, metadata: dict, progress: str = None) -> str:
        task_id = metadata.get("task_id")
        logger.info(f"add_running_task: task_id={task_id}")

        task_id = metadata.get("task_id") or str(uuid4())
        self._running_tasks[task_id] = {
            "context_id": metadata.get("context_id"),
            "user_id": metadata.get("user_id"),
            "session_id": metadata.get("session_id"),
            "channel": metadata.get("channel"),
            "report_channel": None,
            "request": request,
            "progress": progress or "",
        }
        return task_id

    def get_running_tasks(self, context_id: str = None, user_id: str = None) -> List[dict]:
        return [
            {"request": t["request"], "progress": t["progress"]}
            for t in self._running_tasks.values()
            if (context_id and t["context_id"] == context_id)
            or (user_id and t["user_id"] == user_id)
        ]

    def remove_running_task(self, task_id: str):
        self._running_tasks.pop(task_id, None)

    def add_progress(self, task_id: str, progress: str):
        if task_id in self._running_tasks:
            self._running_tasks[task_id]["progress"] += progress
            if self.debug:
                logger.info(f"OpenClaw progress: {progress}")

    def set_report_channel(self, task_id: str, channel: str):

        logger.info(f"set_report_channel: task_id={task_id}")

        if running_task := self._running_tasks.get(task_id):
            running_task["report_channel"] = channel

    def create_set_report_channel_tool(self, name: str = "set_openclaw_report_channel", description: str = None) -> Tool:
        openclaw_tool = self

        async def set_channel(task_id: str, report_channel: str, metadata: dict = None):
            if openclaw_tool._running_tasks.get(task_id):
                openclaw_tool.set_report_channel(task_id, report_channel)
                return {"message": f"Report channel for task {task_id} has been set to '{report_channel}'."}
            else:
                return {"message": f"Task {task_id} not found in running tasks."}

        return Tool(
            name=name,
            spec={
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or "Set the report channel for a running OpenClaw background task. Use this to change where the task results will be reported.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "The ID of the running task."},
                            "report_channel": {"type": "string", "description": "The channel where the task results should be reported."},
                        },
                        "required": ["task_id", "report_channel"]
                    },
                }
            },
            func=set_channel,
        )

    def create_check_tool(self, name: str = "check_running_openclaw_tasks", description: str = None) -> Tool:
        openclaw_tool = self

        async def check(metadata: dict = None):
            tasks = openclaw_tool.get_running_tasks(
                context_id=metadata.get("context_id"),
                user_id=metadata.get("user_id"),
            )
            if tasks:
                return {"running_tasks": tasks}
            else:
                return {"running_tasks": [], "message": "No running tasks."}

        return Tool(
            name=name,
            spec={
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or "Check the progress of running background tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                }
            },
            func=check,
        )

    def get_openclaw_config(self, user_id: str) -> OpenClawConfig:
        config = OpenClawConfig()
        user_config = self.openclaw_configs.get(user_id) or config
        config.openclaw_api_key = user_config.openclaw_api_key or self.openclaw_api_key
        config.openclaw_base_url = user_config.openclaw_base_url or self.openclaw_base_url
        config.openclaw_session_key = user_config.openclaw_session_key or self.openclaw_session_key
        config.openclaw_session_key_key = user_config.openclaw_session_key_key or self.openclaw_session_key_key
        config.openclaw_model = user_config.openclaw_model or self.openclaw_model
        return config

    def update_openclaw_config(self, user_id: str, config: OpenClawConfig):
        self.delete_openclaw_config(user_id)
        self.openclaw_configs[user_id] = config

    def delete_openclaw_config(self, user_id: str):
        if user_id in self.openclaw_configs:
            del self.openclaw_configs[user_id]

    async def _call_openclaw_api(self, query: str, context_id: str, user_id: str, task_id: str) -> str:
        if self.debug:
            logger.info(f"Request to OpenClaw (from {user_id}): {query}")

        config = self.get_openclaw_config(user_id)
        client = openai.AsyncClient(
            api_key=config.openclaw_api_key,
            base_url=config.openclaw_base_url,
            timeout=self.timeout
        )

        try:
            if self.stream:
                stream = await client.chat.completions.create(
                    messages=[{"role": "user", "content": query}],
                    model=config.openclaw_model,
                    stream=True,
                    extra_headers={
                        config.openclaw_session_key_key: context_id or config.openclaw_session_key
                    }
                )
                chunks = []
                async for chunk in stream:
                    if task_id and hasattr(chunk, "tool"):
                        progress_line = ""
                        if tool := chunk.tool:
                            emoji = chunk.emoji or ""
                            progress_line = f"- {emoji} {tool}"
                            if label := chunk.label:
                                progress_line += f": {label}"
                            progress_line += "\n"
                        if progress_line:
                            self.add_progress(task_id, progress_line)

                    if self._on_stream_chunk:
                        await self._on_stream_chunk(task_id, chunk)
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        chunks.append(delta.content)
                answer = "".join(chunks)

            else:
                resp = await client.chat.completions.create(
                    messages=[{"role": "user", "content": query}],
                    model=config.openclaw_model,
                    extra_headers={
                        config.openclaw_session_key_key: context_id or config.openclaw_session_key
                    }
                )
                answer = resp.choices[0].message.content
        finally:
            await client.close()

        if self.debug:
            logger.info(f"Response from OpenClaw: {answer}")
        return answer

    async def invoke_openclaw(self, query: str, report_channel: str = None, metadata: dict = None):
        context_id = metadata.get("context_id", "")
        user_id = metadata.get("user_id", "")

        config = self.get_openclaw_config(user_id)
        if not config.openclaw_base_url:
            logger.warning(f"OpenClaw base_url is not configured for user: {user_id}")
            return {"answer": "OpenClaw is not configured for this user. Please set up your OpenClaw connection first.", "report_channel": report_channel}

        task_id = self.add_running_task(request=query, metadata=metadata, progress="Start processing...\n")
        try:
            answer = await self._call_openclaw_api(query, context_id, user_id, task_id)
            report_channel = self._running_tasks[task_id].get("report_channel") or report_channel
            return {"answer": answer, "report_channel": report_channel}
        except Exception as ex:
            logger.exception("Error at invoke_openclaw")
            report_channel = self._running_tasks.get(task_id, {}).get("report_channel") or report_channel
            return {"answer": f"Error: {ex}", "report_channel": report_channel}
        finally:
            self.remove_running_task(task_id)
