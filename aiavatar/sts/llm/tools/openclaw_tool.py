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
        harness: str = None,
    ):
        self.openclaw_api_key = openclaw_api_key
        self.openclaw_base_url = openclaw_base_url
        self.harness = harness


class OpenClawTool(Tool):
    def __init__(
        self,
        *,
        openclaw_api_key: str = None,
        openclaw_base_url: str = None,
        openclaw_configs: Dict[str, OpenClawConfig] = None,
        harness: str = "openclaw",
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
        self.openclaw_configs = openclaw_configs or {}
        self.harness = harness
        self._request_builders: Dict[str, Callable] = {
            "openclaw": self._openclaw_request_builder,
            "hermes": self._hermes_request_builder,
        }
        self._response_parsers: Dict[str, Callable] = {
            "openclaw": self._openclaw_response_parser,
            "hermes": self._hermes_response_parser,
        }
        self.timeout = timeout
        self.stream = stream
        self.debug = debug
        self._running_tasks: Dict[str, dict] = {}
        self._session_key_map: Dict[str, str] = {}

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

    # Session key mapping

    def get_session_key(self, harness: str, context_id: str) -> str:
        return self._session_key_map.get(f"{harness}:{context_id}")

    def set_session_key(self, harness: str, context_id: str, session_key: str):
        self._session_key_map[f"{harness}:{context_id}"] = session_key

    def delete_session_key(self, harness: str, context_id: str):
        self._session_key_map.pop(f"{harness}:{context_id}", None)

    def request_builder(self, key: str = None):
        def decorator(func: Callable):
            self._request_builders[key or self.harness] = func
            return func
        return decorator

    def response_parser(self, func_or_key=None):
        if callable(func_or_key):
            self._response_parsers[self.harness] = func_or_key
            return func_or_key
        else:
            key = func_or_key or self.harness
            def decorator(func: Callable):
                self._response_parsers[key] = func
                return func
            return decorator

    @staticmethod
    def _openclaw_request_builder(task_id: str, context_id: str) -> dict:
        result = {"model": "openclaw"}
        if context_id:
            result["extra_headers"] = {"x-openclaw-session-key": context_id}
        return result

    @staticmethod
    def _hermes_request_builder(task_id: str, context_id: str) -> dict:
        result = {"model": "hermes-agent"}
        if context_id:
            result["extra_headers"] = {"X-Hermes-Session-Id": context_id}
        return result

    def _openclaw_response_parser(self, task_id: str, context_id: str, chunk) -> str:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            return delta.content
        return None

    def _hermes_response_parser(self, task_id: str, context_id: str, chunk) -> str:
        # Progress tracking
        if hasattr(chunk, "tool"):
            progress_line = ""
            if tool_name := chunk.tool:
                emoji = chunk.emoji if hasattr(chunk, "emoji") else ""
                progress_line = f"- {emoji} {tool_name}"
                label = chunk.label if hasattr(chunk, "label") else ""
                if label:
                    progress_line += f": {label}"
                progress_line += "\n"
            if progress_line and task_id:
                self.add_progress(task_id, progress_line)

        # Extract content
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            return delta.content
        return None

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
        config.harness = user_config.harness or self.harness
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
        build_request = self._request_builders.get(
            config.harness, self._openclaw_request_builder
        )
        parse_response = self._response_parsers.get(
            config.harness, self._openclaw_response_parser
        )
        extra_kwargs = build_request(task_id, context_id)
        model = extra_kwargs.pop("model", None)

        client = openai.AsyncClient(
            api_key=config.openclaw_api_key,
            base_url=config.openclaw_base_url,
            timeout=self.timeout
        )

        try:
            if self.stream:
                stream = await client.chat.completions.create(
                    messages=[{"role": "user", "content": query}],
                    model=model,
                    stream=True,
                    **extra_kwargs
                )
                chunks = []
                async for chunk in stream:
                    content = parse_response(task_id, context_id, chunk)
                    if content:
                        chunks.append(content)
                answer = "".join(chunks)

            else:
                resp = await client.chat.completions.create(
                    messages=[{"role": "user", "content": query}],
                    model=model,
                    **extra_kwargs
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
