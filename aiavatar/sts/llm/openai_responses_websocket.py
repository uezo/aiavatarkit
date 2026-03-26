import asyncio
import copy
import json
from contextlib import asynccontextmanager
from logging import getLogger
from time import time
from typing import AsyncGenerator, Dict, List, Union
import websockets
from . import LLMService, LLMResponse, ToolCall, Tool
from .context_manager import ContextManager

logger = getLogger(__name__)


class WebSocketPool:
    """Async WebSocket connection pool with semaphore-based concurrency control."""

    def __init__(self, url: str, api_key: str, max_size: int = 5, max_age: float = 3300):
        self._url = url
        self._api_key = api_key
        self._max_size = max_size
        self._max_age = max_age  # seconds (default 55 min, before OpenAI's 60 min limit)
        self._idle: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_size)

    @staticmethod
    def _is_closed(ws) -> bool:
        # websockets >= 13 (ClientConnection): no .closed attr, use .close_code
        # websockets < 13 (WebSocketClientProtocol): has .closed bool
        if hasattr(ws, "closed"):
            return ws.closed
        return ws.close_code is not None

    async def _connect(self):
        ws = await websockets.connect(
            self._url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
        )
        return ws, time()

    @asynccontextmanager
    async def connection(self):
        """Acquire a WebSocket connection, yield it, then release back to pool."""
        await self._semaphore.acquire()
        ws = None
        created_at = 0.0
        error_occurred = False
        try:
            # Try to reuse an idle connection
            while not self._idle.empty():
                try:
                    ws, created_at = self._idle.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not self._is_closed(ws) and (time() - created_at) < self._max_age:
                    break
                try:
                    await ws.close()
                except Exception:
                    pass
                ws = None

            # Create new connection if needed
            if ws is None or self._is_closed(ws):
                ws, created_at = await self._connect()

            yield ws

        except Exception:
            error_occurred = True
            raise
        finally:
            if ws is not None:
                if not error_occurred and not self._is_closed(ws) and (time() - created_at) < self._max_age:
                    self._idle.put_nowait((ws, created_at))
                else:
                    try:
                        await ws.close()
                    except Exception:
                        pass
            self._semaphore.release()

    async def close(self):
        """Close all idle connections in the pool."""
        while not self._idle.empty():
            try:
                ws, _ = self._idle.get_nowait()
                await ws.close()
            except Exception:
                pass


class OpenAIResponsesWebSocketService(LLMService):
    def __init__(
        self,
        *,
        openai_api_key: str = None,
        system_prompt: str = None,
        ws_url: str = None,
        model: str = "gpt-5.4",
        reasoning_effort: str = None,
        extra_body: dict = None,
        initial_messages: List[dict] = None,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        split_on_control_tags: bool = True,
        voice_text_tag: Union[str, List[str]] = None,
        max_connections: int = 100,  # Max concurrent WebSocket connections (= max parallel requests)
        max_connection_age: float = 3300,
        context_manager: ContextManager = None,
        shared_context_ids: List[str] = None,
        db_connection_str: str = "aiavatar.db",
        debug: bool = False
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            initial_messages=initial_messages,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            split_on_control_tags=split_on_control_tags,
            voice_text_tag=voice_text_tag,
            context_manager=context_manager,
            shared_context_ids=shared_context_ids,
            db_connection_str=db_connection_str,
            debug=debug
        )
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body
        self.response_ids: Dict[str, str] = {}
        self._edit_response_params = None

        base = ws_url or "wss://api.openai.com"
        self._ws_pool = WebSocketPool(
            url=f"{base.rstrip('/')}/v1/responses",
            api_key=openai_api_key,
            max_size=max_connections,
            max_age=max_connection_age,
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config["reasoning_effort"] = self.reasoning_effort
        config["extra_body"] = self.extra_body
        return config

    @property
    def dynamic_tool_name(self) -> str:
        return None

    def edit_response_params(self, func):
        self._edit_response_params = func
        return func

    async def compose_messages(self, context_id: str, user_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> List[Dict]:
        messages = []

        # Add initial messages for the first call only (no server-side history yet)
        if not self.response_ids.get(context_id):
            initial_msgs = await self._get_initial_messages(context_id, user_id, system_prompt_params)
            if initial_msgs:
                messages.extend(initial_msgs)

        # Build user message
        if files:
            content = []
            for f in files:
                if url := f.get("url"):
                    content.append({"type": "input_image", "image_url": url})
            if text:
                content.append({"type": "input_text", "text": text})
        else:
            content = text
        messages.append({"role": "user", "content": content})

        return messages

    async def update_context(self, context_id: str, user_id: str, messages: List[Dict], response_text: str):
        # Context is managed at OpenAI server via previous_response_id
        await self.context_manager.add_histories(context_id, [{}], "openai_responses_ws")

    def tool(self, spec: Dict):
        def decorator(func):
            # Accept Chat Completions format and convert to Responses API format
            if spec.get("type") == "function" and "function" in spec:
                tool_name = spec["function"]["name"]
                responses_spec = {
                    "type": "function",
                    "name": spec["function"]["name"],
                    "description": spec["function"]["description"],
                    "parameters": spec["function"]["parameters"],
                }
            else:
                tool_name = spec.get("name", func.__name__)
                responses_spec = spec

            self.tools[tool_name] = Tool(
                name=tool_name,
                spec=responses_spec,
                func=func
            )
            return func
        return decorator

    def add_tool(self, tool: Tool, is_dynamic: bool = False, use_original: bool = False):
        if use_original:
            tool_to_add = tool
        else:
            tool_to_add = copy.copy(tool)
            n, d, p = tool.parse_spec(tool.spec)
            tool_to_add.spec = {
                "type": "function",
                "name": n,
                "description": d,
                "parameters": p,
            }
            tool_to_add.is_dynamic = is_dynamic
        self.tools[tool_to_add.name] = tool_to_add

    def _build_response_create(self, context_id: str, input_data: list, system_prompt: str = None) -> dict:
        """Build a response.create message for WebSocket."""
        message = {
            "type": "response.create",
            "model": self.model,
            "input": input_data,
        }

        if system_prompt:
            message["instructions"] = system_prompt

        if previous_response_id := self.response_ids.get(context_id):
            message["previous_response_id"] = previous_response_id

        # Temperature and reasoning effort
        if self.reasoning_effort is not None:
            message["reasoning"] = {"effort": self.reasoning_effort}

        # Tools
        all_tools = [t.spec for _, t in self.tools.items()]
        if all_tools:
            message["tools"] = all_tools

        return message

    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[Dict], system_prompt_params: Dict[str, any] = None, tools: List[Dict[str, any]] = None, inline_llm_params: Dict[str, any] = None) -> AsyncGenerator[LLMResponse, None]:
        # System prompt
        system_prompt = await self._get_system_prompt(context_id, user_id, system_prompt_params)

        try:
            async with self._ws_pool.connection() as ws:
                current_input = messages

                while True:
                    # Build response.create message
                    ws_message = self._build_response_create(context_id, current_input, system_prompt)

                    if self.extra_body:
                        ws_message["extra_body"] = self.extra_body

                    if inline_llm_params:
                        for k, v in inline_llm_params.items():
                            ws_message[k] = v

                    if self._edit_response_params:
                        self._edit_response_params(ws_message, context_id, user_id)

                    if self.debug:
                        logger.info(f"Request to OpenAI Responses API (WebSocket): {ws_message}")

                    await ws.send(json.dumps(ws_message))

                    # Receive and process events until response completes
                    tool_calls: List[ToolCall] = []
                    tool_call_map: Dict[int, int] = {}

                    while True:
                        raw = await ws.recv()
                        event = json.loads(raw)
                        event_type = event.get("type")

                        if event_type == "response.output_text.delta":
                            yield LLMResponse(context_id=context_id, text=event.get("delta", ""))

                        elif event_type == "response.output_item.added":
                            item = event.get("item", {})
                            if item.get("type") == "function_call":
                                tc = ToolCall(item.get("call_id"), item.get("name"), "")
                                tool_call_map[event.get("output_index")] = len(tool_calls)
                                tool_calls.append(tc)

                        elif event_type == "response.function_call_arguments.delta":
                            idx = tool_call_map.get(event.get("output_index"))
                            if idx is not None:
                                tool_calls[idx].arguments += event.get("delta", "")

                        elif event_type == "response.completed":
                            resp = event.get("response", {})
                            if resp_id := resp.get("id"):
                                self.response_ids[context_id] = resp_id
                            break

                        elif event_type == "response.failed":
                            resp = event.get("response", {})
                            error = resp.get("error", {})
                            logger.warning(f"Response failed from OpenAI Responses API (WebSocket): {error}")
                            yield LLMResponse(context_id=context_id, error_info={"exception": Exception(error.get("message", "Unknown error")), "response_json": resp})
                            return

                        elif event_type == "error":
                            error = event.get("error", {})
                            logger.warning(f"Error from OpenAI Responses API (WebSocket): {error}")
                            yield LLMResponse(context_id=context_id, error_info={"exception": Exception(error.get("message", "Unknown error")), "response_json": event})
                            return

                    # No tool calls — done
                    if not tool_calls:
                        break

                    # Execute tool calls
                    await self._on_before_tool_calls(tool_calls)

                    tool_outputs = []
                    has_direct_response = False
                    for tc in tool_calls:
                        if self.debug:
                            logger.info(f"ToolCall: {tc.name}")
                        yield LLMResponse(context_id=context_id, tool_call=ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments, result=None))

                        tool_result = None
                        async for tr in self.execute_tool(tc.name, json.loads(tc.arguments), {"context_id": context_id, "user_id": user_id}):
                            tc.result = tr
                            if tr.text:
                                yield LLMResponse(context_id=context_id, text=tr.text)
                            else:
                                yield LLMResponse(context_id=context_id, tool_call=tc)
                                if tr.is_final:
                                    tool_result = tr.data
                                    break

                        if self.debug:
                            logger.info(f"ToolCall result: {tool_result}")

                        if tool_result:
                            tool_obj = self.tools.get(tc.name)
                            if tool_obj and tool_obj._response_formatter:
                                direct_text = tool_obj._response_formatter(tool_result, json.loads(tc.arguments))
                                yield LLMResponse(context_id=context_id, text=direct_text)
                                has_direct_response = True

                            tool_outputs.append({
                                "type": "function_call_output",
                                "call_id": tc.id,
                                "output": json.dumps(tool_result),
                            })

                    if has_direct_response or not tool_outputs:
                        break

                    # Continue on the same connection with tool outputs
                    current_input = tool_outputs

        except websockets.ConnectionClosed as ex:
            logger.warning(f"WebSocket connection closed: {ex}")
            yield LLMResponse(context_id=context_id, error_info={"exception": ex, "response_json": None})

        except Exception as ex:
            logger.warning(f"Error from OpenAI Responses API (WebSocket): {ex}")
            yield LLMResponse(context_id=context_id, error_info={"exception": ex, "response_json": None})
