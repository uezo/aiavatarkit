import copy
import json
from logging import getLogger
from typing import AsyncGenerator, Dict, List, Union
import openai as openai_module
from . import LLMService, LLMResponse, ToolCall, Tool
from .context_manager import ContextManager
from .response_id_store import ResponseIdStore, SQLiteResponseIdStore

logger = getLogger(__name__)


class OpenAIResponsesService(LLMService):
    def __init__(
        self,
        *,
        openai_api_key: str = None,
        system_prompt: str = None,
        base_url: str = None,
        model: str = "gpt-5.4",
        temperature: float = None,
        reasoning_effort: str = None,
        extra_body: dict = None,
        initial_messages: List[dict] = None,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        split_on_control_tags: bool = True,
        voice_text_tag: Union[str, List[str]] = None,
        context_manager: ContextManager = None,
        response_id_store: ResponseIdStore = None,
        shared_context_ids: List[str] = None,
        db_connection_str: str = "aiavatar.db",
        debug: bool = False
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
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
        if response_id_store:
            self.response_id_store = response_id_store
        else:
            if db_connection_str.startswith("postgresql://"):
                from .response_id_store.postgres import PostgreSQLResponseIdStore
                self.response_id_store = PostgreSQLResponseIdStore(connection_str=db_connection_str)
            else:
                self.response_id_store = SQLiteResponseIdStore(db_path=db_connection_str)
        self._edit_response_params = None

        self.openai_client = openai_module.AsyncClient(
            api_key=openai_api_key, base_url=base_url
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
        if not await self.response_id_store.get(context_id):
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
        # Save context locally for reference by other processes (LLM context is managed at server via previous_response_id)
        if self._update_context_filter:
            if isinstance(messages[0]["content"], list):
                if "text" in messages[0]["content"][-1]:
                    messages[0]["content"][-1]["text"] = self._update_context_filter(messages[0]["content"][-1]["text"])
            elif isinstance(messages[0]["content"], str):
                messages[0]["content"] = self._update_context_filter(messages[0]["content"])
        messages.append({"role": "assistant", "content": response_text})
        await self.context_manager.add_histories(context_id, messages, "openai_responses")

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

    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[Dict], system_prompt_params: Dict[str, any] = None, tools: List[Dict[str, any]] = None, inline_llm_params: Dict[str, any] = None, session_id: str = None, channel: str = None) -> AsyncGenerator[LLMResponse, None]:
        # Build tools list
        all_tools = [t.spec for _, t in self.tools.items()]

        # Build params
        response_params = {
            "model": self.model,
            "input": messages,
            "stream": True,
        }

        # Instructions (system prompt)
        if system_prompt := await self._get_system_prompt(context_id, user_id, system_prompt_params):
            response_params["instructions"] = system_prompt

        # Previous response for conversation continuity
        if previous_response_id := await self.response_id_store.get(context_id):
            response_params["previous_response_id"] = previous_response_id

        # Temperature and reasoning effort
        if self.reasoning_effort is not None:
            response_params["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            response_params["temperature"] = self.temperature

        if self.extra_body:
            response_params["extra_body"] = self.extra_body

        # Tools
        if all_tools:
            response_params["tools"] = all_tools

        # Inline params
        if inline_llm_params:
            for k, v in inline_llm_params.items():
                response_params[k] = v

        # Edit params callback
        if self._edit_response_params:
            self._edit_response_params(response_params, context_id, user_id)

        if self.debug:
            logger.info(f"Request to OpenAI Responses API: {response_params}")

        # Send request
        try:
            stream_resp = await self.openai_client.responses.create(**response_params)

        except openai_module.APIStatusError as aserr:
            response_json = None
            try:
                response_json = aserr.response.json()
            except:
                pass
            logger.warning(f"APIStatusError from OpenAI Responses API: {aserr}")
            yield LLMResponse(context_id=context_id, error_info={"exception": aserr, "response_json": response_json})
            return

        except Exception as ex:
            logger.warning(f"Error from OpenAI Responses API: {ex}")
            yield LLMResponse(context_id=context_id, error_info={"exception": ex, "response_json": None})
            return

        # Process streaming events
        tool_calls: List[ToolCall] = []
        tool_call_map: Dict[int, int] = {}  # output_index -> tool_calls list index

        async for event in stream_resp:
            if event.type == "response.output_text.delta":
                yield LLMResponse(context_id=context_id, text=event.delta)

            elif event.type == "response.output_item.added":
                if event.item.type == "function_call":
                    tc = ToolCall(event.item.call_id, event.item.name, "")
                    tool_call_map[event.output_index] = len(tool_calls)
                    tool_calls.append(tc)

            elif event.type == "response.function_call_arguments.delta":
                idx = tool_call_map.get(event.output_index)
                if idx is not None:
                    tool_calls[idx].arguments += event.delta

            elif event.type == "response.completed":
                await self.response_id_store.set(context_id, event.response.id)

        # Execute tool calls
        if tool_calls:
            try:
                await self._on_before_tool_calls(tool_calls)

                tool_outputs = []
                has_direct_response = False
                for tc in tool_calls:
                    if self.debug:
                        logger.info(f"ToolCall: {tc.name}")
                    yield LLMResponse(context_id=context_id, tool_call=ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments, result=None))

                    tool_result = None
                    async for tr in self.execute_tool(tc.name, json.loads(tc.arguments), {"context_id": context_id, "user_id": user_id, "session_id": session_id, "channel": channel}):
                        tc.result = tr
                        if tr.text:
                            yield LLMResponse(context_id=context_id, text=tr.text)
                        else:
                            yield LLMResponse(context_id=context_id, tool_call=tc, structured_content=tr.structured_content)
                            if tr.is_final:
                                tool_result = tr.data
                                break

                    if self.debug:
                        logger.info(f"ToolCall result: {tool_result}")

                    if tool_result:
                        # Use response_formatter for direct response if available
                        tool_obj = self.tools.get(tc.name)
                        if tool_obj and tool_obj._response_formatter:
                            direct_text = tool_obj._response_formatter(tool_result, json.loads(tc.arguments))
                            yield LLMResponse(context_id=context_id, text=direct_text, structured_content=tc.result.structured_content if tc.result else None)
                            has_direct_response = True

                        tool_outputs.append({
                            "type": "function_call_output",
                            "call_id": tc.id,
                            "output": json.dumps(tool_result),
                        })

                if tool_outputs:
                    # Send tool results back via recursive call with previous_response_id
                    suppress_text = has_direct_response
                    async for llm_response in self.get_llm_stream_response(
                        context_id, user_id, tool_outputs, system_prompt_params=system_prompt_params, session_id=session_id, channel=channel
                    ):
                        if llm_response.tool_call:
                            # Chained tool call detected: stop suppressing so the
                            # subsequent tool's LLM response is yielded normally
                            suppress_text = False
                            yield llm_response
                        elif llm_response.error_info or not suppress_text:
                            yield llm_response

            finally:
                # Start deferred background callbacks regardless of errors
                self._start_deferred_callbacks(tool_calls)
