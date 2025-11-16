from abc import ABC, abstractmethod
import asyncio
import copy
import inspect
import logging
import re
from typing import AsyncGenerator, List, Dict, Any, Callable, Optional, Tuple, Literal
from .context_manager import ContextManager, SQLiteContextManager

logger = logging.getLogger(__name__)


class ToolCallResult:
    def __init__(self, data: dict = None, is_final: bool = True, text: str = None):
        self.data = data or {}
        self.is_final = is_final
        self.text = text


class ToolCall:
    def __init__(self, id: str = None, name: str = None, arguments: any = None, result: ToolCallResult = None):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.result = result or ToolCallResult()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": {"data": self.result.data, "is_final": self.result.is_final}
        }


class GuardrailRespose:
    def __init__(self, guardrail_name: str, is_triggered: bool, action: Literal["replace", "block"] = "replace", text: str = None):
        self.guardrail_name = guardrail_name
        self.is_triggered = is_triggered
        self.action = action    # 'replace', 'warn', or 'block'(request only)
        self.text = text


class Guardrail(ABC):
    def __init__(self, applies_to: Literal["request", "response", "both"], name: str = None):
        self.applies_to = applies_to
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def apply(self, context_id: str, user_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> GuardrailRespose:
        pass


class LLMResponse:
    def __init__(self, context_id: str, text: str = None, voice_text: str = None, tool_call: ToolCall = None, guradrail_name: str = None):
        self.context_id = context_id
        self.text = text
        self.voice_text = voice_text
        self.tool_call = tool_call
        self.guradrail_name = guradrail_name


class Tool:
    def __init__(self, name: str, spec: Dict[str, Any], func: Callable, instruction: str = None, is_dynamic: bool = False):
        self.name = name
        self.spec = spec
        self.func = func
        self.instruction = instruction
        self.is_dynamic = is_dynamic

    def parse_spec(self, spec: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        if spec.get("type") == "function" and "function" in spec:
            f = self.spec["function"]
            return f["name"], f["description"], f["parameters"]
        elif "functionDeclarations" in spec:
            f = self.spec["functionDeclarations"][0]
            return f["name"], f["description"], f["parameters"]
        elif "input_schema" in spec:
            return spec["name"], spec["description"], spec["input_schema"]

        raise ValueError(f"Unknown tool spec format: {spec}")

    def build_spec(self, llm_service_name: str, name: str, description: str, parameters: dict) -> Dict[str, Any]:
        if "gpt" in llm_service_name.lower():
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }
        elif "gemini" in llm_service_name.lower():
            return {
                "functionDeclarations": [{
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }]
            }
        elif "claude" in llm_service_name.lower():
            return {
                "name": name,
                "description": description,
                "input_schema": parameters
            }

        raise ValueError(f"Unknown LLM service: {llm_service_name}")

    def clone_for(self, llm_service_name: str) -> "Tool":
        tool = copy.copy(self)
        n, d, p = self.parse_spec(self.spec)
        tool.spec = self.build_spec(llm_service_name, n, d, p)
        return tool

    def add_to_mcp_server(self, mcp_server):
        mcp_server.add_tool(
            fn=self.func,
            name=self.name,
            description=self.spec["function"]["description"],
        )


class LLMService(ABC):
    def __init__(
        self,
        *,
        system_prompt: str,
        model: str,
        temperature: float = 0.5,
        initial_messages: List[dict] = None,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        use_dynamic_tools: bool = False,
        context_manager: ContextManager = None,
        shared_context_ids: List[str] = None,
        guardrails: List[Guardrail] = None,
        db_connection_str: str = "aiavatar.db",
        debug: bool = False
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.initial_messages = initial_messages
        self.split_chars = split_chars or ["。", "？", "！", ". ", "?", "!", "\n"]
        self.option_split_chars = option_split_chars or ["、", ", "]
        self.option_split_threshold = option_split_threshold

        self.split_chars_pattern = "|".join(
            re.escape(char) for char in sorted(self.split_chars, key=len, reverse=True)
        )
        # option_split_patterns is built with \s? for delimiters that don’t already end in a space so we can consume an optional trailing space when we perform the fallback split.
        self.option_split_patterns = [
            re.escape(char) if char.endswith(" ") else f"{re.escape(char)}\\s?"
            for char in sorted(self.option_split_chars, key=len, reverse=True)
        ]
        self.option_split_chars_regex = (
            f"({'|'.join(self.option_split_patterns)})\\s*(?!.*({'|'.join(self.option_split_patterns)}))"
        )

        self._request_filter = None
        self._update_context_filter = None
        self.voice_text_tag = voice_text_tag
        self.tools: Dict[str, Tool] = {}
        self.use_dynamic_tools = use_dynamic_tools
        self.dynamic_tool_instruction = """

## Important: Use of `{dynamic_tool_name}`

When external tools, knowledge, or data are required to process a user's request, use the appropriate tools.  

Examples where external tools are needed:

- Performing web searches
- Retrieving weather information
- Retrieving memory from past conversations
- Playing game
- Any other cases that requires accessing real-world systems or data to provide better solutions

**NOTE**: Say something before execute tool. (e.g. I will look it up on the web. Wait a moment.)

"""
        self.additional_prompt_for_tool_listing = """
----
Extract up to five tools that could be used to process the above user input.
The response should follow this format. If multiple tools apply, separate them with commas.

[tools:{tool_name},{tool_name},{tool_name}]

If none apply, respond as follows:

[tool_name:NOT_FOUND]

The list of tools is as follows:

"""
        self._get_dynamic_tools = self.get_dynamic_tools_default
        self._on_before_tool_calls = self.on_before_tool_calls_default
        if context_manager:
            self.context_manager = context_manager
        else:
            if db_connection_str.startswith("postgresql://"):
                from .context_manager.postgres import PostgreSQLContextManager
                self.context_manager = PostgreSQLContextManager(connection_str=db_connection_str)
            else:
                self.context_manager = SQLiteContextManager(db_path=db_connection_str)
        self.shared_context_ids = shared_context_ids
        self.guardrails = guardrails or []
        self.debug = debug

    # Decorators
    def request_filter(self, func):
        self._request_filter = func
        return func

    def update_context_filter(self, func):
        self._update_context_filter = func
        return func

    def tool(self, spec):
        def decorator(func):
            return func
        return decorator

    def add_tool(self, tool: Tool, is_dynamic: bool = False, use_original: bool = False):
        if use_original:
            tool_to_add = tool
        else:
            tool_to_add = tool.clone_for(self.__class__.__name__)
            tool_to_add.is_dynamic = is_dynamic
        self.tools[tool_to_add.name] = tool_to_add

    async def get_dynamic_tools(self, func):
        self._get_dynamic_tools = func
        return func

    def on_before_tool_calls(self, func):
        self._on_before_tool_calls = func
        return func

    async def on_before_tool_calls_default(self, tool_calls: List[ToolCall]):
        pass

    def replace_last_option_split_char(self, original):
        return re.sub(self.option_split_chars_regex, r"\1|", original)

    def get_system_prompt(self, context_id: str, user_id: str, system_prompt_params: Dict[str, any]):
        if not system_prompt_params:
            return self.system_prompt
        else:
            return self.system_prompt.format(**system_prompt_params)

    @property
    @abstractmethod
    def dynamic_tool_name(self) -> str:
        pass

    @abstractmethod
    async def compose_messages(self, context_id: str, user_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> List[Dict]:
        pass

    @abstractmethod
    async def update_context(self, context_id: str, user_id: str, messages: List[Dict], response_text: str):
        pass

    async def get_dynamic_tools_default(self, messages: List[dict], metadata: Dict[str, any] = None) -> List[Dict[str, any]]:
        return []

    @abstractmethod
    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[dict], system_prompt_params: Dict[str, any] = None) -> AsyncGenerator[LLMResponse, None]:
        pass

    def remove_control_tags(self, text: str) -> str:
        clean_text = text
        clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", clean_text)
        clean_text = clean_text.strip()
        return clean_text

    async def execute_tool(self, name: str, arguments: dict, metadata: dict) -> AsyncGenerator[ToolCallResult, None]:
        tool = self.tools[name]
        if "metadata" in inspect.signature(tool.func).parameters:
            arguments["metadata"] = metadata

        tool_result = tool.func(**arguments)
        if inspect.isasyncgen(tool_result):
            async for r in tool_result:
                if isinstance(r, Tuple):
                    yield ToolCallResult(data=r[0], is_final=r[1])
                elif isinstance(r, dict):
                    yield ToolCallResult(data=r, is_final=False)
                elif isinstance(r, str):
                    yield ToolCallResult(text=r, is_final=False)
                else:
                    yield r
        elif isinstance(tool_result, ToolCallResult):
            yield tool_result
        else:
            yield ToolCallResult(data=await tool_result)

    async def apply_guardrails(self, applies_to: Literal["request", "response"], context_id: str, user_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> GuardrailRespose:
        guardrails = [gr for gr in self.guardrails if gr.applies_to in (applies_to, "both")]
        if guardrails:
            # Start processing guardrails in parallel
            tasks = [
                asyncio.create_task(gr.apply(
                    context_id=context_id,
                    user_id=user_id,
                    text=text,
                    files=files,
                    system_prompt_params=system_prompt_params
                )) for gr in guardrails
            ]
            try:
                # Process responses from completed guardrails
                for task in asyncio.as_completed(tasks):
                    grresp = await task
                    if not grresp.is_triggered:
                        continue
                    logger.warning(
                        f"Guardrail for {applies_to} '{grresp.guardrail_name}' triggered: action={grresp.action}, text={grresp.text}"
                    )
                    return grresp
            finally:
                # Cancel tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                for task in tasks:
                    if task.done():
                        continue
                    # Await tasks if not done and ignore if error
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        return None

    async def chat_stream(self, context_id: str, user_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> AsyncGenerator[LLMResponse, None]:
        logger.info(f"User: {text}")
        if self._request_filter:
            text = self._request_filter(text)
            logger.info(f"User(Filtered): {text}")

        if not text and not files:
            return

        # Request guardrails
        if request_guardrail_response := await self.apply_guardrails(
            applies_to="request",
            context_id=context_id,
            user_id=user_id,
            text=text,
            files=files,
            system_prompt_params=system_prompt_params
        ):
            if request_guardrail_response.action == "replace":
                text = request_guardrail_response.text
            elif request_guardrail_response.action == "block":
                yield LLMResponse(
                    context_id=context_id,
                    text=request_guardrail_response.text,
                    voice_text=self.remove_control_tags(request_guardrail_response.text),
                    guradrail_name=request_guardrail_response.guardrail_name
                )
                return

        messages = await self.compose_messages(context_id, user_id, text, files, system_prompt_params)
        message_length_at_start = len(messages) - 1

        stream_buffer = ""
        response_text = ""
        
        in_voice_tag = False
        target_start = f"<{self.voice_text_tag}>"
        target_end = f"</{self.voice_text_tag}>"

        def to_voice_text(segment: str) -> Optional[str]:
            if not self.voice_text_tag:
                return self.remove_control_tags(segment)

            nonlocal in_voice_tag
            if target_start in segment and target_end in segment:
                in_voice_tag = False
                start_index = segment.find(target_start)
                end_index = segment.find(target_end)
                voice_segment = segment[start_index + len(target_start): end_index]
                return self.remove_control_tags(voice_segment)

            elif target_start in segment:
                in_voice_tag = True
                start_index = segment.find(target_start)
                voice_segment = segment[start_index + len(target_start):]
                return self.remove_control_tags(voice_segment)

            elif target_end in segment:
                if in_voice_tag:
                    in_voice_tag = False
                    end_index = segment.find(target_end)
                    voice_segment = segment[:end_index]
                    return self.remove_control_tags(voice_segment)

            elif in_voice_tag:
                return self.remove_control_tags(segment)

            return None

        async for chunk in self.get_llm_stream_response(context_id, user_id, messages, system_prompt_params):
            if chunk.tool_call:
                if stream_buffer:
                    # Yield text content before tool call
                    voice_text = to_voice_text(stream_buffer)
                    yield LLMResponse(context_id, stream_buffer, voice_text)
                    response_text += stream_buffer
                    stream_buffer = ""
                yield chunk

                if chunk.tool_call.name == self.dynamic_tool_name:
                    logger.info(f"self.dynamic_tool_name: {self.dynamic_tool_name}")
                    # Clear text content for `execute_external_tool` not to be included in the context
                    response_text = ""
                continue

            stream_buffer += chunk.text

            # Replace consecutive punctuation with the same punctuation followed by delimiter
            stream_buffer = re.sub(f"(({self.split_chars_pattern})+)", r"\1|", stream_buffer)

            if len(stream_buffer) > self.option_split_threshold:
                stream_buffer = self.replace_last_option_split_char(stream_buffer)

            segments = stream_buffer.split("|")
            while len(segments) > 1:
                sentence = segments.pop(0)
                stream_buffer = "|".join(segments)
                voice_text = to_voice_text(sentence)
                yield LLMResponse(context_id, sentence, voice_text)
                response_text += sentence
                segments = stream_buffer.split("|")

            await asyncio.sleep(0.001)   # wait slightly in every loop not to use up CPU

        if stream_buffer:
            voice_text = to_voice_text(stream_buffer)
            yield LLMResponse(context_id, stream_buffer, voice_text)
            response_text += stream_buffer

        if self.voice_text_tag and self.voice_text_tag not in response_text:
            # Fallback for the case when no voice text tags
            if voice_text := self.remove_control_tags(response_text):
                yield LLMResponse(context_id, "", voice_text)

        # Response guardrails
        if response_guardrail_response := await self.apply_guardrails(
            applies_to="response",
            context_id=context_id,
            user_id=user_id,
            text=response_text
        ):
            yield LLMResponse(
                context_id=context_id,
                text=response_guardrail_response.text,
                voice_text=self.remove_control_tags(response_guardrail_response.text),
                guradrail_name=response_guardrail_response.guardrail_name
            )

        logger.info(f"AI: {response_text}")
        if len(messages) > message_length_at_start:
            await self.update_context(
                context_id,
                user_id,
                messages[message_length_at_start - len(messages):],
                response_text.strip(),
            )
