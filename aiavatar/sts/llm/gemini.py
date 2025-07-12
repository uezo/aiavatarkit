import base64
import copy
from logging import getLogger
import re
from typing import AsyncGenerator, Dict, List
from google import genai
from google.genai import types
import httpx
from . import LLMService, LLMResponse, ToolCall, Tool
from .context_manager import ContextManager

logger = getLogger(__name__)


class GeminiService(LLMService):
    def __init__(
        self,
        *,
        gemini_api_key: str = None,
        system_prompt: str = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.5,
        thinking_budget: int = -1,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        use_dynamic_tools: bool = False,
        context_manager: ContextManager = None,
        debug: bool = False
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            voice_text_tag=voice_text_tag,
            use_dynamic_tools=use_dynamic_tools,
            context_manager=context_manager,
            debug=debug
        )
        self.gemini_client = genai.Client(
            api_key=gemini_api_key
        )
        self.thinking_budget = thinking_budget

        self.dynamic_tool_spec = {
            "functionDeclarations": [{
                "name": "execute_external_tool",
                "description": "Execute the most appropriate tool based on the user's intent: what they want to do and to what.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "What the user wants to interact with (e.g., long-term memory, weather, music)."
                        },
                        "action": {
                            "type": "string",
                            "description": "The type of operation to perform on the target (e.g., retrieve, look up, play)."
                        }
                    },
                    "required": ["target", "action"]
                }
            }]
        }

    async def download_image(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    @property
    def dynamic_tool_name(self) -> str:
        return self.dynamic_tool_spec["functionDeclarations"][0]["name"]

    async def compose_messages(self, context_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> List[Dict]:
        messages = []

        # Extract the history starting from the first message where the role is 'user'
        histories = await self.context_manager.get_histories(context_id)
        while histories and histories[0]["role"] != "user":
            histories.pop(0)
        messages.extend(histories)

        parts = []
        if files:
            for f in files:
                if url := f.get("url"):
                    if url.startswith("http://") or url.startswith("https://"):
                        image_bytes = await self.download_image(url)
                    elif url.startswith("data:"):
                        image_bytes = base64.b64decode(url.split(",", 1)[1])
                    if image_bytes:
                        parts.append(types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png",
                        ))

        if text:
            parts.append(types.Part.from_text(text=text))

        messages.append(types.Content(role="user", parts=parts))
        return messages

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        messages.append(types.Content(role="model", parts=[types.Part.from_text(text=response_text)]))
        dict_messages = []
        for m in messages:
            dumped = m.model_dump()
            for part in dumped.get("parts", []):
                inline_data = part.get("inline_data")
                if inline_data and "data" in inline_data:
                    inline_data["data"] = base64.b64encode(inline_data["data"]).decode("utf-8")
            dict_messages.append(dumped)

        if self._update_context_filter:
            if "text" in dict_messages[0]["parts"][-1]:
                dict_messages[0]["parts"][-1]["text"] = self._update_context_filter(dict_messages[0]["parts"][-1]["text"])
        await self.context_manager.add_histories(context_id, dict_messages, "gemini")

    async def preflight(self):
        # Dummy request to initialize client (The first message takes long time)
        stream_resp = await self.gemini_client.aio.models.generate_content_stream(
            model=self.model,
            contents="say just \"hello\""
        )
        async for chunk in stream_resp:
            pass
        logger.info("Gemini client initialized.")

    def tool(self, spec: Dict):
        def decorator(func):
            tool_name = spec["functionDeclarations"][0]["name"]
            self.tools[tool_name] = Tool(
                name=tool_name,
                spec=spec,
                func=func
            )
            return func
        return decorator

    async def get_dynamic_tools_default(self, messages: List[dict], metadata: Dict[str, any] = None) -> List[Dict[str, any]]:
        # Make additional prompt with registered tools
        tool_listing_prompt = self.additional_prompt_for_tool_listing
        for _, t in self.tools.items():
            tool_listing_prompt += f'- {t.name}: {t.spec["functionDeclarations"][0]["description"]}\n'
        tool_listing_prompt += "- NOT_FOUND: Use this if no suitable tools are found.\n"

        # Build user message content
        user_content_parts = messages[-1].parts
        user_content_parts_for_tool = []
        text_updated = False
        for p in user_content_parts:
            if p.text and not text_updated:
                # Update text content
                user_content_parts_for_tool.append(types.Part.from_text(text=p.text + tool_listing_prompt))
                text_updated = True
            else:
                # Use original non-text content (e.g. image)
                user_content_parts_for_tool.append(p)
        # Add text content if no text content are found
        if not text_updated:
            user_content_parts_for_tool.append(types.Part.from_text(text=tool_listing_prompt))

        # Thinking config
        thinking_config = None
        if self.thinking_budget >= 0:
            thinking_config = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        # Call LLM to filter tools
        tool_choice_resp = await self.gemini_client.aio.models.generate_content(
            model=self.model,
            config = types.GenerateContentConfig(
                system_instruction=metadata["system_prompt"],
                temperature=0.0,
                thinking_config=thinking_config
            ),
            contents=messages[:-1] + [types.Content(
                role="user",
                parts=user_content_parts_for_tool
            )],
        )

        if match := re.search(r"\[tools:(.*?)\]", tool_choice_resp.candidates[0].content.parts[0].text):
            tool_names = match.group(1)
        else:
            tool_names = "NOT_FOUND"

        tools = []
        for t in tool_names.split(","):
            if tool := self.tools.get(t.strip()):
                tools.append(tool.spec)

        return tools

    def rename_tool_names(self, messages: list) -> list:
        renamed_messages = copy.deepcopy(messages)
        dynamic_tool_name = self.dynamic_tool_name

        for message in renamed_messages:
            if isinstance(message, types.Content):
                m = message.model_dump()
            else:
                m = message

            for part in m["parts"]:
                if fc := part.get("function_call"):
                    fc["name"] = dynamic_tool_name
                if fr := part.get("function_response"):
                    fr["name"] = dynamic_tool_name

        return renamed_messages

    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[dict], system_prompt_params: Dict[str, any] = None, tools: List[Dict[str, any]] = None) -> AsyncGenerator[LLMResponse, None]:
        if self.thinking_budget >= 0:
            thinking_config = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        else:
            thinking_config = None

        # Select tools to use
        tool_instruction = ""
        if tools:
            filtered_tools = tools
            for t in filtered_tools:
                if ti := self.tools.get(t["functionDeclarations"][0]["name"]).instruction:
                    tool_instruction += f"{ti}\n\n"
        elif self.use_dynamic_tools:
            filtered_tools = [self.dynamic_tool_spec]
            tool_instruction = self.dynamic_tool_instruction.format(
                dynamic_tool_name=self.dynamic_tool_name
            )
        else:
            filtered_tools = [t.spec for _, t in self.tools.items() if not t.is_dynamic] or None

        stream_resp = await self.gemini_client.aio.models.generate_content_stream(
            model=self.model,
            config = types.GenerateContentConfig(
                system_instruction=self.get_system_prompt(system_prompt_params) + tool_instruction,
                temperature=self.temperature,
                tools=filtered_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=thinking_config
            ),
            contents=self.rename_tool_names(messages) if not tools and self.use_dynamic_tools else messages
        )

        tool_calls: List[ToolCall] = []
        try_dynamic_tools = False
        response_text = ""
        async for chunk in stream_resp:
            if not chunk.candidates or not chunk.candidates[0].content.parts:
                continue
            for part in chunk.candidates[0].content.parts:
                if content := part.text:
                    response_text += content
                    if tools and messages[-1].parts[0].function_response is None:
                        # Do not yield text content in the response for retrying request with retrieved tools
                        # - Request with `execute_external_tool` -> "Wait a moment."
                        # - Request with `google_search` -> "Wait a moment." <= **Skip this text content**
                        # - Request with google_search result -> "Ui Shigure is a world-wide popular illustrator."
                        pass
                    else:
                        yield LLMResponse(context_id=context_id, text=content)
                elif part.function_call:
                    tool_calls.append(ToolCall(part.function_call.id, part.function_call.name, dict(part.function_call.args)))
                    yield LLMResponse(context_id=context_id, text="\n")    # Add "\n" to flush text content immediately
                    if part.function_call.name == self.dynamic_tool_name:
                        logger.info("Get dynamic tool")
                        filtered_tools = await self._get_dynamic_tools(
                            messages,
                            {"system_prompt": self.get_system_prompt(system_prompt_params)}
                        )
                        logger.info(f"Dynamic tools: {filtered_tools}")
                        try_dynamic_tools = True

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            await self._on_before_tool_calls(tool_calls)

            # NOTE: Gemini 2.0 Flash doesn't return multiple tools at once for now (2025-01-07), but it's not explicitly documented.
            #       Multiple tools will be called sequentially: user -(llm)-> function_call -> function_response -(llm)-> function_call -> function_response -(llm)-> assistant
            # Execute tools
            messages_length = len(messages)
            for tc in tool_calls:
                if self.debug:
                    logger.info(f"ToolCall: {tc.name}")
                yield LLMResponse(context_id=context_id, tool_call=tc)

                tool_result = None
                if tc.name == self.dynamic_tool_name:
                    if not filtered_tools:
                        tool_result = {"message": "No tools found"}
                else:
                    if self.debug:
                        tool_names = [t["functionDeclarations"][0]["name"] for t in filtered_tools]
                        logger.info(f"Execute tool: {tc.name} / tools: {tool_names}")

                    async for tr in self.execute_tool(tc.name, tc.arguments, {"user_id": user_id}):
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
                    model_parts = []
                    if response_text:
                        model_parts.append(types.Part.from_text(text=response_text))
                    model_parts.append(types.Part.from_function_call(name=tc.name, args=tc.arguments))
                    messages.append(types.Content(
                        role="model",
                        parts=model_parts
                    ))
                    messages.append(types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(name=tc.name, response=tool_result)]
                    ))

            if len(messages) > messages_length or try_dynamic_tools:
                # Generate human-friendly message that explains tool result
                async for llm_response in self.get_llm_stream_response(
                    context_id, user_id, messages, system_prompt_params=system_prompt_params, tools=filtered_tools
                ):
                    yield llm_response
