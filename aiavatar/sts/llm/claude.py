import json
from logging import getLogger
import re
from typing import AsyncGenerator, Dict, List
from anthropic import AsyncAnthropic
from . import LLMService, LLMResponse, ToolCall, Tool
from .context_manager import ContextManager

logger = getLogger(__name__)


class ClaudeService(LLMService):
    def __init__(
        self,
        *,
        anthropic_api_key: str = None,
        system_prompt: str = None,
        base_url: str = None,
        model: str = "claude-3-7-sonnet-latest",
        temperature: float = 0.5,
        max_tokens: int = 1024,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        use_dynamic_tools: bool = False,
        context_manager: ContextManager = None,
        debug: bool = False
    ):
        super().__init__(
            system_prompt=system_prompt or "",
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
        self.anthropic_client = AsyncAnthropic(
            api_key=anthropic_api_key,
            base_url=base_url
        )
        self.max_tokens = max_tokens

        self.dynamic_tool_spec = {
            "name": "execute_external_tool",
            "description": "Execute the most appropriate tool based on the user's intent: what they want to do and to what.",
            "input_schema": {
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
        }

    @property
    def dynamic_tool_name(self) -> str:
        return self.dynamic_tool_spec["name"]

    async def compose_messages(self, context_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> List[Dict]:
        messages = []

        # Extract the history starting from the first message where the role is 'user'
        histories = await self.context_manager.get_histories(context_id)
        while histories and histories[0]["role"] != "user":
            histories.pop(0)
        messages.extend(histories)

        content = []
        if files:
            for f in files:
                if url := f.get("url"):
                    content.append({"type": "image", "source": {"type": "url", "url": url}})
        if text:
            content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})

        return messages

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        if self._update_context_filter:
            if "text" in messages[0]["content"][-1]:
                messages[0]["content"][-1]["text"] = self._update_context_filter(messages[0]["content"][-1]["text"])
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        await self.context_manager.add_histories(context_id, messages, "claude")

    def tool(self, spec: Dict):
        def decorator(func):
            tool_name = spec["name"]
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
            tool_listing_prompt += f'- {t.name}: {t.spec["description"]}\n'
        tool_listing_prompt += "- NOT_FOUND: Return this if no suitable tools are found.\n"

        # Build user message content
        user_content = messages[-1]["content"]
        if isinstance(user_content, list):
            user_content_for_tool = []
            text_updated = False
            for c in user_content:
                content_type = c["type"]
                if content_type == "text" and not text_updated:
                    # Update text content
                    user_content_for_tool.append({"type": "text", "text": c["text"] + tool_listing_prompt})
                    text_updated = True
                else:
                    # Use original non-text content (e.g. image)
                    user_content_for_tool.append(c)
            # Add text content if no text content are found
            if not text_updated:
                user_content_for_tool.append({"type": "text", "text": tool_listing_prompt})
        elif isinstance(user_content, str):
            user_content_for_tool = user_content + tool_listing_prompt

        # Call LLM to filter tools
        tool_choice_resp = await self.anthropic_client.messages.create(
            messages=messages[:-1] + [{"role": "user", "content": user_content_for_tool}],
            system=metadata["system_prompt"],
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens
        )

        # Parse tools from response
        if match := re.search(r"\[tools:(.*?)\]", tool_choice_resp.content[0].text):
            tool_names = match.group(1)
        else:
            tool_names = "NOT_FOUND"

        tools = []
        for t in tool_names.split(","):
            if tool := self.tools.get(t.strip()):
                tools.append(tool.spec)

        return tools

    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[dict], system_prompt_params: Dict[str, any] = None, tools: List[Dict[str, any]] = None) -> AsyncGenerator[LLMResponse, None]:
        # Select tools to use
        tool_instruction = ""
        if tools:
            filtered_tools = tools
            for t in filtered_tools:
                if ti := self.tools.get(t["name"]).instruction:
                    tool_instruction += f"{ti}\n\n"
        elif self.use_dynamic_tools:
            filtered_tools = [self.dynamic_tool_spec]
            tool_instruction = self.dynamic_tool_instruction.format(
                dynamic_tool_name=self.dynamic_tool_name
            )
        else:
            filtered_tools = [t.spec for _, t in self.tools.items() if not t.is_dynamic] or []

        async with self.anthropic_client.messages.stream(
            messages=messages,
            system=self.get_system_prompt(system_prompt_params) + tool_instruction,
            model=self.model,
            temperature=self.temperature,
            tools=filtered_tools,
            max_tokens=self.max_tokens
        ) as stream_resp:
            tool_calls: List[ToolCall] = []
            try_dynamic_tools = False
            response_text = ""
            async for chunk in stream_resp:
                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        tool_calls.append(ToolCall(chunk.content_block.id, chunk.content_block.name, ""))
                        yield LLMResponse(context_id=context_id, text="\n")    # Add "\n" to flush text content immediately
                        if chunk.content_block.name == self.dynamic_tool_name:
                            logger.info("Get dynamic tool")
                            filtered_tools = await self._get_dynamic_tools(
                                messages,
                                {"system_prompt": self.get_system_prompt(system_prompt_params)}
                            )
                            logger.info(f"Dynamic tools: {filtered_tools}")
                            try_dynamic_tools = True
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        response_text += chunk.delta.text
                        if tools and messages[-1]["content"][0]["type"] != "tool_result":
                            # Do not yield text content in the response for retrying request with retrieved tools
                            # - Request with `execute_external_tool` -> "Wait a moment."
                            # - Request with `google_search` -> "Wait a moment." <= **Skip this text content**
                            # - Request with google_search result -> "Ui Shigure is a world-wide popular illustrator."
                            pass
                        else:
                            yield LLMResponse(context_id=context_id, text=chunk.delta.text)
                    elif chunk.delta.type == "input_json_delta":
                        tool_calls[-1].arguments += chunk.delta.partial_json

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            await self._on_before_tool_calls(tool_calls)

            # NOTE: Claude 3.5 Sonnet doesn't return multiple tools at once for now (2025-01-07), but it's not explicitly documented.
            #       Multiple tools will be called sequentially: user -(llm)-> tool_use -> tool_result -(llm)-> tool_use -> tool_result -(llm)-> assistant
            # Execute tools
            messages_length = len(messages)
            for tc in tool_calls:
                if self.debug:
                    logger.info(f"ToolCall: {tc.name}")
                yield LLMResponse(context_id=context_id, tool_call=tc)

                if tc.arguments:
                    arguments_json = json.loads(tc.arguments)
                else:
                    arguments_json = {}

                tool_result = None
                if tc.name == self.dynamic_tool_name:
                    if not filtered_tools:
                        tool_result = {"message": "No tools found"}
                else:
                    async for tr in self.execute_tool(tc.name, arguments_json, {"user_id": user_id}):
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
                    assistant_content = []
                    if response_text:
                        assistant_content.append({
                            "type": "text",
                            "text": response_text
                        })
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": arguments_json
                    })
                    messages.append({
                        "role": "assistant",
                        "content": assistant_content
                    })

                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": json.dumps(tool_result)
                        }]
                    })

            if len(messages) > messages_length or try_dynamic_tools:
                # Generate human-friendly message that explains tool result
                async for llm_response in self.get_llm_stream_response(
                    context_id, user_id, messages, system_prompt_params=system_prompt_params, tools=filtered_tools
                ):
                    yield llm_response
