import json
from logging import getLogger
import re
from typing import AsyncGenerator, Dict, List
from litellm import acompletion
from . import LLMService, LLMResponse, ToolCall, Tool
from .context_manager import ContextManager

logger = getLogger(__name__)


class LiteLLMService(LLMService):
    def __init__(
        self,
        *,
        api_key: str = None,
        system_prompt: str = None,
        system_prompt_by_user_prompt: bool = False,
        base_url: str = None,
        model: str = None,
        temperature: float = 0.5,
        initial_messages: List[dict] = None,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        use_dynamic_tools: bool = False,
        context_manager: ContextManager = None,
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
            voice_text_tag=voice_text_tag,
            use_dynamic_tools=use_dynamic_tools,
            context_manager=context_manager,
            shared_context_ids=shared_context_ids,
            db_connection_str=db_connection_str,
            debug=debug
        )
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt_by_user_prompt = system_prompt_by_user_prompt

        self.dynamic_tool_spec = {
            "type": "function",
            "function": {
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
            }
        }

    @property
    def dynamic_tool_name(self) -> str:
        return self.dynamic_tool_spec["function"]["name"]

    async def compose_messages(self, context_id: str, user_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> List[Dict]:
        messages = []
        if self.system_prompt:
            if self.system_prompt_by_user_prompt:
                messages.append({"role": "user", "content": self.get_system_prompt(context_id, user_id, system_prompt_params)})
                messages.append({"role": "assistant", "content": "ok"})
            else:
                messages.append({"role": "system", "content": self.get_system_prompt(context_id, user_id, system_prompt_params)})

        # Add initial messages (e.g. few-shot)
        if self.initial_messages:
            messages.extend(self.initial_messages)

        # Extract the history starting from the first message where the role is 'user'
        histories = await self.context_manager.get_histories(
            context_id=[context_id] + self.shared_context_ids if self.shared_context_ids else context_id
        )
        while histories and histories[0]["role"] != "user":
            histories.pop(0)
        messages.extend(histories)

        if files:
            content = []
            for f in files:
                if url := f.get("url"):
                    content.append({"type": "image_url", "image_url": {"url": url}})
            if text:
                content.append({"type": "text", "text": text})
        else:
            content = text
        messages.append({"role": "user", "content": content})

        return messages

    async def update_context(self, context_id: str, user_id: str, messages: List[Dict], response_text: str):
        if self._update_context_filter:
            if isinstance(messages[0]["content"], list):
                if "text" in messages[0]["content"][-1]:
                    messages[0]["content"][-1]["text"] = self._update_context_filter(messages[0]["content"][-1]["text"])
            elif isinstance(messages[0]["content"], str):
                messages[0]["content"] = self._update_context_filter(messages[0]["content"])
        messages.append({"role": "assistant", "content": response_text})
        await self.context_manager.add_histories(context_id, messages, "chatgpt")

    def tool(self, spec: Dict):
        def decorator(func):
            tool_name = spec["function"]["name"]
            self.tools[tool_name] = Tool(
                name=tool_name,
                spec=spec,
                func=func
            )
        return decorator

    async def get_dynamic_tools_default(self, messages: List[dict], metadata: Dict[str, any] = None) -> List[Dict[str, any]]:
        # Make additional prompt with registered tools
        tool_listing_prompt = self.additional_prompt_for_tool_listing
        for _, t in self.tools.items():
            tool_listing_prompt += f'- {t.name}: {t.spec["function"]["description"]}\n'
        tool_listing_prompt += "- NOT_FOUND: Use this if no suitable tools are found.\n"

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
        tool_choice_resp = await acompletion(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=messages[:-1] + [{"role": "user", "content": user_content_for_tool}],
            temperature=0.0,
            tools = []  # Somtimes error occurs with Claude without empty tools
        )

        # Parse tools from response
        if match := re.search(r"\[tools:(.*?)\]", tool_choice_resp.choices[0].message.content):
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
                if ti := self.tools.get(t["function"]["name"]).instruction:
                    tool_instruction += f"{ti}\n\n"
        elif self.use_dynamic_tools:
            filtered_tools = [self.dynamic_tool_spec]
            tool_instruction = self.dynamic_tool_instruction.format(
                dynamic_tool_name=self.dynamic_tool_name
            )
        else:
            filtered_tools = [t.spec for _, t in self.tools.items() if not t.is_dynamic] or None

        # Update system prompt
        if tool_instruction:
            if messages[0]["role"] == "system":
                system_message_for_tool = {"role": "system", "content": messages[0]["content"] + tool_instruction}
            elif messages[0]["role"] == "user":
                if len(messages) > 1 and messages[1]["role"] == "assistant" and messages[1]["content"] == "ok":
                    system_message_for_tool = {"role": "user", "content": messages[0]["content"] + tool_instruction}
        else:
            system_message_for_tool = messages[0]

        stream_resp = await acompletion(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=[system_message_for_tool] + messages[1:],
            temperature=self.temperature,
            tools=filtered_tools,
            stream=True
        )

        tool_calls: List[ToolCall] = []
        try_dynamic_tools = False
        response_text = ""
        async for chunk in stream_resp:
            if not chunk.choices:
                continue

            if chunk.choices[0].delta.tool_calls:
                t = chunk.choices[0].delta.tool_calls[0]
                if t.id:
                    tool_calls.append(ToolCall(t.id, t.function.name, ""))
                    if t.function.name == self.dynamic_tool_name:
                        logger.info("Get dynamic tool")
                        filtered_tools = await self._get_dynamic_tools(messages)
                        logger.info(f"Dynamic tools: {filtered_tools}")
                        try_dynamic_tools = True
                if t.function.arguments:
                    tool_calls[-1].arguments += t.function.arguments

            elif content := chunk.choices[0].delta.content:
                if not try_dynamic_tools:
                    response_text += content
                    yield LLMResponse(context_id=context_id, text=content)

        if tool_calls:
            # Do something before tool calls (e.g. say to user that it will take a long time)
            await self._on_before_tool_calls(tool_calls)

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
                    async for tr in self.execute_tool(tc.name, json.loads(tc.arguments), {"user_id": user_id}):
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
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments
                            }
                        }],
                        "content": response_text if response_text else None
                    })

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result),
                        "tool_call_id": tc.id
                    })

            if len(messages) > messages_length or try_dynamic_tools:
                # Generate human-friendly message that explains tool result
                async for llm_response in self.get_llm_stream_response(
                    context_id, user_id, messages, system_prompt_params=system_prompt_params, tools=filtered_tools
                ):
                    yield llm_response
