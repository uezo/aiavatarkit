import json
from logging import getLogger
import re
from typing import AsyncGenerator, Dict, List, Protocol, Type
from urllib.parse import urlparse, parse_qs
import openai as openai_module
from . import LLMService, LLMResponse, ToolCall, Tool
from .context_manager import ContextManager

class OpenAICompatibleModule(Protocol):
    AsyncClient: Type[openai_module.AsyncClient]
    AsyncAzureOpenAI: Type[openai_module.AsyncAzureOpenAI]

logger = getLogger(__name__)


class ChatGPTService(LLMService):
    def __init__(
        self,
        *,
        openai_api_key: str = None,
        system_prompt: str = None,
        base_url: str = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.5,
        reasoning_effort: str = None,
        enable_tool_filtering: bool = True,
        extra_body: dict = None,
        initial_messages: List[dict] = None,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        use_dynamic_tools: bool = False,
        context_manager: ContextManager = None,
        shared_context_ids: List[str] = None,
        db_connection_str: str = "aiavatar.db",
        custom_openai_module: OpenAICompatibleModule = None,
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
        self.reasoning_effort = reasoning_effort
        self.enable_tool_filtering = enable_tool_filtering
        self.extra_body = extra_body

        client_module = custom_openai_module or openai_module
        if "azure" in model:
            api_version = parse_qs(urlparse(base_url).query).get("api-version", [None])[0]
            self.openai_client = client_module.AsyncAzureOpenAI(
                api_key=openai_api_key,
                api_version=api_version,
                base_url=base_url
            )
        else:
            self.openai_client = client_module.AsyncClient(api_key=openai_api_key, base_url=base_url)

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
            return func
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
        chat_completion_params = {
            "model": self.model,
        }
        if len(messages) > 1 and messages[-1]["role"] == "tool":
            chat_completion_params["messages"] = messages + [{"role": "user", "content": user_content_for_tool}]
        else:
            chat_completion_params["messages"] = messages[:-1] + [{"role": "user", "content": user_content_for_tool}]

        if self.reasoning_effort:
            chat_completion_params["reasoning_effort"] = self.reasoning_effort
        elif self.model.startswith("gpt-5.1"):
            chat_completion_params["reasoning_effort"] = "none"
        elif self.model.startswith("gpt-5"):
            chat_completion_params["reasoning_effort"] = "minimal"
        else:
            chat_completion_params["temperature"] = self.temperature

        if self.extra_body:
            chat_completion_params["extra_body"] = self.extra_body

        if self.debug:
            logger.info(f"Request to ChatGPT to get dynamic tools: {chat_completion_params}")

        tool_choice_resp = await self.openai_client.chat.completions.create(**chat_completion_params)

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

    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[Dict], system_prompt_params: Dict[str, any] = None, tools: List[Dict[str, any]] = None) -> AsyncGenerator[LLMResponse, None]:
        # Prepare all tools list (always include all tools)
        all_tools = [t.spec for _, t in self.tools.items()]
        if self.use_dynamic_tools:
            all_tools.append(self.dynamic_tool_spec)

        # Determine which tools can be used
        if tools:
            # Specific tools provided
            allowed_tools = [{"type": "function", "function": {"name": t["function"]["name"]}} for t in tools]
        elif self.use_dynamic_tools:
            # Dynamic mode: start with detection tool only
            allowed_tools = [{"type": "function", "function": {"name": self.dynamic_tool_name}}]
        else:
            # Normal mode: all non-dynamic tools available
            non_dynamic_tools = [t.spec for _, t in self.tools.items() if not t.is_dynamic] or None
            allowed_tools = [{"type": "function", "function": {"name": t["function"]["name"]}} for t in non_dynamic_tools] if non_dynamic_tools else None

        # Make params
        chat_completion_params = {
            "messages": messages,
            "model": self.model,
            "stream": True
        }

        # Temperature and Reasoning Effort
        if self.reasoning_effort:
            chat_completion_params["reasoning_effort"] = self.reasoning_effort
        elif self.model.startswith("gpt-5.1"):
            chat_completion_params["reasoning_effort"] = "none"
        elif self.model.startswith("gpt-5"):
            chat_completion_params["reasoning_effort"] = "minimal"
        else:
            chat_completion_params["temperature"] = self.temperature

        if self.extra_body:
            chat_completion_params["extra_body"] = self.extra_body

        # Tools
        if all_tools:
            chat_completion_params["tools"] = all_tools
            if any(m in self.model for m in ("grok", "gemini", "claude")):
                # Skip setting `tool_choice` for non-GPT models
                pass
            elif self.enable_tool_filtering:
                if allowed_tools:
                    chat_completion_params["tool_choice"] = {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": allowed_tools}}
                else:
                    chat_completion_params["tool_choice"] = "none"

        if self.debug:
            logger.info(f"Request to ChatGPT: {chat_completion_params}")

        # Send request
        stream_resp = await self.openai_client.chat.completions.create(**chat_completion_params)

        tool_calls: List[ToolCall] = []
        try_dynamic_tools = False
        filtered_tools = []
        async for chunk in stream_resp:
            if not chunk.choices or not chunk.choices[0].delta: # Azure OpenAI with content filter (streaming mode) returns choices without delta
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
                        }]
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
