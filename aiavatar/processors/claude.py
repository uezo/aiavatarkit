from abc import abstractmethod
import base64
from datetime import datetime
import json
from logging import getLogger, NullHandler
import traceback
from typing import Callable, AsyncGenerator
from anthropic import AsyncAnthropic
from . import ChatProcessor


class ClaudeFunction:
    def __init__(self, name: str, description: str=None, input_schema: dict=None, func: Callable=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.func = func
    
    def get_spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class ClaudeProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, model: str="claude-3-sonnet-20240229", temperature: float=1.0, max_tokens: int=200, functions: dict=None, system_message_content: str=None, history_count: int=10, history_timeout: float=60.0):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.functions = functions
        self.system_message_content = system_message_content
        self.history_count = history_count
        self.history_timeout = history_timeout
        self.histories = []
        self.last_chat_at = datetime.utcnow()
        self.on_start_processing = None

    def reset_histories(self):
        self.histories.clear()

    async def build_messages(self, text):
        messages = []
        try:
            # Histories
            messages.extend(self.histories[-1 * self.history_count:])

            # Current user message
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        
        except Exception as ex:
            self.logger.error(f"Error at build_messages: {ex}\n{traceback.format_exc()}")

        return messages

    async def messages_create_stream(self, async_client: AsyncAnthropic, messages: list, call_functions: bool=True):
        params = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        if self.system_message_content:
            params["system"] = self.system_message_content

        if call_functions and self.functions:
            params["tools"] = [v.get_spec() for _, v in self.functions.items()]

        return await async_client.beta.tools.messages.create(**params)

    async def chat(self, text: str) -> AsyncGenerator[str, None]:
        try:
            async_client = AsyncAnthropic(api_key=self.api_key)

            if (datetime.utcnow() - self.last_chat_at).total_seconds() > self.history_timeout:
                self.reset_histories()

            if self.on_start_processing:
                await self.on_start_processing()

            # Get context
            messages = await self.build_messages(text)

            # Process stream response
            stream_resp = await self.messages_create_stream(async_client, messages)

            content_block_type = ""
            response_text = ""
            tool_use_id = ""
            function_name = ""
            function_arguments_str = ""

            async for chunk in stream_resp:
                if chunk.type == "content_block_start":
                    content_block_type = chunk.content_block.type
                    if content_block_type == "tool_use":
                        # Set tool info from start block
                        tool_use_id = chunk.content_block.id
                        function_name = chunk.content_block.name

                elif chunk.type == "content_block_delta":
                    if content_block_type == "text":
                        # yield content
                        content = chunk.delta.text
                        response_text += content
                        yield content
                    elif content_block_type == "tool_use":
                        # correct chunked args
                        function_arguments_str += chunk.delta.partial_json

                elif chunk.type == "content_block_stop":
                    if content_block_type == "tool_use":
                        function_arguments = json.loads(function_arguments_str)

                        # Make context
                        contents = []
                        if response_text:
                            # Add text content response includes text message before function
                            contents.append({
                                "type": "text",
                                "text": response_text
                            })

                        contents.append({
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": function_name,
                            "input": function_arguments
                        })

                        messages.append({
                            "role": "assistant",
                            "content": contents
                        })

                        # Add messages to history
                        self.histories.append(messages[-2])
                        self.histories.append(messages[-1])

                        # Call function
                        api_resp = await self.functions[function_name].func(**function_arguments)

                        # Convert API response to human friendly response
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(api_resp)
                            }]
                        })

                        response_text = ""
                        stream_resp = await self.messages_create_stream(async_client, messages)

                        async for chunk in stream_resp:
                            if chunk.type == "content_block_delta":
                                content = chunk.delta.text
                                response_text += content
                                yield content

            # Save context after receiving stream
            if response_text:
                self.histories.append(messages[-1])
                self.histories.append({"role": "assistant", "content": response_text})

        except Exception as ex:
            self.logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            self.last_chat_at = datetime.utcnow()
            if not async_client.is_closed():
                await async_client.close()


class ClaudeProcessorWithVisionBase(ClaudeProcessor):
    def __init__(self, *, api_key: str, model: str = "claude-3-sonnet-20240229", temperature: float = 1, max_tokens: int = 200, functions: dict = None, system_message_content: str = None, history_count: int = 10, history_timeout: float = 60, use_vision: bool = True):
        super().__init__(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens, functions=functions, system_message_content=system_message_content, history_count=history_count, history_timeout=history_timeout)
        self.use_vision = use_vision
        self.is_vision_required_func = {
            "name": "is_vision_required",
            "description": "Determine whether the vision input is required to process the user input.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "is_required": {"type": "boolean"}
                },
                "required": ["is_required"]
            }
        }

    @abstractmethod
    async def get_image(self) -> bytes:
        pass

    async def build_messages(self, text):
        messages = await super().build_messages(text)
        if not self.use_vision:
            return messages

        if len(self.histories) > 1:
            last_user_message = self.histories[-2:-1][0]
            if isinstance(last_user_message["content"], list):
                for i in range(len(last_user_message["content"]) - 1, -1, -1):
                    # Remove image from last request
                    if last_user_message["content"][i]["type"] == "image":
                        del last_user_message["content"][i]

        async_client = AsyncAnthropic(api_key=self.api_key)
        try:
            # Determine whether the vision input is required to process the user input
            params = {
                "messages": messages[:-1] + [{
                    "role": "user",
                    "content": f"以下はユーザーからの入力内容です。このメッセージを処理するために新たな画像の入力が必要か判断してください。\n\n入力: {text}"
                }],
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system": self.system_message_content,
                "tools": [self.is_vision_required_func],
                "tool_choice": {"type": "tool", "name": "is_vision_required"},
                "stream": False,
            }

            resp = await async_client.beta.tools.messages.create(**params)

            if resp.content[0].input["is_required"] is True:
                self.logger.info("Vision input is required")
                image_bytes = await self.get_image()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                messages[-1]["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    }
                })

        except Exception as ex:
            self.logger.error(f"Error at build_messages: {str(ex)}\n{traceback.format_exc()}")

        finally:
            self.last_chat_at = datetime.utcnow()
            if not async_client.is_closed():
                await async_client.close()

        return messages
