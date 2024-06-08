from abc import abstractmethod
import base64
from datetime import datetime
import json
from logging import getLogger, NullHandler
import re
import traceback
from typing import Callable, AsyncGenerator
from anthropic import AsyncAnthropic
from . import ChatProcessor


class ClaudeFunction:
    def __init__(self, *, name: str, description: str=None, input_schema: dict=None, acknowledgement_message_key: str="acknowledgement_message", func: Callable=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.acknowledgement_message_key = acknowledgement_message_key
        self.func = func
    
    def get_spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class ClaudeProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, model: str="claude-3-sonnet-20240229", temperature: float=1.0, max_tokens: int=200, functions: dict=None, system_message_content: str=None, history_count: int=10, history_timeout: float=60.0,  use_vision: bool=False):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.functions = functions or {}
        self.system_message_content = system_message_content
        self.history_count = history_count
        self.history_timeout = history_timeout
        self.histories = []
        self.last_chat_at = datetime.utcnow()
        self.on_start_processing = None

        # Vision
        self.use_vision = use_vision
        self.get_image = None

    def add_function(self, *, name: str, description: str=None, parameters: dict=None, acknowledgement_message_key: str="acknowledgement_message", func: Callable=None):
        self.functions[name] = ClaudeFunction(name=name, description=description, input_schema=parameters, acknowledgement_message_key=acknowledgement_message_key, func=func)

    def reset_histories(self):
        self.histories.clear()

    def extract_tags(self, text) -> dict:
        tag_pattern = r"\[(\w+):([^\]]+)\]"
        matches = re.findall(tag_pattern, text)
        return {key: value for key, value in matches}

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
                        func = self.functions[function_name]
                        if func.acknowledgement_message_key and func.acknowledgement_message_key in function_arguments:
                            yield function_arguments[func.acknowledgement_message_key]
                            del function_arguments[func.acknowledgement_message_key]

                        api_resp = await func.func(**function_arguments)

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

            response_tags = self.extract_tags(response_text)

            # Save context after receiving stream
            if response_text:
                self.histories.append(messages[-1])
                self.histories.append({"role": "assistant", "content": response_text})

            if vision_source := response_tags.get("vision"):
                self.logger.info(f"Use vision: {vision_source}")

                # Context
                messages.append({"role": "assistant", "content": response_text})

                # Convert image to data url
                image_bytes = await self.get_image(vision_source)
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
                ]})

                # Send image and yield response chunks
                response_text = ""

                stream_resp = await self.messages_create_stream(async_client, messages)

                async for chunk in stream_resp:
                    if chunk.type == "content_block_delta":
                        if content_block_type == "text":
                            # yield content
                            content = chunk.delta.text
                            response_text += content
                            yield content

                if response_text:
                    # Save context without image to keep context light
                    self.histories.append({"role": "user", "content": text})
                    self.histories.append({"role": "assistant", "content": response_text})

        except Exception as ex:
            self.logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            self.last_chat_at = datetime.utcnow()
            if not async_client.is_closed():
                await async_client.close()
