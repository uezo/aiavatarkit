import base64
from logging import getLogger, NullHandler
from datetime import datetime
import traceback
import json
import re
from typing import Iterator, Callable, AsyncGenerator
from openai import AsyncClient
from . import ChatProcessor


class ChatGPTFunction:
    def __init__(self, *, name: str, description: str=None, parameters: dict=None, acknowledgement_message_key: str="acknowledgement_message", func: Callable=None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.acknowledgement_message_key = acknowledgement_message_key
        self.func = func
    
    def get_spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ChatCompletionStreamResponse:
    def __init__(self, stream: Iterator[str], function_name: str=None, tool_call_id: str=None):
        self.stream = stream
        self.function_name = function_name
        self.tool_call_id = tool_call_id

    @property
    def response_type(self):
        return "function_call" if self.function_name else "content"


class ChatGPTProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, base_url: str=None, model: str="gpt-3.5-turbo", temperature: float=1.0, max_tokens: int=0, functions: dict=None, parse_function_call_in_response: bool=True, system_message_content: str=None, history_count: int=10, history_timeout: float=60.0, use_vision: bool=False):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.functions = functions or {}
        self.parse_function_call_in_response = parse_function_call_in_response
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
        self.functions[name] = ChatGPTFunction(name=name, description=description, parameters=parameters, acknowledgement_message_key=acknowledgement_message_key, func=func)

    def reset_histories(self):
        self.histories.clear()

    def extract_tags(self, text) -> dict:
        tag_pattern = r"\[(\w+):([^\]]+)\]"
        matches = re.findall(tag_pattern, text)
        return {key: value for key, value in matches}

    async def build_messages(self, text):
        messages = []
        try:
            # System message
            if self.system_message_content:
                messages.append({"role": "system", "content": self.system_message_content})

            # Histories
            messages.extend(self.histories[-1 * self.history_count:])

            # Current user message
            messages.append({"role": "user", "content": text})
        
        except Exception as ex:
            self.logger.error(f"Error at build_messages: {ex}\n{traceback.format_exc()}")

        return messages

    async def chat_completion_stream(self, async_client: AsyncClient, messages: list, call_functions: bool=True):
        params = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        if call_functions and self.functions:
            params["tools"] = [{"type": "function", "function": v.get_spec()} for _, v in self.functions.items()]

        stream_resp = ChatCompletionStreamResponse(await async_client.chat.completions.create(**params))

        if self.parse_function_call_in_response:
            async for chunk in stream_resp.stream:
                if chunk:
                    delta = chunk.choices[0].delta
                    if delta.tool_calls:
                        stream_resp.function_name = delta.tool_calls[0].function.name
                        stream_resp.tool_call_id = delta.tool_calls[0].id
                    break

        return stream_resp

    async def chat(self, text: str) -> AsyncGenerator[str, None]:
        try:
            async_client = AsyncClient(api_key=self.api_key)
            if self.base_url:
                async_client.base_url = self.base_url

            if (datetime.utcnow() - self.last_chat_at).total_seconds() > self.history_timeout:
                self.reset_histories()

            if self.on_start_processing:
                await self.on_start_processing()

            messages = await self.build_messages(text)

            response_text = ""
            stream_resp = await self.chat_completion_stream(async_client, messages)

            async for chunk in stream_resp.stream:
                delta = chunk.choices[0].delta
                if stream_resp.response_type == "content":
                    content = delta.content
                    if content:
                        response_text += delta.content
                        yield content

                elif stream_resp.response_type == "function_call":
                    if delta.tool_calls:
                        response_text += delta.tool_calls[0].function.arguments

            response_tags = self.extract_tags(response_text)

            if stream_resp.response_type == "function_call":
                # Context
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": stream_resp.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": stream_resp.function_name,
                            "arguments": response_text
                        }
                    }]
                })

                self.histories.append(messages[-2])
                self.histories.append(messages[-1])

                # Execute function
                func = self.functions[stream_resp.function_name]
                arguments = json.loads(response_text)

                if func.acknowledgement_message_key and func.acknowledgement_message_key in arguments:
                    yield arguments[func.acknowledgement_message_key]
                    del arguments[func.acknowledgement_message_key]

                api_resp = await func.func(**arguments)

                messages.append({"role": "tool", "content": json.dumps(api_resp), "tool_call_id": stream_resp.tool_call_id})

                # Make human-friendly response message
                response_text = ""
                stream_resp = await self.chat_completion_stream(async_client, messages, False)

                async for chunk in stream_resp.stream:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        response_text += content
                        yield content

            if response_text:
                self.histories.append(messages[-1])
                self.histories.append({"role": "assistant", "content": response_text})

            if vision_source := response_tags.get("vision"):
                self.logger.info(f"Use vision: {vision_source}")

                # Convert image to data url
                image_bytes = await self.get_image(vision_source)
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_data_url = f"data:image/png;base64,{image_b64}"

                # Build message with image and former request text
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                })

                # Send image and yield response chunks
                response_text = ""
                stream_resp = await self.chat_completion_stream(async_client, messages)
                async for chunk in stream_resp.stream:
                    delta = chunk.choices[0].delta
                    if stream_resp.response_type == "content":
                        content = delta.content
                        if content:
                            response_text += delta.content
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
