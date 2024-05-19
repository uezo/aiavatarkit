from abc import abstractmethod
import base64
from logging import getLogger, NullHandler
from datetime import datetime
import traceback
import json
from typing import Iterator, Callable, AsyncGenerator
from openai import AsyncClient
from . import ChatProcessor

class ChatGPTFunction:
    def __init__(self, name: str, description: str=None, parameters: dict=None, func: Callable=None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
    
    def get_spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ChatCompletionStreamResponse:
    def __init__(self, stream: Iterator[str], function_name: str=None):
        self.stream = stream
        self.function_name = function_name

    @property
    def response_type(self):
        return "function_call" if self.function_name else "content"


class ChatGPTProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, base_url: str=None, model: str="gpt-3.5-turbo", temperature: float=1.0, max_tokens: int=0, functions: dict=None, parse_function_call_in_response: bool=True, system_message_content: str=None, history_count: int=10, history_timeout: float=60.0):
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

    def add_function(self, name: str, description: str=None, parameters: dict=None, func: Callable=None):
        self.functions[name] = ChatGPTFunction(name=name, description=description, parameters=parameters, func=func)

    def reset_histories(self):
        self.histories.clear()

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
            params["functions"] = [v.get_spec() for _, v in self.functions.items()]

        stream_resp = ChatCompletionStreamResponse(await async_client.chat.completions.create(**params))

        if self.parse_function_call_in_response:
            async for chunk in stream_resp.stream:
                if chunk:
                    delta = chunk.choices[0].delta
                    if delta.function_call:
                        stream_resp.function_name = delta.function_call.name
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
                    function_call = delta.function_call
                    if function_call:
                        arguments = function_call.arguments
                        response_text += arguments

            if stream_resp.response_type == "function_call":
                self.histories.append(messages[-1])
                self.histories.append({
                    "role": "assistant",
                    "function_call": {
                        "name": stream_resp.function_name,
                        "arguments": response_text
                    },
                    "content": None
                })

                api_resp = await self.functions[stream_resp.function_name].func(**json.loads(response_text))

                messages.append({"role": "function", "content": json.dumps(api_resp), "name": stream_resp.function_name})

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

        except Exception as ex:
            self.logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            self.last_chat_at = datetime.utcnow()
            if not async_client.is_closed():
                await async_client.close()


class ChatGPTProcessorWithVisionBase(ChatGPTProcessor):
    def __init__(self, *, api_key: str, base_url: str = None, model: str = "gpt-3.5-turbo", temperature: float = 1, max_tokens: int = 0, functions: dict = None, parse_function_call_in_response: bool = True, system_message_content: str = None, history_count: int = 10, history_timeout: float = 60, use_vision: bool = True):
        super().__init__(api_key=api_key, base_url=base_url, model=model, temperature=temperature, max_tokens=max_tokens, functions=functions, parse_function_call_in_response=parse_function_call_in_response, system_message_content=system_message_content, history_count=history_count, history_timeout=history_timeout)
        self.use_vision = use_vision

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
                    if last_user_message["content"][i].get("type") != "text":
                        del last_user_message["content"][i]

        async_client = AsyncClient(api_key=self.api_key)
        try:
            # Determine whether the vision input is required to process the user input
            resp = await async_client.chat.completions.create(
                model=self.model,
                messages=messages[:-1] + [{"role": "user", "content": f"以下はユーザーからの入力内容です。このメッセージを処理するのに新たな画像の入力が必要か判断してください。\n\n入力: {text}"}],
                functions=[{
                    "name": "is_vision_required",
                    "description": "Determine whether the vision input is required to process the user input.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_required": {"type": "boolean"}
                        },
                        "required": ["is_required"]
                    }
                }],
                function_call={"name": "is_vision_required"},
                stream=False
            )

            if "true" in resp.choices[0].message.function_call.arguments:
                self.logger.info("Vision input is required")

                # Convert image to data url
                image_bytes = await self.get_image()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_data_url = f"data:image/png;base64,{image_b64}"
                # Overwrite content
                messages[-1]["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]

        except Exception as ex:
            self.logger.error(f"Error at build_messages: {str(ex)}\n{traceback.format_exc()}")

        finally:
            if not async_client.is_closed():
                await async_client.close()

        return messages
