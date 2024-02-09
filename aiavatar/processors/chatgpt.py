from logging import getLogger, NullHandler
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
    def __init__(self, api_key: str, model: str="gpt-3.5-turbo", temperature: float=1.0, max_tokens: int=0, functions: dict=None, system_message_content: str=None, history_count: int=10):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.functions = functions or {}
        self.system_message_content = system_message_content
        self.history_count = history_count
        self.histories = []
        self.on_start_processing = None

    def add_function(self, name: str, description: str=None, parameters: dict=None, func: Callable=None):
        self.functions[name] = ChatGPTFunction(name=name, description=description, parameters=parameters, func=func)

    def reset_histories(self):
        self.histories.clear()

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

            if self.on_start_processing:
                await self.on_start_processing()

            messages = []
            if self.system_message_content:
                messages.append({"role": "system", "content": self.system_message_content})
            messages.extend(self.histories[-1 * self.history_count:])
            messages.append({"role": "user", "content": text})

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
            if not async_client.is_closed():
                await async_client.close()
