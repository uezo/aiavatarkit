from logging import getLogger, NullHandler
import traceback
from typing import Iterator
from openai import ChatCompletion
from . import ChatProcessor

class ChatGPTProcessor(ChatProcessor):
    def __init__(self, api_key: str, model: str="gpt-3.5-turbo", temperature: float=1.0, max_tokens: int=0, system_message_content: str=None, history_count: int=10):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message_content = system_message_content
        self.history_count = history_count
        self.histories = []

    def reset_histories(self):
        self.histories.clear()

    async def chat(self, text: str) -> Iterator[str]:
        try:
            messages = []
            if self.system_message_content:
                messages.append({"role": "system", "content": self.system_message_content})
            messages.extend(self.histories[-1 * self.history_count:])
            messages.append({"role": "user", "content": text})

            params = {
                "api_key": self.api_key,
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
                "stream": True
            }
            if self.max_tokens:
                params["max_tokens"] = self.max_tokens

            completion = await ChatCompletion.acreate(**params)
            
            response_text = "" 
            async for chunk in completion:
                if chunk:
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        response_text += content
                        yield content
            
            if response_text:
                self.histories.append(messages[-1])
                self.histories.append({"role": "assistant", "content": response_text})

        except Exception as ex:
            self.logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
