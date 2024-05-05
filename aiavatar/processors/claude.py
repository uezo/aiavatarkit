from datetime import datetime
from logging import getLogger, NullHandler
import traceback
from typing import AsyncGenerator
from anthropic import AsyncAnthropic
from . import ChatProcessor

class ClaudeProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, model: str="claude-3-sonnet-20240229", temperature: float=1.0, max_tokens: int=200, functions: dict=None, system_message_content: str=None, history_count: int=10, history_timeout: float=60.0):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message_content = system_message_content
        self.history_count = history_count
        self.history_timeout = history_timeout
        self.histories = []
        self.last_chat_at = datetime.utcnow()
        self.on_start_processing = None

    def reset_histories(self):
        self.histories.clear()

    def build_messages(self, text):
        messages = []
        try:
            # Histories
            messages.extend(self.histories[-1 * self.history_count:])

            # Current user message
            messages.append({"role": "user", "content": text})
        
        except Exception as ex:
            self.logger.error(f"Error at build_messages: {ex}\n{traceback.format_exc()}")

        return messages

    async def chat(self, text: str) -> AsyncGenerator[str, None]:
        try:
            async_client = AsyncAnthropic(api_key=self.api_key)

            if (datetime.utcnow() - self.last_chat_at).total_seconds() > self.history_timeout:
                self.reset_histories()

            if self.on_start_processing:
                await self.on_start_processing()

            # Get context
            messages = self.build_messages(text)

            # Params
            params = {
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            if self.system_message_content:
                params["system"] = self.system_message_content

            # Stream
            response_text = ""
            stream_resp = await async_client.messages.create(**params)
            async for chunk in stream_resp:
                if chunk.type == "content_block_delta":
                    content = chunk.delta.text
                    response_text += content
                    yield content
            
            # Save context
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
