from datetime import datetime
from logging import getLogger, NullHandler
import traceback
from typing import AsyncGenerator
import google.generativeai as genai
from . import ChatProcessor

class GeminiProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, model: str="gemini-pro", temperature: float=1.0, max_tokens: int=200, functions: dict=None, system_message_content: str=None, system_message_content_acknowledgement_content: str="了解しました。", history_count: int=10, history_timeout: float=60.0):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message_content = system_message_content
        self.system_message_content_acknowledgement_content = system_message_content_acknowledgement_content
        self.history_count = history_count
        self.history_timeout = history_timeout
        self.histories = []
        self.last_chat_at = datetime.utcnow()
        self.on_start_processing = None

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def reset_histories(self):
        self.histories.clear()

    def build_messages(self, text):
        messages = []
        try:
            # System message
            if self.system_message_content:
                messages.append({"role": "user", "parts": [{"text": self.system_message_content}]})
                messages.append({"role": "model", "parts": [{"text": self.system_message_content_acknowledgement_content}]})

            # Histories
            messages.extend(self.histories[-1 * self.history_count:])

            # Current user message
            messages.append({"role": "user", "parts": [{"text": text}]})
        
        except Exception as ex:
            self.logger.error(f"Error at build_messages: {ex}\n{traceback.format_exc()}")

        return messages

    async def chat(self, text: str) -> AsyncGenerator[str, None]:
        try:
            if (datetime.utcnow() - self.last_chat_at).total_seconds() > self.history_timeout:
                self.reset_histories()

            if self.on_start_processing:
                await self.on_start_processing()

            # Get context
            messages = self.build_messages(text)

            # Params
            generation_config = {
                "temperature": self.temperature
            }
            if self.max_tokens:
                generation_config["max_output_tokens"] = self.max_tokens

            # Stream
            response_text = ""
            stream_resp = await self.client.generate_content_async(messages, generation_config=generation_config)
            async for chunk in stream_resp:
                content = chunk.candidates[0].content.parts[0].text
                response_text += content
                yield content

            # Save context
            if response_text:
                self.histories.append(messages[-1])
                self.histories.append({"role": "model", "parts": [{"text": response_text}]})

        except Exception as ex:
            self.logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            self.last_chat_at = datetime.utcnow()
