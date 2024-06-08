import base64
from datetime import datetime
from logging import getLogger, NullHandler
import re
import traceback
from typing import Callable, AsyncGenerator
import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.protobuf.struct_pb2 import Struct
from . import ChatProcessor


class GeminiFunction:
    def __init__(self, name: str, description: str=None, parameters: dict=None, acknowledgement_message_key: str="acknowledgement_message", func: Callable=None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.acknowledgement_message_key = acknowledgement_message_key
        self.func = func
    
    def get_spec(self):
        return genai.types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )


class GeminiProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, model: str="gemini-1.5-flash-latest", temperature: float=1.0, max_tokens: int=200, functions: dict=None, system_message_content: str=None, history_count: int=10, history_timeout: float=60.0, use_vision: bool=False):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.functions = functions or {}
        self.history_count = history_count
        self.history_timeout = history_timeout
        self.histories = []
        self.last_chat_at = datetime.utcnow()
        self.on_start_processing = None
        self.dump_messages = False

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        # Set system_message_content after creating client
        if system_message_content:
            self.system_message_content = system_message_content

        # Vision
        self.use_vision = use_vision
        self.get_image = None

    @property
    def system_message_content(self):
        if not self.client._system_instruction:
            return None
        return self.client._system_instruction["parts"][0]["text"]
    
    @system_message_content.setter
    def system_message_content(self, value):
        self.client._system_instruction = {"role": "user", "parts": [{"text": value}]}

    def add_function(self, name: str, description: str=None, parameters: dict=None, acknowledgement_message_key: str="acknowledgement_message", func: Callable=None):
        self.functions[name] = GeminiFunction(name=name, description=description, parameters=parameters, acknowledgement_message_key=acknowledgement_message_key, func=func)

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

            if len(messages) > 0:
                part = messages[0]["parts"][0]
                if isinstance(part, glm.Part) and part.function_response:
                    # Invalid context if it starts with function_response
                    del messages[0]

            # Current user message
            messages.append({"role": "user", "parts": [{"text": text}]})
        
        except Exception as ex:
            self.logger.error(f"Error at build_messages: {ex}\n{traceback.format_exc()}")

        return messages

    async def generate_content_stream(self, messages: list, call_functions: bool=True):
        # Params
        generation_config = {
            "temperature": self.temperature
        }
        if self.max_tokens:
            generation_config["max_output_tokens"] = self.max_tokens

        tools = [v.get_spec() for _, v in self.functions.items()] \
            if call_functions and self.functions else None

        if self.dump_messages:
            self.logger.info(f"messages to generate_content:\n{messages}")

        return await self.client.generate_content_async(
            messages,
            generation_config=generation_config,
            tools=tools,
            stream=True
        )

    async def chat(self, text: str) -> AsyncGenerator[str, None]:
        try:
            if (datetime.utcnow() - self.last_chat_at).total_seconds() > self.history_timeout:
                self.reset_histories()

            if self.on_start_processing:
                await self.on_start_processing()

            # Get context
            messages = await self.build_messages(text)

            # Process stream response
            response_text = ""
            stream_resp = await self.generate_content_stream(messages)

            async for chunk in stream_resp:
                if chunk.candidates and chunk.candidates[0].content.parts:
                    for part in chunk.candidates and chunk.candidates[0].content.parts:
                        if function_call := part.function_call:
                            # Make context
                            parts = []
                            if response_text:
                                # Add text content response includes text message before function
                                parts.append({"text": response_text})                            
                            parts.append(part)
                            messages.append({"role": "model", "parts": parts})

                            # Add messages to history
                            self.histories.append(messages[-2])
                            self.histories.append(messages[-1])

                            # Call function
                            called_func = self.functions[function_call.name]
                            arguments = dict(function_call.args)

                            if called_func.acknowledgement_message_key and called_func.acknowledgement_message_key in arguments:
                                yield arguments[called_func.acknowledgement_message_key]
                                del arguments[called_func.acknowledgement_message_key]

                            api_resp = await called_func.func(**arguments)

                            # Build function response
                            # See also: https://github.com/google-gemini/generative-ai-python/issues/243
                            s = Struct()
                            s.update({"result": api_resp})
                            function_response = glm.Part(
                                function_response=glm.FunctionResponse(name=function_call.name, response=s)
                            )

                            # Convert API response to human friendly response
                            messages.append({"role": "user", "parts": [function_response]})

                            response_text = ""
                            stream_resp = await self.generate_content_stream(messages, False)

                            async for chunk in stream_resp:
                                if chunk.candidates and chunk.candidates[0].content.parts:
                                    content = chunk.candidates[0].content.parts[0].text
                                    response_text += content
                                    yield content

                        else:
                            content = chunk.candidates[0].content.parts[0].text
                            response_text += content
                            yield content

            # Save context after receiving stream
            if response_text:
                self.histories.append(messages[-1])
                self.histories.append({"role": "model", "parts": [{"text": response_text}]})

            response_tags = self.extract_tags(response_text)

            if vision_source := response_tags.get("vision"):
                self.logger.info(f"Use vision: {vision_source}")

                # Context
                messages.append({"role": "model", "parts": [{"text": response_text}]})

                # Convert image to data url
                image_bytes = await self.get_image(vision_source)
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                # Build message with image and former request text
                messages.append({
                    "role": "user", "parts": [
                        {"text": text},
                        {"mime_type": "image/png", "data": image_b64}
                    ]
                })

                # Send image and yield response chunks
                response_text = ""
                stream_resp = await self.generate_content_stream(messages)

                async for chunk in stream_resp:
                    if chunk.candidates and chunk.candidates[0].content.parts:
                        for part in chunk.candidates and chunk.candidates[0].content.parts:
                            content = chunk.candidates[0].content.parts[0].text
                            response_text += content
                            yield content

                if response_text:
                    # Save context without image to keep context light
                    self.histories.append({"role": "user", "parts": [{"text": text}]})
                    self.histories.append({"role": "model", "parts": [{"text": response_text}]})

        except Exception as ex:
            self.logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            self.last_chat_at = datetime.utcnow()
