from datetime import datetime
import json
from logging import getLogger, NullHandler
import traceback
from typing import AsyncGenerator
import httpx
from . import ChatProcessor


class DifyProcessor(ChatProcessor):
    def __init__(self, *, api_key: str, user: str, base_url: str="http://localhost/v1", history_timeout: float=60.0, use_vision: bool=False, openai_api_key: str=None):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.api_key = api_key
        self.user = user
        self.base_url = base_url
        self.timeout = 60.0
        self.history_timeout = history_timeout
        self.last_chat_at = datetime.utcnow()
        self.on_start_processing = None
        self.current_conversation_id = None

        # Vision
        self.use_vision = use_vision
        self.openai_api_key = openai_api_key
        self.local_histories = []
        self.is_vision_required_func = {
            "name": "is_vision_required",
            "description": "Determine whether the vision input is required to process the user input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_required": {"type": "boolean"},
                    "acknowledgement_message": {"type": "string", "description": "Message to user to wait while checking the image."}
                },
                "required": ["is_required", "acknowledgement_message"]
            }
        }
        self.get_image = None

    def reset_histories(self):
        self.current_conversation_id = None

    async def is_vision_required(self, async_client: httpx.AsyncClient, text: str) -> tuple[bool, str]:
        resp = await async_client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}"
            },
            json={
                "messages": self.local_histories[-10:] + [{
                    "role": "user",
                    "content": f"以下はユーザーからの入力内容です。このメッセージを処理するために新たな画像の入力が必要か判断してください。\n\n入力: {text}"
                }],
                "model": "gpt-4o",
                "temperature": 0.0,
                "tools": [{"type": "function", "function": self.is_vision_required_func}],
                "tool_choice": {"type": "function", "function": {"name": "is_vision_required"}},
                "stream": False,
            }
        )

        arguments = json.loads(resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        if arguments["is_required"]:
            return True, arguments["acknowledgement_message"]
        else:
            return False, None

    async def upload_image(self, async_client: httpx.AsyncClient, image_bytes: str) -> str:
        resp = await async_client.post(
            self.base_url + "/files/upload",
            headers={
                "Authorization": f"Bearer {self.api_key}"
            },
            data={
                "user": self.user
            },
            files={
                "file": ("image.png", image_bytes, "image/png")
            }
        )
        return resp.json()["id"]

    async def chat(self, text: str) -> AsyncGenerator[str, None]:
        async_client = httpx.AsyncClient(timeout=self.timeout)

        try:
            if (datetime.utcnow() - self.last_chat_at).total_seconds() > self.history_timeout:
                self.reset_histories()

            if self.on_start_processing:
                await self.on_start_processing()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "inputs": {},
                "query": text,
                "response_mode": "streaming",
                "user": self.user,
                "auto_generate_name": False
            }
            if self.current_conversation_id:
                data["conversation_id"] = self.current_conversation_id
            if self.use_vision and self.get_image is not None:
                is_vision_required, ack_message = await self.is_vision_required(async_client, text)
                if is_vision_required:
                    self.logger.info("Vision input is required")
                    yield ack_message
                    image_bytes = await self.get_image()
                    uploaded_image_id = await self.upload_image(async_client, image_bytes)
                    data["files"] = [{
                        "type": "image",
                        "transfer_method": "local_file",
                        "upload_file_id": uploaded_image_id
                    }]

            stream_resp = await async_client.post(self.base_url + "/chat-messages", headers=headers, json=data)
            stream_resp.raise_for_status()

            response_text = ""
            async for chunk in stream_resp.aiter_lines():
                if chunk.startswith("data:"):
                    chunk_json = json.loads(chunk[5:])
                    if chunk_json["event"] == "message":
                        self.current_conversation_id = chunk_json["conversation_id"]
                        answer = chunk_json["answer"]
                        response_text += answer
                        yield answer

            if self.use_vision:
                self.local_histories.append({"role": "user", "content": text})
                self.local_histories.append({"role": "assistant", "content": response_text})

        except Exception as ex:
            self.logger.error(f"Error at chat: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            self.last_chat_at = datetime.utcnow()
            if not async_client.is_closed:
                await async_client.aclose()
