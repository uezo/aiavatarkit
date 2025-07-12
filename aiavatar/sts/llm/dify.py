from logging import getLogger
import json
from typing import AsyncGenerator, Dict, List
import httpx
from . import LLMService, LLMResponse

logger = getLogger(__name__)


class DifyService(LLMService):
    def __init__(
        self,
        *,
        api_key: str = None,
        user: str = None,
        base_url: str = "http://127.0.0.1",
        is_agent_mode: bool = False,
        make_inputs: callable = None,
        split_chars: List[str] = None,
        option_split_chars: List[str] = None,
        option_split_threshold: int = 50,
        voice_text_tag: str = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0
    ):
        super().__init__(
            system_prompt=None,
            model=None,
            temperature=0.0,
            split_chars=split_chars,
            option_split_chars=option_split_chars,
            option_split_threshold=option_split_threshold,
            voice_text_tag=voice_text_tag
        )
        self.conversation_ids: Dict[str, str] = {}
        self.api_key = api_key
        self.user = user
        self.base_url = base_url
        self.is_agent_mode = is_agent_mode
        self.make_inputs = make_inputs
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )

    @property
    def dynamic_tool_name(self) -> str:
        pass

    async def compose_messages(self, context_id: str, text: str, files: List[Dict[str, str]] = None, system_prompt_params: Dict[str, any] = None) -> List[Dict]:
        if self.make_inputs:
            inputs = self.make_inputs(context_id, text, files, system_prompt_params)
        else:
            inputs = {}

        message = {
            "inputs": inputs,
            "query": text,
            "response_mode": "streaming",
            "user": self.user,
            "auto_generate_name": False,
            "conversation_id": self.conversation_ids.get(context_id, "")
        }
        if files:
            for f in files:
                if url := f.get("url"):
                    files.append({"type": "image", "transfer_method": "remote_url", "url": url})
            message["files"] = files
        
        return [message]

    async def update_context(self, context_id: str, messages: List[Dict], response_text: str):
        # Context is managed at Dify server
        pass


    async def get_llm_stream_response(self, context_id: str, user_id: str, messages: List[dict], system_prompt_params: Dict[str, any] = None) -> AsyncGenerator[LLMResponse, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        if user_id:
            messages[0]["user"] = user_id
        stream_resp = await self.http_client.post(
            self.base_url + "/chat-messages",
            headers=headers,
            json=messages[0]
        )
        stream_resp.raise_for_status()

        message_event_value = "agent_message" if self.is_agent_mode else "message"
        async for chunk in stream_resp.aiter_lines():
            if chunk.startswith("data:"):
                chunk_json = json.loads(chunk[5:])
                if chunk_json["event"] == message_event_value:
                    answer = chunk_json["answer"]
                    yield LLMResponse(context_id=context_id, text=answer)
                elif chunk_json["event"] == "message_end":
                    # Save conversation id instead of managing context locally
                    self.conversation_ids[context_id] = chunk_json["conversation_id"]
