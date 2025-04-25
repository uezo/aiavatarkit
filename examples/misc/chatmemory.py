import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import httpx
from litests.models import STSRequest, STSResponse

logger = logging.getLogger(__name__)

@dataclass
class RearchResult:
    answer: Optional[str]
    retrieved_data: Optional[str]

class ChatMemoryClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        top_k: int = 5,
        search_content: bool = False,
        include_retrieved_data: bool = False,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 60.0,
        debug: bool = False
    ):
        self.base_url = base_url
        self.top_k = top_k
        self.search_content = search_content
        self.include_retrieved_data = include_retrieved_data
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )
        self.debug = debug

        self._queue: asyncio.Queue[Tuple[STSRequest, STSResponse]] = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._process_queue())

    async def search(self, user_id: str, query: str) -> RearchResult:
        if not user_id or not query:
            return RearchResult(answer=None, retrieved_data=None)

        if self.debug:
            logger.info(f"ChatMemory.search: user_id={user_id} / query={query}")

        try:
            resp = await self.http_client.post(
                url=f"{self.base_url}/search",
                json={
                    "user_id": user_id,
                    "query": query,
                    "top_k": self.top_k,
                    "search_content": self.search_content,
                    "include_retrieved_data": self.include_retrieved_data
                }
            )
            resp.raise_for_status()
            resp_json = resp.json()

            if self.debug:
                logger.info(f"ChatMemory.search: result={resp_json} ")

            return RearchResult(
                answer=resp_json["result"]["answer"],
                retrieved_data=resp_json["result"]["retrieved_data"]
            )

        except Exception as ex:
            logger.exception(f"Error at search ChatMemory: {ex}")
            raise ex

    async def add_messages(self, request: STSRequest, response: STSResponse):
        if not request.user_id or not request.context_id or not request.text or not response.voice_text:
            return

        try:
            resp = await self.http_client.post(
                url=f"{self.base_url}/history",
                json={
                    "user_id": request.user_id,
                    "session_id": request.context_id,
                    "messages": [
                        {"role": "user", "content": request.text},
                        {"role": "assistant", "content": response.voice_text}
                    ]
                }
            )
            resp.raise_for_status()

        except Exception as ex:
            logger.exception(f"Error at add_messages to ChatMemory: {ex}")
            raise ex

    async def _process_queue(self):
        while True:
            request, response = await self._queue.get()
            try:
                await self.add_messages(request, response)
            except Exception as ex:
                logger.exception(f"Error processing queued messages: {ex}")
            finally:
                self._queue.task_done()

    async def enqueue_messages(self, request: STSRequest, response: STSResponse):
        await self._queue.put((request, response))
