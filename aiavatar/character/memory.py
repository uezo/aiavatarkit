from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
import logging
from typing import Tuple
import httpx
from aiavatar.sts.models import STSRequest, STSResponse
from .models import MemorySearchResult

logger = logging.getLogger(__name__)


class MemoryClientBase(ABC):
    @abstractmethod
    async def search(
        self,
        *,
        character_id: str,
        user_id: str,
        query: str,
        since: str = None,
        until: str = None
    ) -> MemorySearchResult:
        pass

    @abstractmethod
    async def add_messages(
        self,
        *,
        character_id: str,
        request: STSRequest,
        response: STSResponse,
    ):
        pass

    @abstractmethod
    async def upsert_diary(
        self,
        *,
        character_id: str,
        content: str,
        diary_date: datetime
    ):
        pass


class MemoryClient(MemoryClientBase):
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
        self._worker_task: asyncio.Task = None

    async def search(
        self,
        *,
        character_id: str,
        user_id: str,
        query: str,
        since: str = None,
        until: str = None
    ) -> MemorySearchResult:
        if not user_id or not character_id or not query:
            return MemorySearchResult(answer=None, retrieved_data=None)

        if self.debug:
            logger.info(f"ChatMemory.search: user_id={user_id} / query={query} / since={since} / until={until}")

        try:
            resp = await self.http_client.post(
                url=f"{self.base_url}/search",
                json={
                    "user_id": user_id + "_" + character_id,
                    "query": query,
                    "top_k": self.top_k,
                    "search_content": self.search_content,
                    "include_retrieved_data": self.include_retrieved_data,
                    "since": since,
                    "until": until
                }
            )
            resp.raise_for_status()
            resp_json = resp.json()

            if self.debug:
                logger.info(f"ChatMemory.search: result={resp_json} ")

            return MemorySearchResult(
                answer=resp_json["result"]["answer"],
                retrieved_data=resp_json["result"]["retrieved_data"]
            )

        except Exception as ex:
            logger.exception(f"Error at search ChatMemory: {ex}")
            raise ex

    # Message
    async def add_messages(
        self,
        *,
        character_id: str,
        request: STSRequest,
        response: STSResponse,
    ):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())
        await self._queue.put((request, response, character_id))

    async def _process_queue(self):
        while True:
            request, response, character_id = await self._queue.get()
            try:
                if not request.user_id or not request.context_id or not request.text or not response.voice_text or not character_id:
                    continue
                resp = await self.http_client.post(
                    url=f"{self.base_url}/history",
                    json={
                        "user_id": request.user_id + "_" + character_id,
                        "session_id": request.context_id,
                        "messages": [
                            {"role": "user", "content": request.text},
                            {"role": "assistant", "content": response.voice_text}
                        ]
                    }
                )
                resp.raise_for_status()
            except Exception as ex:
                logger.exception(f"Error processing queued messages: {ex}")
            finally:
                self._queue.task_done()

    # Diary
    async def upsert_diary(
        self,
        *,
        character_id: str,
        content: str,
        diary_date: datetime
    ):
        resp = await self.http_client.post(
            url=f"{self.base_url}/diary",
            json={
                "user_id": character_id,
                "content": content,
                "diary_date": diary_date.strftime("%Y-%m-%d")
            }
        )
        return resp.json()

    async def close(self):
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.http_client.aclose()
