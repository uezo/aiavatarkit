from abc import ABC, abstractmethod
from typing import List
import httpx
import logging

logger = logging.getLogger(__name__)


class SpeechRecognizer(ABC):
    def __init__(
        self,
        *,
        language: str = None,
        alternative_languages: List[str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        self.language = language
        self.alternative_languages = alternative_languages or []
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )

        self.debug = debug

    @abstractmethod
    async def transcribe(self, data: bytes) -> str:
        pass

    async def close(self):
        await self.http_client.aclose()


class SpeechRecognizerDummy(SpeechRecognizer):
    async def transcribe(self, data: bytes) -> str:
        pass
