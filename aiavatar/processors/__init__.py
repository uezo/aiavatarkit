from abc import ABC, abstractmethod
from typing import Iterator

class ChatProcessor(ABC):
    @abstractmethod
    async def chat(self, text: str) -> Iterator[str]:
        pass
