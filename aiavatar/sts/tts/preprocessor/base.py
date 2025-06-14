from abc import ABC, abstractmethod


class TTSPreprocessor(ABC):
    @abstractmethod
    async def process(self, text: str, style_info: dict = None, language: str = None) -> str:
        pass
