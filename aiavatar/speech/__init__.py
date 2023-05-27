from abc import ABC, abstractmethod

class SpeechController(ABC):
    @abstractmethod
    def prefetch(self, text: str):
        pass

    @abstractmethod
    async def speak(self, text: str):
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        pass
