from abc import ABC, abstractmethod
import asyncio
from logging import getLogger, NullHandler
from time import time

class FaceController(ABC):
    @abstractmethod
    async def set_face(self, name: str, duration: float):
        pass

class FaceControllerDummy(FaceController):
    def __init__(self) -> None:
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.faces = {
            "neutral": "('_')",
            "joy": "(^o^)",
            "angry": "(#｀Д´)",
            "sorrow": "(; ;)",
            "fun": "(*^_^*)",
        }

    async def set_face(self, name: str, duration: float):
        start_at = time()
        self.logger.info(f"face: {self.faces[name]} ({name})")
        while time() - start_at <= duration:
            await asyncio.sleep(0.1)
        self.logger.info(f"face: {self.faces['neutral']} (neutral)")
