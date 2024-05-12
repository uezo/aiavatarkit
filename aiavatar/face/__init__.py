from abc import ABC, abstractmethod
from logging import getLogger, NullHandler
from threading import Thread
from time import time, sleep

class FaceController(ABC):
    @property
    @abstractmethod
    def current_face(self):
        pass

    @abstractmethod
    async def set_face(self, name: str, duration: float):
        pass

    @abstractmethod
    def reset(self):
        pass


class FaceControllerBase(FaceController):
    def __init__(self, verbose: bool=False):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())
        self.verbose = verbose

        self.faces = {
            "neutral": "('_')",
            "joy": "(^o^)",
            "angry": "(#｀Д´)",
            "sorrow": "(; ;)",
            "fun": "(*^_^*)",
        }

        self.reset_at = None
        self.reset_thread = Thread(target=self.reset_worker, daemon=True)
        self.reset_thread.start()
        self._current_face = "neutral"

    @property
    def current_face(self) -> str:
        return self._current_face
    
    @current_face.setter
    def current_face(self, name: str):
        self._current_face = name

    def reset_worker(self):
        while True:
            if self.reset_at and time() >= self.reset_at:
                if self.verbose:
                    self.logger.info(f"Time to reset: {self.reset_at}")
                self.reset()
                self.reset_at = None

            sleep(0.1)

    def subscribe_reset(self, reset_at: float):
        self.reset_at = reset_at
        if self.verbose:
            self.logger.info(f"Reset subscribed at {self.reset_at}")

    async def set_face(self, name: str, duration: float):
        if duration > 0:
            self.subscribe_reset(time() + duration)
        self.logger.info(f"face: {self.faces[name]} ({name})")
        self.current_face = name

    def reset(self):
        self.logger.info(f"Reset face: {self.faces['neutral']} (neutral)")
        self.current_face = "neutral"


class FaceControllerDummy(FaceControllerBase):
    pass
