from abc import ABC, abstractmethod
from logging import getLogger, NullHandler
from threading import Thread
from time import time, sleep

class AnimationController(ABC):
    @property
    @abstractmethod
    def current_animation(self):
        pass

    @abstractmethod
    async def animate(self, name: str, duration: float):
        pass


class AnimationControllerBase(AnimationController):
    def __init__(self, animations: dict=None, idling_key: str="idling", debug: bool=False):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())
        self.debug = debug

        self.animations = animations or {
            idling_key: 0,
            "angry_hands_on_waist": 1,
            "concern_right_hand_front": 2,
            "waving_arm": 3,
            "nodding_once": 4
        }
        self.idling_key = idling_key
        self._current_animation = idling_key

        self.reset_at = None
        self.reset_thread = Thread(target=self.reset_worker, daemon=True)
        self.reset_thread.start()

    @property
    def current_animation(self) -> str:
        return self._current_animation
    
    @current_animation.setter
    def current_animation(self, name: str):
        self._current_animation = name

    def reset_worker(self):
        while True:
            if self.reset_at and time() >= self.reset_at:
                if self.debug:
                    self.logger.info(f"Time to reset: {self.reset_at}")
                self.reset()
                self.reset_at = None

            sleep(0.1)

    def subscribe_reset(self, reset_at: float):
        self.reset_at = reset_at
        if self.debug:
            self.logger.info(f"Reset subscribed at {self.reset_at}")

    async def animate(self, name: str, duration: float):
        self.subscribe_reset(time() + duration)
        self.logger.info(f"animation: {self.animations[name]} ({name})")
        self.current_animation = name

    def reset(self):
        self.logger.info(f"Reset animation: {self.animations[self.idling_key]} ({self.idling_key})")
        self.current_animation = self.idling_key


class AnimationControllerDummy(AnimationControllerBase):
    pass
