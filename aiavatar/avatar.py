import asyncio
from logging import getLogger, NullHandler
import re
from typing import Callable
from .speech import SpeechController
from .animation import AnimationController
from .face import FaceController

class AvatarRequest:
    def __init__(self, text_to_speech: str=None, animation_name: str=None, animation_duration: float = 3.0, face_name: str=None, face_duration: float=3.0):
        self.text_to_speech = text_to_speech
        self.animation_name = animation_name
        self.animation_duration = animation_duration
        self.face_name = face_name
        self.face_duration = face_duration


class AvatarController:
    def __init__(self, speech_controller: SpeechController, animation_controller: AnimationController, face_controller: FaceController, parser: Callable=None):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.speech_controller = speech_controller
        self.animation_controller = animation_controller
        self.animation_task = None
        self.face_controller = face_controller
        self.face_task = None
        self.parse = parser or self.parse_default
        self.requests = []

    async def start(self):
        # TODO: Stop exisiting tasks before start processing new requests

        while True:
            if len(self.requests) > 0:
                req = self.requests.pop(0)
                if req is None:
                    break
                await self.perform(req)
            else:
                await asyncio.sleep(0.01)

    def parse_default(self, text: str) -> AvatarRequest:
        avreq = AvatarRequest()

        # Face
        face_pattarn = r"\[face:(\w+)\]"
        faces = re.findall(face_pattarn, text)
        if faces:
            avreq.face_name = faces[0]
            avreq.face_duration = 4.0
            text = re.sub(face_pattarn, "", text)

        # Animation
        animation_pattarn = r"\[animation:(\w+)\]"
        animations = re.findall(animation_pattarn, text)
        if animations:
            avreq.animation_name = animations[0]
            avreq.animation_duration = 4.0
            text = re.sub(animation_pattarn, "", text)

        # Speech
        avreq.text_to_speech = text

        return avreq

    def set_text(self, text: str):
        avreq = self.parse(text)
        self.speech_controller.prefetch(avreq.text_to_speech)
        self.requests.append(avreq)

    def set_stop(self):
        self.requests.append(None)

    async def perform(self, avatar_request: AvatarRequest):
        # Face
        if avatar_request.face_name:
            if self.face_task:
                self.face_task.cancel()
            self.face_task = asyncio.create_task(
                self.face_controller.set_face(avatar_request.face_name, avatar_request.face_duration)
            )
        
        # Animation
        if avatar_request.animation_name:
            if self.animation_task:
                self.animation_task.cancel()
            self.animation_task = asyncio.create_task(
                self.animation_controller.animate(avatar_request.animation_name, avatar_request.animation_duration)
            )

        # Speech
        self.logger.info(avatar_request.text_to_speech)
        await self.speech_controller.speak(avatar_request.text_to_speech)

    def is_speaking(self) -> bool:
        return self.speech_controller.is_speaking()
