import asyncio
from time import time
from pythonosc import udp_client
from . import FaceControllerBase

class VRChatFaceController(FaceControllerBase):
    def __init__(self, osc_address: str="/avatar/parameters/FaceOSC", faces: dict=None, neutral_key: str="neutral", host: str="127.0.0.1", port: int=9000, debug: bool=False):
        super().__init__(debug)

        self.osc_address = osc_address
        self.faces = faces or {
            "neutral": 0,
            "joy": 1,
            "angry": 2,
            "sorrow": 3,
            "fun": 4,
            "surprise": 5,
        }
        self.neutral_key = neutral_key
        self._current_face = self.neutral_key
        self.host = host
        self.port = port

        self.client = udp_client.SimpleUDPClient(self.host, self.port)

    async def set_face(self, name: str, duration: float):
        if duration > 0:
            self.subscribe_reset(time() + duration)

        osc_value = self.faces.get(name)
        if osc_value is None:
            self.logger.warning(f"Face '{name}' is not registered")
            return

        self.logger.info(f"face: {name} ({osc_value})")
        self.client.send_message(self.osc_address, osc_value)
        self.current_face = name

    def reset(self):
        self.logger.info(f"Reset face: {self.neutral_key} ({self.faces[self.neutral_key]})")
        self.client.send_message(self.osc_address, self.faces[self.neutral_key])
        self.current_face = self.neutral_key

    def test_osc(self):
        while True:
            self.set_face(input("Face name: "), 3.0)

if __name__ == "__main__":
    vrc_face_controller = VRChatFaceController()
    asyncio.run(vrc_face_controller.test_osc())
