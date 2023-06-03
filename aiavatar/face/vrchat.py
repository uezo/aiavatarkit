import asyncio
from logging import getLogger, NullHandler
from time import time
from pythonosc import udp_client
from . import FaceController

class VRChatFaceController(FaceController):
    def __init__(self, osc_address: str="/avatar/parameters/FaceOSC", faces: dict=None, neutral_key: str="neutral", host: str="127.0.0.1", port: int=9000):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

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
        self.host = host
        self.port = port

        self.client = udp_client.SimpleUDPClient(self.host, self.port)

    async def set_face(self, name: str, duration: float):
        start_at = time()
        osc_value = self.faces.get(name)
        if osc_value is None:
            print(f"Face '{name}' is not registered")
            self.logger.warning(f"Face '{name}' is not registered")
            return

        self.logger.info(f"face: {name} ({osc_value})")
        self.client.send_message(self.osc_address, osc_value)
        while time() - start_at <= duration:
            await asyncio.sleep(0.1)
        self.logger.info(f"face: {self.neutral_key} ({self.faces[self.neutral_key]})")
        self.client.send_message(self.osc_address, self.faces[self.neutral_key])

    def test_osc(self):
        while True:
            self.set_face(input("Face name: "), 3.0)

if __name__ == "__main__":
    vrc_face_controller = VRChatFaceController()
    asyncio.run(vrc_face_controller.test_osc())
