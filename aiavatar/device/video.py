import asyncio
import cv2

class VideoDevice:
    MAX_DEVICE_INDEX = 10

    def __init__(self, device_index: int=0, width: int=None, height: int=None, rate: float=None, wait: float=0.1):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.rate = rate
        self.wait = wait

    async def capture_image(self, filename: str=None):
        capture = cv2.VideoCapture(self.device_index)

        if self.width and self.height:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.rate:
            capture.set(cv2.CAP_PROP_FPS, self.rate)

        try:
            # Wait slightly before read frame from camera
            await asyncio.sleep(self.wait)
            ret, frame = capture.read()
            if ret:
                if filename:
                    cv2.imwrite(filename, frame)
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    return buffer.tobytes()
        finally:
            capture.release()

    @classmethod
    def get_video_devices(cls):
        devices = []
        for i in range(cls.MAX_DEVICE_INDEX):
            try:
                capture = cv2.VideoCapture(i)
                if capture and capture.isOpened():
                    devices.append({
                        "id": i,
                        "name": f"{capture.getBackendName()}",
                        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        "rate": capture.get(cv2.CAP_PROP_FPS)
                    })
            finally:
                capture.release()
        return devices
