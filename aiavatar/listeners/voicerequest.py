from . import SpeechListenerBase

class VoiceRequestListener(SpeechListenerBase):
    def __init__(self, api_key: str, volume_threshold: int=3000, timeout: float=1.0, detection_timeout: float=10.0, min_duration: float=0.3, max_duration: float=20.0, lang: str="ja-JP", rate: int=44100, device_index: int=-1):
        super().__init__(api_key, self.on_request, volume_threshold, timeout, detection_timeout, min_duration, max_duration, lang, rate, device_index)
        self.last_recognized_text = None

    async def on_request(self, text: str):
        self.last_recognized_text = text
        self.stop_listening()

    async def get_request(self):
        await self.start_listening()
        resp = self.last_recognized_text
        self.last_recognized_text = None
        return resp
