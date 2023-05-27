from typing import Callable
from . import SpeechListenerBase

class WakewordListener(SpeechListenerBase):
    def __init__(self, api_key: str, wakewords: list, on_wakeword: Callable, volume_threshold: int=3000, timeout: float=0.3, min_duration: float=0.2, max_duration: float=2, lang: str="ja-JP", rate: int=44100, device_index: int=-1):
        super().__init__(api_key, self.invoke_on_wakeword, volume_threshold, timeout, 0.0, min_duration, max_duration, lang, rate, device_index)
        self.wakewords = wakewords
        self.on_wakeword = on_wakeword
    
    async def invoke_on_wakeword(self, text: str):
        if text in self.wakewords:
            await self.on_wakeword(text)
