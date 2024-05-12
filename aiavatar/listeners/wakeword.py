import asyncio
from threading import Thread
from typing import Callable
from . import WakewordListenerBase, SpeechListenerBase

class WakewordListener(WakewordListenerBase, SpeechListenerBase):
    def __init__(self, api_key: str, wakewords: list, on_wakeword: Callable, volume_threshold: int=-50, timeout: float=0.2, min_duration: float=0.2, max_duration: float=2, lang: str="ja-JP", rate: int=16000, chennels: int=1, device_index: int=-1, verbose: bool=False):
        super().__init__(api_key, self.invoke_on_wakeword, volume_threshold, timeout, 0.0, min_duration, max_duration, lang, rate, chennels, device_index)
        self.wakewords = wakewords
        self.on_wakeword = on_wakeword
        self.verbose = verbose
    
    async def invoke_on_wakeword(self, text: str):
        if self.verbose:
            self.logger.info(f"Recognized: {text}")

        if text in self.wakewords:
            await self.on_wakeword(text)

    def start(self):
        th = Thread(target=asyncio.run, args=(self.start_listening(),), daemon=True)
        th.start()
        return th

    def stop(self):
        self.is_listening = False
