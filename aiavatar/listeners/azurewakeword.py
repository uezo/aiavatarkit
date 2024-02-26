# pip install azure-cognitiveservices-speech
import asyncio
from logging import getLogger, NullHandler
from threading import Thread
from typing import Callable
import azure.cognitiveservices.speech as speechsdk

class AzureWakewordListener:
    def __init__(self, api_key: str, region: str, wakewords: list, on_wakeword: Callable, lang: str="ja-JP", device_name: str=None, verbose: bool=False):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)

        if device_name:
            # NOTE: You can see the way to check the device_name at Microsoft Learn.
            # https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/how-to-select-audio-input-devices
            self.audio_config = speechsdk.AudioConfig(device_name=device_name)
        else:
            self.audio_config = speechsdk.AudioConfig(use_default_microphone=True)

        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config, language=lang)
        self.speech_recognizer.recognized.connect(lambda evt: self.on_recognized(evt))

        self.wakewords = wakewords
        self.on_wakeword = on_wakeword
        self.verbose = verbose
    
    def on_recognized(self, evt):
        recognized_text = evt.result.text.replace("。", "").replace("、", "").replace("!", "").replace("！", "").replace("?", "").replace("？", "").strip()

        if self.verbose:
            self.logger.info(f"AzureWakeWordListener: {recognized_text}")

        if recognized_text in self.wakewords:
            asyncio.run(self.on_wakeword(recognized_text))

    async def start_listening(self):
        self.logger.info(f"Listening... ({self.__class__.__name__})")
        self.speech_recognizer.start_continuous_recognition()
        while True:
            await asyncio.sleep(0.1)

    def start(self):
        th = Thread(target=asyncio.run, args=(self.start_listening(),), daemon=True)
        th.start()
        return th
