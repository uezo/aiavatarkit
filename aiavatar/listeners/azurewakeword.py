# pip install azure-cognitiveservices-speech
import asyncio
from logging import getLogger, NullHandler
import queue
from threading import Thread
import time
import traceback
from typing import Callable
import azure.cognitiveservices.speech as speechsdk
from . import WakewordListenerBase

class AzureWakewordListener(WakewordListenerBase):
    def __init__(self, api_key: str, region: str, wakewords: list, on_wakeword: Callable, lang: str="ja-JP", device_name: str=None, restart_interval: float=5.0, verbose: bool=False):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.lang = lang
        self.speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)

        if device_name:
            # NOTE: You can see the way to check the device_name at Microsoft Learn.
            # https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/how-to-select-audio-input-devices
            self.audio_config = speechsdk.AudioConfig(device_name=device_name)
        else:
            self.audio_config = speechsdk.AudioConfig(use_default_microphone=True)

        self.speech_recognizer: speechsdk.SpeechRecognizer = None

        self.wakewords = wakewords
        self.on_wakeword = on_wakeword
        self.recognized_queue = queue.Queue()
        self.restart_interval = restart_interval
        self.verbose = verbose

        self.is_processing_wakeword = False

    def on_recognized(self, evt):
        recognized_text = evt.result.text.replace("。", "").replace("、", "").replace("!", "").replace("！", "").replace("?", "").replace("？", "").strip()

        if self.verbose:
            self.logger.info(f"AzureWakeWordListener: {recognized_text}")

        if recognized_text in self.wakewords:
            self.recognized_queue.put_nowait(recognized_text)

    async def on_wakeword_async(self, recognized_text: str):
        await self.on_wakeword(recognized_text)

    def enable_recognition(self):
        self.logger.info(f"Listening... ({self.__class__.__name__})")
        self.speech_recognizer.recognized.connect(lambda evt: self.on_recognized(evt))
    
    def disable_recognition(self):
        self.speech_recognizer.recognized.disconnect_all()

    def on_canceled(self, evt):
        self.logger.warning(f"Continuous recognition was canceled for some reasons.(e.g. network disconnection).: {evt}")
        self.disable_recognition()

        self.logger.info(f"Restart recognition after {self.restart_interval} seconds...")
        time.sleep(self.restart_interval)

        # Setup recognizer
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config, language=self.lang)
        self.speech_recognizer.canceled.connect(lambda evt: self.on_canceled(evt))

        # Start and enable recognition
        self.speech_recognizer.start_continuous_recognition()
        if not self.is_processing_wakeword:
            self.enable_recognition()

    async def start_listening(self):
        try:
            # Setup recognizer before start
            if self.speech_recognizer is None:
                self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config, language=self.lang)
                self.speech_recognizer.canceled.connect(lambda evt: self.on_canceled(evt))

            self.speech_recognizer.start_continuous_recognition()
            self.enable_recognition()
            while True:
                self.is_listening = True
                # Wait for queue
                recognized_text = self.recognized_queue.get()
                if recognized_text is None:
                    break

                # Process recognized text
                self.disable_recognition()
                self.is_processing_wakeword = True
                try:
                    await self.on_wakeword(recognized_text)
                except Exception as ex:
                    self.logger.error(f"Error at on_wake: {ex}\n{traceback.format_exc()}")
                finally:
                    self.is_processing_wakeword = False
                self.recognized_queue.task_done()
                self.enable_recognition()
        
        except Exception as ex:
            self.logger.error(f"Error at start_listening: {ex}\n{traceback.format_exc()}")

        finally:
            self.logger.info("Wakeword Listener stopped.")
            if self.speech_recognizer is not None:
                self.speech_recognizer.stop_continuous_recognition()
                self.speech_recognizer = None
            self.recognized_queue.task_done()
            self.is_listening = False

    def start(self):
        th = Thread(target=asyncio.run, args=(self.start_listening(),), daemon=True)
        th.start()
        return th

    def stop(self):
        self.recognized_queue.put(None)
        self.is_listening = False
