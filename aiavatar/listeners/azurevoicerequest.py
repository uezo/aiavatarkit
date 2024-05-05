# pip install azure-cognitiveservices-speech
import asyncio
from logging import getLogger, NullHandler
import time
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import PropertyId
from . import RequestListenerBase

class AzureVoiceRequestListener(RequestListenerBase):
    def __init__(self, api_key: str, region: str, timeout: float=0.5, detection_timeout: float=10.0, lang: str="ja-JP", device_name: str=None):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.lang = lang
        self.detection_timeout = detection_timeout
        self.speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
        self.speech_config.set_property(PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, str(detection_timeout * 1000))
        self.speech_config.set_property(PropertyId.Speech_SegmentationSilenceTimeoutMs, str(timeout * 1000))

        if device_name:
            # NOTE: You can see the way to check the device_name at Microsoft Learn.
            # https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/how-to-select-audio-input-devices
            self.audio_config = speechsdk.AudioConfig(device_name=device_name)
        else:
            self.audio_config = speechsdk.AudioConfig(use_default_microphone=True)

        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config, language=self.lang)
        self.recognizer_wait_time = 0.3

        self.on_start_listening = None

    async def get_request(self):
        if self.on_start_listening:
            await self.on_start_listening()

        self.logger.info(f"Listening... ({self.__class__.__name__})")

        start_at = time.time()
        while True:
            result = self.speech_recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.Canceled:
                self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config, language=self.lang)

            if result.text:
                self.logger.info(f"AzureVoiceRequestListener: {result.text}")
                return result.text
            else:
                elapsed = time.time() - start_at
                if elapsed < self.detection_timeout:
                    self.logger.info(f"AzureVoiceRequestListener: Noise detected. Retrying... (elapsed: {elapsed})")
                    await asyncio.sleep(self.recognizer_wait_time)    # Wait a bit before calling recognize_once() again. 0.2 was too short on my environment.
                else:
                    self.logger.info(f"AzureVoiceRequestListener: No speech recognized.")
                    return ""
