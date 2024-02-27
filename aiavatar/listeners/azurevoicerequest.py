# pip install azure-cognitiveservices-speech
from logging import getLogger, NullHandler
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import PropertyId
from . import RequestListenerBase

class AzureVoiceRequestListener(RequestListenerBase):
    def __init__(self, api_key: str, region: str, timeout: float=0.5, detection_timeout: float=10.0, lang: str="ja-JP", device_name: str=None):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
        self.speech_config.set_property(PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, str(detection_timeout * 1000))
        self.speech_config.set_property(PropertyId.Speech_SegmentationSilenceTimeoutMs, str(timeout * 1000))

        if device_name:
            # NOTE: You can see the way to check the device_name at Microsoft Learn.
            # https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/how-to-select-audio-input-devices
            self.audio_config = speechsdk.AudioConfig(device_name=device_name)
        else:
            self.audio_config = speechsdk.AudioConfig(use_default_microphone=True)

        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config, language=lang)

        self.on_start_listening = None

    async def get_request(self):
        if self.on_start_listening:
            await self.on_start_listening()

        self.logger.info(f"Listening... ({self.__class__.__name__})")
        result = self.speech_recognizer.recognize_once()

        if result.text:
            self.logger.info(f"AzureVoiceRequestListener: {result.text}")
        else:
            self.logger.info(f"AzureVoiceRequestListener: No speech recognized.")

        return result.text
