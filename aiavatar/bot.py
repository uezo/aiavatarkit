import asyncio
from logging import getLogger, NullHandler
import traceback
from typing import List, Callable
# Device
from .device import AudioDevice
# Processor
from .processors.chatgpt import ChatGPTProcessor
# Listener
from .listeners.wakeword import WakewordListener
from .listeners.voicerequest import VoiceRequestListener
# Avatar
from .speech.voicevox import VoicevoxSpeechController
from .animation import AnimationController, AnimationControllerDummy
from .face import FaceController, FaceControllerDummy
from .avatar import AvatarController

class AIAvatar:
    def __init__(
        self,
        google_api_key: str,
        openai_api_key: str,
        voicevox_url: str,
        voicevox_speaker_id: int=46,
        wakewords: List[str]=None,
        wakevoice: str="どうしたの",
        system_message_content: str=None,
        animation_controller: AnimationController=None,
        face_controller: FaceController=None,
        avatar_request_parser: Callable=None,
        input_device: int=-1,
        output_device: int=-1
    ):

        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.google_api_key = google_api_key
        self.openai_api_key = openai_api_key
        self.voicevox_url = voicevox_url
        self.voicevox_speaker_id = voicevox_speaker_id

        # Audio Devices
        if input_device < 0:
            input_device_info = AudioDevice.get_default_input_device_info()
            input_device = input_device_info["index"]
        else:
            input_device_info = AudioDevice.get_device_info(input_device)
        self.input_device = input_device
        self.logger.info(f"Input device: [{input_device}] {input_device_info['name']}")

        if output_device < 0:
            output_device_info = AudioDevice.get_default_output_device_info()
            output_device = output_device_info["index"]
        else:
            output_device_info = AudioDevice.get_device_info(output_device)
        self.output_device = output_device
        self.logger.info(f"Output device: [{output_device}] {output_device_info['name']}")

        # Processor
        self.chat_processor = ChatGPTProcessor(self.openai_api_key, system_message_content=system_message_content)

        # Listeners
        self.request_listener = VoiceRequestListener(self.google_api_key, device_index=self.input_device)
        self.wakewords = wakewords or ["こんにちは"]
        self.wakevoice = wakevoice
        async def on_ww(text):
            self.logger.info(f"Wakeword: {text}")
            await self.chat()
        self.wakeword_listener = WakewordListener(self.google_api_key, self.wakewords, on_ww, device_index=self.input_device)

        # Avatar
        speech_controller = VoicevoxSpeechController(self.voicevox_url, self.voicevox_speaker_id, device_index=self.output_device)
        animation_controller = animation_controller or AnimationControllerDummy()
        face_controller = face_controller or FaceControllerDummy()
        self.avatar_controller = AvatarController(speech_controller, animation_controller, face_controller, avatar_request_parser)

    async def chat(self):
        try:
            await self.avatar_controller.speech_controller.speak(self.wakevoice)
        except Exception as ex:
            self.logger.error(f"Error at starting chat: {str(ex)}\n{traceback.format_exc()}")

        while True:
            try:
                req = await self.request_listener.get_request()
                if not req:
                    break

                self.logger.info(f"User: {req}")
                self.logger.info("AI:")

                avatar_task = asyncio.create_task(self.avatar_controller.start())

                stream_buffer = ""
                async for t in self.chat_processor.chat(req):
                    stream_buffer += t
                    sp = stream_buffer.replace("。", "。|").replace("、", "、|").replace("！", "！|").replace("？", "？|").split("|")
                    if len(sp) > 1: # >1 means `|` is found (splited at the end of sentence)
                        sentence = sp.pop(0)
                        stream_buffer = "".join(sp)
                        self.avatar_controller.set_text(sentence)

                self.avatar_controller.set_stop()
                await avatar_task
            
            except Exception as ex:
                self.logger.error(f"Error at chatting loop: {str(ex)}\n{traceback.format_exc()}")
    

    async def start(self):
        await self.wakeword_listener.start_listening()
