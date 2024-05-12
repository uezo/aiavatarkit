from aiavatar import AIAvatar
from aiavatar.animation import AnimationController, AnimationControllerDummy
from aiavatar.face import FaceController, FaceControllerDummy
from aiavatar.listeners import RequestListenerBase, WakewordListenerBase
from aiavatar.processors import ChatProcessor
from aiavatar.speech import SpeechController
from aiavatar.speech.voicevox import VoicevoxSpeechController

def test_init():
    app = AIAvatar(
        openai_api_key="OPENAI_API_KEY",
        google_api_key="GOOGLE_API_KEY"
    )

    # Audio device
    assert app.audio_devices.input_device_info["max_input_channels"] > 0
    assert app.audio_devices.output_device_info["max_output_channels"] > 0

    # Chat processor
    assert app.chat_processor.api_key == "OPENAI_API_KEY"
    assert app.chat_processor.model == "gpt-3.5-turbo"
    
    # Request Listener
    assert app.request_listener.api_key == "GOOGLE_API_KEY"
    assert app.request_listener.volume_threshold > -80
    assert app.request_listener.volume_threshold < -30
    assert app.request_listener.device_index == app.audio_devices.input_device
    assert app.request_listener.lang == "ja-JP"

    # Wakeword Listener
    assert app.wakeword_listener.api_key == "GOOGLE_API_KEY"
    assert app.wakeword_listener.wakewords == ["こんにちは"]
    assert app.wakeword_listener.volume_threshold > -80
    assert app.wakeword_listener.volume_threshold < -30
    assert app.wakeword_listener.device_index == app.audio_devices.input_device
    assert app.wakeword_listener.lang == "ja-JP"
    assert app.wakeword_listener.on_wakeword is not None

    # Avatar Controller with Speech, Animation and Face
    assert app.avatar_controller is not None
    speech_controller: VoicevoxSpeechController = app.avatar_controller.speech_controller
    assert speech_controller.base_url == "http://127.0.0.1:50021"
    assert speech_controller.speaker_id == 46
    assert speech_controller.device_index == app.audio_devices.output_device
    assert isinstance(app.avatar_controller.animation_controller, AnimationControllerDummy) is True
    assert isinstance(app.avatar_controller.face_controller, FaceControllerDummy) is True

    # Chat
    assert app.start_voice is None
    assert app.split_chars == ["。", "、", "？", "！", ".", ",", "?", "!"]
    assert app.on_turn_end == app.on_turn_end_default

def test_init_with_args():
    app = AIAvatar(
        openai_api_key="OPENAI_API_KEY",
        google_api_key="GOOGLE_API_KEY",
        model="gpt-4-turbo",
        system_message_content="You are a cat.",
        voicevox_speaker_id=2,
        wakewords=["もしもし", "はろー"],
        start_voice="どうしたの",
        split_chars=["-", "_"],
        language="en-US",
        verbose=True
    )

    assert app.chat_processor.model == "gpt-4-turbo"
    assert app.chat_processor.system_message_content == "You are a cat."
    assert app.request_listener.lang == "en-US"
    assert app.wakeword_listener.wakewords == ["もしもし", "はろー"]
    assert app.wakeword_listener.lang == "en-US"
    assert app.wakeword_listener.verbose is True
    assert app.avatar_controller.speech_controller.speaker_id == 2
    assert app.avatar_controller.animation_controller.verbose is True
    assert app.avatar_controller.face_controller.verbose is True
    assert app.start_voice == "どうしたの"
    assert app.split_chars == ["-", "_"]

def test_init_with_components():
    class MyChatProcessor(ChatProcessor):
        async def chat(self, text: str): ...

    class MyRequestListener(RequestListenerBase):
        async def get_request(self): ...

    class MyWakewordListener(WakewordListenerBase):
        async def start(self): ...
        async def stop(self): ...

    class MySpeechController(SpeechController):
        def prefetch(self, text: str): ...
        async def speak(self, text: str): ...
        def is_speaking(self) -> bool: ...

    class MyAnimationController(AnimationController):
        async def animate(self, name: str, duration: float): ...

    class MyFaceController(FaceController):
        async def set_face(self, name: str, duration: float): ...
        def reset(self): ...

    app = AIAvatar(
        chat_processor=MyChatProcessor(),
        request_listener=MyRequestListener(),
        wakeword_listener=MyWakewordListener(),
        speech_controller=MySpeechController(),
        animation_controller=MyAnimationController(),
        face_controller=MyFaceController()
    )

    assert isinstance(app.chat_processor, MyChatProcessor) is True
    assert isinstance(app.request_listener, MyRequestListener) is True
    assert isinstance(app.wakeword_listener, MyWakewordListener) is True
    assert isinstance(app.avatar_controller.speech_controller, MySpeechController) is True
    assert isinstance(app.avatar_controller.animation_controller, MyAnimationController) is True
    assert isinstance(app.avatar_controller.face_controller, MyFaceController) is True
