import logging

logger = logging.getLogger("aiavatar")
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
logger.addHandler(streamHandler)

# Device
from .device import AudioDevice
# Processor
from .processors.chatgpt import ChatGPTProcessor
# Listener
from .listeners import WakewordListenerBase
from .listeners import RequestListenerBase
from .listeners.wakeword import WakewordListener
from .listeners.voicerequest import VoiceRequestListener
# Avatar
from .speech.voicevox import VoicevoxSpeechController
from .avatar import AvatarController
# Bot
from .bot import AIAvatar
