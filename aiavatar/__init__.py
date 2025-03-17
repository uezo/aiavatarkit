import logging
from .adapter import AvatarControlRequest, AIAvatarRequest, AIAvatarResponse

logger = logging.getLogger(__name__)


try:
    from .adapter.client import AIAvatarClientBase
    from .adapter.local.client import AIAvatar
except ModuleNotFoundError as mnferr:
    if "pyaudio" in mnferr.msg:
        logger.warning("PyAudio is not found in this environment. Import PortAudio and `pip install pyaudio` to use audio devices.")
    else:
        raise
