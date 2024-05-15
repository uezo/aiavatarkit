from abc import ABC, abstractmethod
import asyncio
from logging import getLogger, NullHandler
import traceback
from .soundplayers import SoundPlayerBase
from .soundplayers.sounddevice_player import SoundDevicePlayer


class SpeechController(ABC):
    @abstractmethod
    def prefetch(self, text: str):
        pass

    @abstractmethod
    async def speak(self, text: str):
        pass

    @abstractmethod
    def clear_cache(self):
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        pass


class VoiceClip:
    def __init__(self, text: str):
        self.text = text
        self.download_task = None
        self.audio_clip = None


class SpeechControllerBase(SpeechController):
    def __init__(self, *, base_url: str, rate: int=None, device_index: int=-1, playback_margin: float=0.1, use_subprocess=True, subprocess_timeout: float=0.5, sound_player: SoundPlayerBase=None):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.base_url = base_url
        self.rate = rate
        self.device_index = device_index
        self.playback_margin = playback_margin
        self.subprocess_timeout = subprocess_timeout
        self.use_subprocess = use_subprocess

        if sound_player:
            self.sound_player = sound_player
        else:
            self.sound_player = SoundDevicePlayer(
                device_index=self.device_index,
                playback_margin=self.playback_margin,
                subprocess_timeout=self.subprocess_timeout
            )

        self.voice_clips = {}
        self._is_speaking = False

    async def download(self, voice: VoiceClip):
        raise NotImplementedError("`download` is not implemented")

    def prefetch(self, text: str):
        v = self.voice_clips.get(text)
        if v:
            return v

        v = VoiceClip(text)
        v.download_task = asyncio.create_task(self.download(v))
        self.voice_clips[text] = v
        return v

    async def speak(self, text: str):
        voice = self.prefetch(text)
        
        if not voice.audio_clip:
            await voice.download_task
        
        try:
            self._is_speaking = True

            if self.use_subprocess:
                await self.sound_player.play_wave_on_subprocess(voice.audio_clip)
            else:
                await self.sound_player.play_wave(voice.audio_clip)

        except Exception as ex:
            self.logger.error(f"Error at speaking: {str(ex)}\n{traceback.format_exc()}")

        finally:
            self._is_speaking = False

    def clear_cache(self):
        self.voice_clips.clear()

    def is_speaking(self) -> bool:
        return self._is_speaking
