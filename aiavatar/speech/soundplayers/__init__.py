from abc import ABC, abstractmethod

class SoundPlayerBase(ABC):
    @abstractmethod
    async def play_wave(self, data):
        pass

    @abstractmethod
    async def play_wave_on_subprocess(self, data):
        pass
