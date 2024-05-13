import aiohttp
from . import SpeechControllerBase, VoiceClip
from .soundplayers import SoundPlayerBase

class OpenAISpeechController(SpeechControllerBase):
    def __init__(self, *, api_key: str=None, base_url: str=None, voice: str="alloy", model: str="tts-1", speed: float=1.0, device_index: int=-1, playback_margin: float=0.1, use_subprocess=True, subprocess_timeout: float=5.0, sound_player: SoundPlayerBase=None):
        super().__init__(
            base_url=base_url,
            rate=None,
            device_index=device_index,
            playback_margin=playback_margin,
            use_subprocess=use_subprocess,
            subprocess_timeout=subprocess_timeout,
            sound_player=sound_player
        )
        self.api_key = api_key
        # alloy, echo, fable, onyx, nova, and shimmer
        self.voice = voice
        self.model = model
        self.speed = speed

    async def download(self, voice: VoiceClip):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        json = {
            "model": self.model,
            "voice": self.voice,
            "input": voice.text,
            "speed": self.speed,
            "response_format": "wav"
        }
        url = (self.base_url or "https://api.openai.com/v1") + "/audio/speech"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=json) as audio_resp:
                voice.audio_clip = await audio_resp.read()
