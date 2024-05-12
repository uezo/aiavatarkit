import aiohttp
from . import SpeechControllerBase, VoiceClip
from .soundplayers import SoundPlayerBase


class VoicevoxSpeechController(SpeechControllerBase):
    def __init__(self, *, base_url: str, speaker_id: int, device_index: int=-1, rate: int=None, playback_margin: float=0.1, use_subprocess=True, subprocess_timeout: float=5.0, sound_player: SoundPlayerBase=None):
        super().__init__(
            base_url=base_url,
            rate=rate,
            device_index=device_index,
            playback_margin=playback_margin,
            use_subprocess=use_subprocess,
            subprocess_timeout=subprocess_timeout,
            sound_player=sound_player
        )
        self.speaker_id = speaker_id

    async def download(self, voice: VoiceClip):
        params = {"speaker": self.speaker_id, "text": voice.text}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url + "/audio_query", params=params) as query_resp:
                audio_query = await query_resp.json()
                if self.rate is not None:
                    audio_query["outputSamplingRate"] = self.rate
                async with session.post(self.base_url + "/synthesis", params={"speaker": self.speaker_id}, json=audio_query) as audio_resp:
                    voice.audio_clip = await audio_resp.read()
