import aiohttp
from . import SpeechControllerBase, VoiceClip
from .soundplayers import SoundPlayerBase


class AzureSpeechController(SpeechControllerBase):
    def __init__(self, api_key: str, region: str, *, speaker_name: str="en-US-JennyMultilingualNeural", speaker_gender: str="Female", lang: str="ja-JP", base_url: str=None, device_index: int=-1, playback_margin: float=0.1, use_subprocess=True, subprocess_timeout: float=5.0, sound_player: SoundPlayerBase=None):
        super().__init__(
            base_url=base_url or "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
            device_index=device_index,
            playback_margin=playback_margin,
            use_subprocess=use_subprocess,
            subprocess_timeout=subprocess_timeout,
            sound_player=sound_player
        )
        self.api_key = api_key
        self.region = region
        # Speaker list: https://learn.microsoft.com/ja-jp/azure/ai-services/speech-service/language-support?tabs=tts
        self.speaker_name = speaker_name
        self.speaker_gender = speaker_gender
        self.lang = lang

    async def download(self, voice: VoiceClip):
        url = self.base_url.format(region=self.region)
        headers = {
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            "Content-Type": "application/ssml+xml",
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        ssml_text = f"<speak version='1.0' xml:lang='{self.lang}'><voice xml:lang='{self.lang}' xml:gender='{self.speaker_gender}' name='{self.speaker_name}'>{voice.text}</voice></speak>"
        data = ssml_text.encode("utf-8")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 200:
                    voice.audio_clip = await response.read()
