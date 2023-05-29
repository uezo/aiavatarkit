import aiohttp
import asyncio
import io
from logging import getLogger, NullHandler
import traceback
import wave
import pyaudio
from . import SpeechController

class VoiceClip:
    def __init__(self, text: str):
        self.text = text
        self.download_task = None
        self.audio_clip = None


class VoicevoxSpeechController(SpeechController):
    def __init__(self, base_url: str, speaker_id: int, device_index: int=-1):
        self.logger = getLogger(__name__)
        self.logger.addHandler(NullHandler())

        self.base_url = base_url
        self.speaker_id = speaker_id
        self.device_index = device_index
        self.voice_clips = {}
        self.pa = pyaudio.PyAudio()
        self._is_speaking = False

    async def download(self, voice: VoiceClip):
        params = {"speaker": self.speaker_id, "text": voice.text}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url + "/audio_query", params=params) as query_resp:
                audio_query = await query_resp.json()
                async with session.post(self.base_url + "/synthesis", params={"speaker": self.speaker_id}, json=audio_query) as audio_resp:
                    voice.audio_clip = await audio_resp.read()

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
        
        with wave.open(io.BytesIO(voice.audio_clip), "rb") as f:
            def stream_callback(in_data, frame_count, time_info, status):
                data = f.readframes(frame_count)
                return (data, pyaudio.paContinue)

            stream = self.pa.open(format=self.pa.get_format_from_width(width=f.getsampwidth()),
                output_device_index=self.device_index,
                channels=f.getnchannels(),
                rate=f.getframerate(),
                frames_per_buffer=1024,
                output=True,
                stream_callback=stream_callback,
            )

            try:
                self._is_speaking = True
                stream.start_stream()
                while stream.is_active():
                    await asyncio.sleep(0.1)

            except Exception as ex:
                self.logger.error(f"Error at speaking: {str(ex)}\n{traceback.format_exc()}")

            finally:
                self._is_speaking = False
                stream.stop_stream()
                stream.close()

    def is_speaking(self) -> bool:
        return self._is_speaking
