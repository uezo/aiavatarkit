from logging import getLogger, NullHandler
from io import BytesIO
import wave
import aiohttp
from .voicerequest import VoiceRequestListener
from .wakeword import WakewordListener


logger = getLogger(__name__)
logger.addHandler(NullHandler())


def to_wave_file(raw_audio, rate):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16bit
        wf.setframerate(rate)  # sample rate
        wf.writeframes(raw_audio)
    buffer.seek(0)
    return buffer


async def openai_transcribe(api_key: str, rate: int, audio_data: list) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    data = aiohttp.FormData()
    data.add_field("file", to_wave_file(audio_data, rate), filename="audio.wav", content_type="audio/wav")
    data.add_field("model", "whisper-1")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            data=data
        ) as resp:
            j = await resp.json()

            if resp.status != 200:
                logger.error(f"Failed in recognition: {resp.status}\n{j}")
                return None

            return j.get("text")


class OpenAIVoiceRequestListener(VoiceRequestListener):
    async def transcribe(self, audio_data: list) -> str:
        return await openai_transcribe(self.api_key, self.rate, audio_data)


class OpenAIWakewordListener(WakewordListener):
    async def transcribe(self, audio_data: list) -> str:
        return await openai_transcribe(self.api_key, self.rate, audio_data)
