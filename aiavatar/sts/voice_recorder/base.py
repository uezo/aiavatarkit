from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import logging
import struct
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Voice:
    transaction_id: str


@dataclass
class RequestVoice(Voice):
    voice_bytes: bytes


@dataclass
class ResponseVoices(Voice):
    voice_chunks: List[bytes]
    audio_format: str


class VoiceRecorder(ABC):
    def __init__(self, *, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        self.format_extension_mapper = {
            "LINEAR16": "wav",  # Google TTS
            "riff-16khz-16bit-mono-pcm": "wav"  # Azure TTS
        }

        self.queue: asyncio.Queue[Voice] = asyncio.Queue()
        self.worker_task = None

    def to_extension(self, format: str) -> str:
        return self.format_extension_mapper.get(format) or format

    def create_wav_header(self, data_size: int, sample_rate: int, channels: int, sample_width: int) -> bytes:
        byte_rate = sample_rate * channels * sample_width
        block_align = channels * sample_width
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",                # ChunkID
            36 + data_size,         # ChunkSize = 36 + SubChunk2Size
            b"WAVE",                # Format
            b"fmt ",                # Subchunk1ID
            16,                     # Subchunk1Size (PCM)
            1,                      # AudioFormat (PCM: 1)
            channels,               # NumChannels
            sample_rate,            # SampleRate
            byte_rate,              # ByteRate
            block_align,            # BlockAlign
            sample_width * 8,       # BitsPerSample
            b"data",                # Subchunk2ID
            data_size               # Subchunk2Size
        )
        return header

    @abstractmethod
    async def save_voice(self, id: str, voice_bytes: bytes, audio_format: str):
        pass

    async def _worker(self):
        while True:
            voice = await self.queue.get()
            if voice is None:
                break

            try:
                if isinstance(voice, RequestVoice):
                    if not voice.voice_bytes.startswith(b"RIFF"):
                        # Add header if missing
                        header = self.create_wav_header(
                            data_size=len(voice.voice_bytes),
                            sample_rate=self.sample_rate,
                            channels=self.channels,
                            sample_width=self.sample_width
                        )
                        voice.voice_bytes = header + voice.voice_bytes
                    await self.save_voice(
                        id=f"{voice.transaction_id}_request",
                        voice_bytes=voice.voice_bytes,
                        audio_format="wav"
                    )

                elif isinstance(voice, ResponseVoices):
                    for idx, v in enumerate(voice.voice_chunks):
                        await self.save_voice(
                            id=f"{voice.transaction_id}_response_{idx}",
                            voice_bytes=v,
                            audio_format=voice.audio_format
                        )

            except Exception as ex:
                logger.error(f"Error at saving voice: {ex}")

            finally:
                if not self.queue.empty():
                    self.queue.task_done()

    async def record(self, voice: Voice):
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker())
        await self.queue.put(voice)

    async def stop(self):
        await self.queue.put(None)
        if self.worker_task:
            self.worker_task.cancel()
