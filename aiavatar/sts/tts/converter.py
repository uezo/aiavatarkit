import asyncio
import logging
import struct
import time

logger = logging.getLogger(__name__)


class AudioConverter:
    def __init__(
        self,
        output_format: str = "wav",
        output_sample_rate: int = 16000,
        output_channels: int = 1,
        input_format: str = None,
        input_sample_rate: int = None,
        input_channels: int = None,
        debug: bool = False
    ):
        self.output_format = output_format
        self.output_sample_rate = output_sample_rate
        self.output_channels = output_channels
        self.input_format = input_format
        self.input_sample_rate = input_sample_rate
        self.input_channels = input_channels
        self.debug = debug

    async def convert(self, http_response) -> bytes:
        start_time = time.perf_counter()

        args = ["ffmpeg"]
        if self.input_format:
            args.extend(["-f", self.input_format])
        if self.input_sample_rate:
            args.extend(["-ar", str(self.input_sample_rate)])
        if self.input_channels:
            args.extend(["-ac", str(self.input_channels)])
        args.extend(["-i", "pipe:0"])
        args.extend(["-f", self.output_format])
        args.extend(["-ar", str(self.output_sample_rate)])
        args.extend(["-ac", str(self.output_channels)])
        args.append("pipe:1")

        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output_bytes, _ = await process.communicate(http_response.content)

        if self.debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"AudioConverter ({self.input_format} -> {self.output_format}): {elapsed_ms:.2f}ms")

        return output_bytes

    async def pcm_to_wave(self, http_response) -> bytes:
        start_time = time.perf_counter()

        data = http_response.content

        if len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            # Remove wave header from data if exists
            pos = 12
            while pos < len(data) - 8:
                if data[pos:pos+4] == b"data":
                    data = data[pos+8:]
                    break
                else:
                    chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
                    pos += 8 + chunk_size

        channels = self.input_channels or 1
        sample_rate = self.input_sample_rate or 16000
        bits_per_sample = 16   # 16bit audio

        data_size = len(data)
        file_size = data_size + 36
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",           # ChunkID
            file_size,         # ChunkSize
            b"WAVE",           # Format
            b"fmt ",           # Subchunk1ID
            16,                # Subchunk1Size (PCM)
            1,                 # AudioFormat (1 = PCM)
            channels,          # NumChannels
            sample_rate,       # SampleRate
            byte_rate,         # ByteRate
            block_align,       # BlockAlign
            bits_per_sample,   # BitsPerSample
            b"data",           # Subchunk2ID
            data_size          # Subchunk2Size
        )

        if self.debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"AudioConverter (PCM -> WAV): {elapsed_ms:.2f}ms")

        return header + data
