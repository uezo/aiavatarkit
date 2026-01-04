import asyncio
import logging
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
