from abc import ABC, abstractmethod
import io
import wave
from typing import List
import httpx
import logging

logger = logging.getLogger(__name__)


class SpeechRecognizer(ABC):
    def __init__(
        self,
        *,
        language: str = None,
        alternative_languages: List[str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        self.language = language
        self.alternative_languages = alternative_languages or []
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )

        self.debug = debug

    @abstractmethod
    async def transcribe(self, data: bytes) -> str:
        pass

    async def close(self):
        await self.http_client.aclose()

    def downsample(self, audio_bytes: bytes, sample_rate: int, target_sample_rate: int) -> bytes:
        if target_sample_rate <= 0 or sample_rate == target_sample_rate:
            return audio_bytes
 
        if sample_rate < target_sample_rate:
            logger.warning(f"Cannot upsample from {sample_rate}Hz to {target_sample_rate}Hz")
            return audio_bytes
            
        # Convert bytes to 16-bit signed integers
        import struct
        audio_data = struct.unpack(f'<{len(audio_bytes)//2}h', audio_bytes)
        
        # Calculate decimation factor
        decimation_factor = sample_rate // target_sample_rate
        
        if decimation_factor <= 1:
            # Use linear interpolation for non-integer ratios
            ratio = sample_rate / target_sample_rate
            output_length = int(len(audio_data) / ratio)
            downsampled = []
            
            for i in range(output_length):
                source_index = i * ratio
                index = int(source_index)
                
                if index + 1 < len(audio_data):
                    # Linear interpolation between adjacent samples
                    fraction = source_index - index
                    interpolated = audio_data[index] * (1 - fraction) + audio_data[index + 1] * fraction
                    downsampled.append(int(interpolated))
                else:
                    downsampled.append(audio_data[index] if index < len(audio_data) else 0)
        else:
            # Simple decimation for integer ratios
            downsampled = audio_data[::decimation_factor]
        
        # Convert back to bytes
        return struct.pack(f'<{len(downsampled)}h', *downsampled)

    def to_wave_file(self, audio_bytes: bytes, sample_rate: int):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        buffer.seek(0)
        return buffer


class SpeechRecognizerDummy(SpeechRecognizer):
    async def transcribe(self, data: bytes) -> str:
        pass
