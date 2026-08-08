import asyncio
import io
import wave

from .base import TTSPostprocessor


class WavSampleRatePostprocessor(TTSPostprocessor):
    @staticmethod
    def _is_wav(audio: bytes) -> bool:
        return (
            len(audio) >= 12
            and audio[:4] == b"RIFF"
            and audio[8:12] == b"WAVE"
        )

    @staticmethod
    def _resample(audio: bytes, sample_rate: int) -> bytes:
        with wave.open(io.BytesIO(audio), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            source_sample_rate = source.getframerate()
            compression_type = source.getcomptype()
            frames = source.readframes(source.getnframes())

        if source_sample_rate == sample_rate:
            return audio
        if compression_type != "NONE":
            raise ValueError("WAV sample rate conversion requires uncompressed PCM audio")

        import audioop

        if sample_width == 1:
            frames = audioop.bias(frames, sample_width, -128)
        resampled_frames, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            source_sample_rate,
            sample_rate,
            None,
        )
        if sample_width == 1:
            resampled_frames = audioop.bias(resampled_frames, sample_width, 128)

        output = io.BytesIO()
        with wave.open(output, "wb") as destination:
            destination.setnchannels(channels)
            destination.setsampwidth(sample_width)
            destination.setframerate(sample_rate)
            destination.writeframes(resampled_frames)
        return output.getvalue()

    async def process(self, audio: bytes, *, synthesizer) -> bytes:
        if synthesizer.sample_rate is None or not self._is_wav(audio):
            return audio
        return await asyncio.to_thread(
            self._resample,
            audio,
            synthesizer.sample_rate,
        )

    def get_cache_config(self, synthesizer) -> dict:
        if synthesizer.sample_rate is None:
            return None
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "sample_rate": synthesizer.sample_rate,
        }
