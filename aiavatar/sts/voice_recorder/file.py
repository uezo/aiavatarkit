from pathlib import Path
import aiofiles
from . import VoiceRecorder


class FileVoiceRecorder(VoiceRecorder):
    def __init__(self, *, record_dir: str = "recorded_voices", sample_rate = 16000, channels = 1, sample_width = 2):
        super().__init__(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
        self.record_dir = Path(record_dir)
        if not self.record_dir.exists():
            self.record_dir.mkdir(parents=True)

    async def save_voice(self, id: str, voice_bytes: bytes, audio_format: str):
        file_extension = self.to_extension(audio_format)
        async with aiofiles.open(self.record_dir / f"{id}.{file_extension}", "wb") as f:
            await f.write(voice_bytes)
