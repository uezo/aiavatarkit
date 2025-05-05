from azure.storage.blob.aio import BlobServiceClient    # pip install azure-storage-blob
from . import VoiceRecorder

class AzureBlobVoiceRecorder(VoiceRecorder):
    def __init__(
        self,
        *,
        connection_string: str,
        container_name: str,
        directory: str = "recorded_voices",
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2
    ):
        super().__init__(sample_rate=sample_rate, channels=channels, sample_width=sample_width)

        self.connection_string = connection_string
        self.container_name = container_name
        self.directory = directory
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

    async def save_voice(self, id: str, voice_bytes: bytes, audio_format: str):
        file_extension = self.to_extension(audio_format)
        blob_name = f"{self.directory}/{id}.{file_extension}"
        blob_client = self.container_client.get_blob_client(blob_name)
        await blob_client.upload_blob(voice_bytes, overwrite=True)

    async def close(self):
        await self.blob_service_client.close()
