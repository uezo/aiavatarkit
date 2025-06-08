import base64
import logging
import httpx
from ...sts.vad import SpeechDetector, StandardSpeechDetector
from .. import AIAvatarRequest, AIAvatarResponse, AIAvatarException
from ..client import AIAvatarClientBase
from ...device import NoiseLevelDetector

logger = logging.getLogger(__name__)


class AIAvatarHttpClient(AIAvatarClientBase):
    def __init__(
        self,
        *,
        # STS Pipeline server
        url: str = "http://localhost:8000/chat",
        api_key: str = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        # Client configurations
        vad: SpeechDetector = None,
        vad_volume_db_threshold: float = -50.0,
        vad_silence_duration_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        face_controller = None,
        animation_controller = None,
        input_device_index = -1,
        input_sample_rate = 16000,
        input_channels = 1,
        input_chunk_size = 512,
        output_device_index = -1,
        output_chunk_size = 1024,
        audio_devices = None,
        cancel_echo = True,
        # Noise filter
        auto_noise_filter_threshold: bool = True,
        noise_margin: float = 20.0,
        debug = False,
    ):
        super().__init__(
            face_controller=face_controller,
            animation_controller=animation_controller,
            input_device_index=input_device_index,
            input_sample_rate=input_sample_rate,
            input_channels=input_channels,
            input_chunk_size=input_chunk_size,
            output_device_index=output_device_index,
            output_chunk_size=output_chunk_size,
            audio_devices=audio_devices,
            cancel_echo=cancel_echo,
            debug=debug
        )

        # VAD
        self.vad = vad or StandardSpeechDetector(
            volume_db_threshold=vad_volume_db_threshold,
            silence_duration_threshold=vad_silence_duration_threshold,
            sample_rate=vad_sample_rate,
            debug=debug
        )

        @self.vad.on_speech_detected
        async def on_speech_detected(data: bytes, recorded_duration: float, session_id: str):
            await self.send_request(AIAvatarRequest(
                type="start",
                session_id=session_id,
                user_id=self.vad.get_session_data(session_id, "user_id"),
                context_id=self.vad.get_session_data(session_id, "context_id"),
                text=None,
                audio_data=data,
                files=[],
                system_prompt_params={},
                metadata={}
            ))

        # HTTP Client
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )

        # Noise filter
        self.auto_noise_filter_threshold = auto_noise_filter_threshold
        self.noise_margin = noise_margin

        self.url = url
        self.api_key = api_key

    async def send_request(self, request: AIAvatarRequest):
        if request.audio_data and isinstance(request.audio_data, bytes):
            request.audio_data = base64.b64encode(request.audio_data).decode("utf-8")

        async with self.http_client.stream(
            method="post",
            url=self.url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
            json=request.model_dump()
        ) as response:
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {await response.aread()}")
                response.raise_for_status()

            async for chunk in response.aiter_lines():
                if chunk.startswith("data:"):
                    response = AIAvatarResponse.model_validate_json(chunk[5:].strip())

                    if response.type == "start":
                        logger.info(f"User: {response.metadata.get('request_text')}")
                        self.vad.set_session_data(request.session_id, "context_id", response.context_id)

                    elif response.type == "chunk" and response.audio_data:
                        if response.voice_text:
                            logger.info(f"AI: {response.voice_text}")
                        response.audio_data = base64.b64decode(response.audio_data)

                    elif response.type == "error":
                        raise AIAvatarException(
                            message=response.metadata.get("error", "Error in processing pipeline"),
                            response=response
                        )

                    await self.response_queue.put(response)

    async def send_microphone_data(self, audio_bytes, session_id):
        await self.vad.process_samples(samples=audio_bytes, session_id=session_id)

    async def initialize_session(self, session_id: str, user_id: str, context_id: str):
        if user_id:
            self.vad.set_session_data(session_id, "user_id", user_id, True)
        if context_id:
            self.vad.set_session_data(session_id, "context_id", context_id, True)

    async def start_listening(self, session_id: str = "http_session", user_id: str = "http_user", context_id: str = None):
        # Set noise filter
        if hasattr(self.vad, "set_volume_db_threshold"):
            if self.auto_noise_filter_threshold:
                noise_level_detector = NoiseLevelDetector(
                    rate=self.audio_recorder.sample_rate,
                    channels=self.audio_recorder.channels,
                    device_index=self.audio_devices.input_device
                )
                noise_level = noise_level_detector.get_noise_level()
                volume_threshold_db = int(noise_level) + self.noise_margin

                logger.info(f"Set volume threshold: {volume_threshold_db}dB")
                self.vad.set_volume_db_threshold(session_id, volume_threshold_db)
            else:
                logger.info(f"Set volume threshold: {self.vad.volume_db_threshold}dB")

        await super().start_listening(session_id, user_id, context_id)
