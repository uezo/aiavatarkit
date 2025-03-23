import base64
import logging
import re
from typing import List
from fastapi import WebSocket
from litests.models import STSRequest, STSResponse
from litests.pipeline import LiteSTS
from litests.adapter.websocket import WebSocketAdapter, WebSocketSessionData
from litests.vad import SpeechDetector
from litests.stt import SpeechRecognizer
from litests.stt.openai import OpenAISpeechRecognizer
from litests.llm import LLMService
from litests.tts import SpeechSynthesizer
from litests.performance_recorder import PerformanceRecorder
from ..models import AvatarControlRequest, AIAvatarRequest, AIAvatarResponse

logger = logging.getLogger(__name__)


class WebSocketSessionData:
    def __init__(self):
        self.id = None
        self.data = {}


class AIAvatarWebSocketServer(WebSocketAdapter):
    def __init__(
        self,
        *,
        sts: LiteSTS = None,
        # STS Pipeline components
        vad: SpeechDetector = None,
        stt: SpeechRecognizer = None,
        llm: LLMService = None,
        tts: SpeechSynthesizer = None,
        # STS Pipeline params for default components
        volume_db_threshold: float = -50.0,
        silence_duration_threshold: float = 0.5,
        input_sample_rate: int = 16000,
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-4o-mini",
        system_prompt: str = None,
        voicevox_url: str = "http://127.0.0.1:50021",
        voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        wakeword_timeout: float = 60.0,
        performance_recorder: PerformanceRecorder = None,
        # Debug
        debug: bool = False
    ):
        # Speech-to-Speech pipeline
        self.sts = sts or LiteSTS(
            vad=vad,
            vad_volume_db_threshold=volume_db_threshold,
            vad_silence_duration_threshold=silence_duration_threshold,
            vad_sample_rate=input_sample_rate,
            stt=stt or OpenAISpeechRecognizer(
                openai_api_key=openai_api_key,
                sample_rate=input_sample_rate
            ),
            llm=llm,
            llm_openai_api_key=openai_api_key,
            llm_base_url=openai_base_url,
            llm_model=openai_model,
            llm_system_prompt=system_prompt,
            tts=tts,
            tts_voicevox_url=voicevox_url,
            tts_voicevox_speaker=voicevox_speaker,
            wakewords=wakewords,
            wakeword_timeout=wakeword_timeout,
            performance_recorder=performance_recorder,
            debug=debug
        )

        # Call base after self.sts is set
        super().__init__(self.sts)

        # Debug
        self.debug = debug
        self.last_response = None

    # Request
    async def process_websocket(self, websocket: WebSocket, session_data: WebSocketSessionData):
        data = await websocket.receive_text()
        request = AIAvatarRequest.model_validate_json(data)

        if not request.session_id:
            await websocket.send_text(AIAvatarResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"error": "WebSocket disconnect: session_id is required."}
            ).model_dump_json())
            logger.info("WebSocket disconnect: session_id is required.")
            await websocket.close()
            return

        if request.type == "start":
            self.websockets[request.session_id] = websocket

            logger.info(f"WebSocket connected for session: {request.session_id}")

            # Store session data to initialize
            if request.user_id:
                self.sts.vad.set_session_data(request.session_id, "user_id", request.user_id, True)
            if request.context_id:
                self.sts.vad.set_session_data(request.session_id, "context_id", request.context_id, True)
            session_data.data["metadata"] = request.metadata
            self.sessions[session_data.id] = session_data

            await self.send_response(AIAvatarResponse(
                type="connected",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id
            ))

        elif request.type == "invoke":
            async for r in self.sts.invoke(STSRequest(
                type=request.type,
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                text=request.text,
                audio_data=request.audio_data,
                files=request.files
            )):
                await self.sts.handle_response(r)

        elif request.type == "data":
            audio_data = base64.b64decode(request.audio_data)
            await self.sts.vad.process_samples(audio_data, request.session_id)

        elif request.type == "config":
            volume_db_threshold = request.metadata.get("volume_db_threshold")
            if volume_db_threshold:
                self.sts.vad.set_volume_db_threshold(request.session_id, volume_db_threshold)

        elif request.type == "stop":
            logger.info(f"WebSocket disconnect for session: {request.session_id}")
            await websocket.close()

    # Response
    def parse_avatar_control_request(self, text: str) -> AvatarControlRequest:
        avreq = AvatarControlRequest()

        if not text:
            return avreq

        # Face
        face_pattarn = r"\[face:(\w+)\]"
        faces = re.findall(face_pattarn, text)
        if faces:
            avreq.face_name = faces[0]
            avreq.face_duration = 4.0

        # Animation
        animation_pattarn = r"\[animation:(\w+)\]"
        animations = re.findall(animation_pattarn, text)
        if animations:
            avreq.animation_name = animations[0]
            avreq.animation_duration = 4.0

        return avreq

    async def send_response(self, aiavatar_response: AIAvatarResponse):
        await self.websockets[aiavatar_response.session_id].send_text(
            aiavatar_response.model_dump_json()
        )

    async def handle_response(self, response: STSResponse):
        aiavatar_response = AIAvatarResponse(
            type=response.type,
            session_id=response.session_id,
            user_id=response.user_id,
            context_id=response.context_id,
            text=response.text,
            voice_text=response.voice_text,
            audio_data=response.audio_data,
            metadata=response.metadata or {}
        )

        if response.type == "chunk":
            # Face and Animation
            aiavatar_response.avatar_control_request = self.parse_avatar_control_request(response.text)

            # Voice
            if response.audio_data:
                b64_chunk = base64.b64encode(response.audio_data).decode("utf-8")
                aiavatar_response.audio_data = b64_chunk

        elif response.type == "tool_call":
            aiavatar_response.metadata["tool_call"] = response.tool_call.__dict__

        elif response.type == "final":
            if image_source_match := re.search(r"\[vision:(\w+)\]", response.text):
                aiavatar_response.type = "vision"
                aiavatar_response.metadata={"source": image_source_match.group(1)}

        elif response.type == "stop":
            await self.stop_response(response)

        await self.send_response(aiavatar_response)

    async def stop_response(self, session_id: str, context_id: str):
        # Send stop message to client
        await self.send_response(AIAvatarResponse(
            type="stop",
            session_id=session_id,
            context_id=context_id
        ))
