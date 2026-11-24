import asyncio
import base64
import io
import logging
import math
import re
import wave
from time import monotonic
from typing import List, Dict, Callable, Awaitable, Optional, Iterator
from uuid import uuid4
from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketException, status
from ...database import PoolProvider
from ...sts.models import RequestContext, STSRequest, STSResponse
from ...sts.pipeline import (
    DEFAULT_INTERRUPTION_BEFORE_AUDIO_TEMPLATE, DEFAULT_INTERRUPTION_SPOKEN_TEMPLATE,
    InsertChannelTagConfig, STSPipeline,
)
from ...sts.vad import SpeechDetector
from ...sts.stt import SpeechRecognizer
from ...sts.stt.openai import OpenAISpeechRecognizer
from ...sts.llm import LLMService
from ...sts.llm.context_manager import ContextManager
from ...sts.tts import SpeechSynthesizer
from ...sts.session_state_manager import SessionStateManager
from ...sts.performance_recorder import PerformanceRecorder
from ...sts.voice_recorder import VoiceRecorder
from ..models import AvatarControlRequest, AIAvatarRequest, AIAvatarResponse
from ..base import Adapter
from ..interruption import InterruptionState

logger = logging.getLogger(__name__)


def iter_audio_responses(response: AIAvatarResponse, wav_data: bytes,
                         chunk_size: int, *, debug: bool = False) -> Iterator[AIAvatarResponse]:
    """Frame WAV audio for complete-audio or PCM WebSocket delivery.

    A positive chunk_size is a byte count and must be a multiple of the WAV's
    channels * sample_width. PCM consumers receive one descriptor before the
    chunks, whose shared audio_id identifies the original audio.
    """
    if chunk_size <= 0:
        if debug:
            logger.info(
                "WebSocket audio: format=wav, session=%s, transaction_id=%s, "
                "audio_id=None, audio_bytes=%s, chunk_size=0, planned_chunks=1, voice_text_tag=%s",
                response.session_id, response.transaction_id, len(wav_data),
                response.voice_text_tag,
            )
        yield response.model_copy(update={
            "audio_data": base64.b64encode(wav_data).decode("ascii")
        })
        return

    with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
        pcm_format = {
            "sample_rate": wav_file.getframerate(),
            "channels": wav_file.getnchannels(),
            "sample_width": wav_file.getsampwidth(),
        }
        pcm_data = wav_file.readframes(wav_file.getnframes())

    audio_id = str(uuid4())
    if debug:
        logger.info(
            "WebSocket audio: format=pcm, session=%s, transaction_id=%s, audio_id=%s, "
            "sample_rate=%s, channels=%s, sample_width=%s, audio_bytes=%s, "
            "chunk_size=%s, planned_chunks=%s, voice_text_tag=%s",
            response.session_id, response.transaction_id, audio_id,
            pcm_format["sample_rate"], pcm_format["channels"], pcm_format["sample_width"],
            len(pcm_data), chunk_size, (len(pcm_data) + chunk_size - 1) // chunk_size,
            response.voice_text_tag,
        )
    yield response.model_copy(update={
        "audio_data": None,
        "metadata": {
            **(response.metadata or {}),
            "pcm_format": pcm_format,
            "audio_id": audio_id,
            "audio_frame_count": len(pcm_data) // (
                pcm_format["channels"] * pcm_format["sample_width"]),
        },
    })
    for offset in range(0, len(pcm_data), chunk_size):
        yield AIAvatarResponse(
            type=response.type,
            session_id=response.session_id,
            user_id=response.user_id,
            context_id=response.context_id,
            transaction_id=response.transaction_id,
            audio_data=base64.b64encode(pcm_data[offset:offset + chunk_size]).decode("ascii"),
            metadata={"pcm_format": pcm_format, "audio_id": audio_id},
        )


class WebSocketSessionData:
    def __init__(self):
        self.id = None
        self.data = {}
        self.send_lock = asyncio.Lock()
        self.active_transaction_id: Optional[str] = None
        self.last_response_at = monotonic()
        # Barge-in mute: whether the user is still speaking after on_recording_started,
        # and the transaction whose remaining output is dropped.
        self.barge_in_active = False
        self.muted_transaction_id: Optional[str] = None
        self.interruption_state: Optional[InterruptionState] = None


class AIAvatarWebSocketServer(Adapter):
    def __init__(
        self,
        *,
        # Quick start
        volume_db_threshold: float = -50.0,
        silence_duration_threshold: float = 0.5,
        input_sample_rate: int = 16000,
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-5.6-terra",
        system_prompt: str = None,
        voicevox_speaker: int = 46,
        voicevox_url: str = "http://127.0.0.1:50021",

        # STS Pipeline and its components
        sts: STSPipeline = None,
        vad: SpeechDetector = None,
        stt: SpeechRecognizer = None,
        llm: LLMService = None,
        tts: SpeechSynthesizer = None,

        # STS Pipeline params for default components
        vad_volume_db_threshold: float = -50.0,
        vad_silence_duration_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        stt_sample_rate: int = 16000,
        llm_openai_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "gpt-5.6-terra",
        llm_temperature: float = None,
        llm_reasoning_effort: str = None,
        llm_system_prompt: str = None,
        llm_context_manager: ContextManager = None,
        tts_voicevox_url: str = "http://127.0.0.1:50021",
        tts_voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        wakeword_timeout: float = 60.0,
        merge_request_threshold: float = 0.0,
        merge_request_prefix: str = "$Previous user's request and your response have been canceled. Please respond again to the following request:\n\n",
        interruption_spoken_template: str = DEFAULT_INTERRUPTION_SPOKEN_TEMPLATE,
        interruption_before_audio_template: str = DEFAULT_INTERRUPTION_BEFORE_AUDIO_TEMPLATE,
        timestamp_interval_seconds: float = 0.0,
        timestamp_prefix: str = "$Current date and time: ",
        timestamp_timezone: str = "UTC",
        mute_on_barge_in: bool = False,
        barge_in_protected_tags: Optional[List[str]] = None,
        include_interruption_context: bool = False,
        db_pool_provider: PoolProvider = None,
        db_connection_str: str = "aiavatar.db",
        session_state_manager: SessionStateManager = None,
        performance_recorder: PerformanceRecorder = None,
        voice_recorder: VoiceRecorder = None,
        voice_recorder_enabled: bool = True,
        voice_recorder_dir: str = "recorded_voices",
        invoke_queue_idle_timeout: float = 10.0,
        invoke_timeout: float = 60.0,
        use_invoke_queue: bool = False,

        # WebSocket processing
        response_audio_chunk_size: int = 0, # 0 = Send whole audio data at once
        send_voiced: bool = False,
        # API server auth
        api_key: str = None,
        # Channel
        channel: str = "websocket",
        insert_channel_tag: InsertChannelTagConfig = False,
        # Debug
        debug: bool = False,
    ):
        if include_interruption_context:
            detector = sts.vad if sts is not None else vad
            if not callable(getattr(detector, "on_speech_started", None)):
                raise ValueError("include_interruption_context requires a VAD with on_speech_started")

        # Speech-to-Speech pipeline
        self.sts = sts or STSPipeline(
            # VAD
            vad=vad,
            vad_volume_db_threshold=vad_volume_db_threshold or volume_db_threshold,
            vad_silence_duration_threshold=vad_silence_duration_threshold or silence_duration_threshold,
            vad_sample_rate=vad_sample_rate or input_sample_rate,
            # STT (Overwrite default)
            stt=stt or OpenAISpeechRecognizer(
                openai_api_key=openai_api_key,
                sample_rate=stt_sample_rate or input_sample_rate
            ),
            # LLM
            llm=llm,
            llm_openai_api_key=llm_openai_api_key or openai_api_key,
            llm_base_url=llm_base_url or openai_base_url,
            llm_model=llm_model or openai_model,
            llm_temperature=llm_temperature,
            llm_reasoning_effort=llm_reasoning_effort,
            llm_system_prompt=llm_system_prompt or system_prompt,
            llm_context_manager=llm_context_manager,
            # TTS
            tts=tts,
            tts_voicevox_url=tts_voicevox_url or voicevox_url,
            tts_voicevox_speaker=tts_voicevox_speaker or voicevox_speaker,
            # Pipeline
            wakewords=wakewords,
            wakeword_timeout=wakeword_timeout,
            merge_request_threshold=merge_request_threshold,
            merge_request_prefix=merge_request_prefix,
            interruption_spoken_template=interruption_spoken_template,
            interruption_before_audio_template=interruption_before_audio_template,
            timestamp_interval_seconds=timestamp_interval_seconds,
            timestamp_prefix=timestamp_prefix,
            timestamp_timezone=timestamp_timezone,
            db_pool_provider=db_pool_provider,
            db_connection_str=db_connection_str,
            session_state_manager=session_state_manager,
            performance_recorder=performance_recorder,
            voice_recorder=voice_recorder,
            voice_recorder_enabled=voice_recorder_enabled,
            voice_recorder_dir=voice_recorder_dir,
            invoke_queue_idle_timeout=invoke_queue_idle_timeout,
            invoke_timeout=invoke_timeout,
            use_invoke_queue=use_invoke_queue,
            insert_channel_tag=insert_channel_tag,
            debug=debug
        )

        # Call base after self.sts is set
        super().__init__(self.sts)
        self.websockets: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, WebSocketSessionData] = {}

        # Callbacks
        self._on_connect: Callable[[AIAvatarRequest, WebSocketSessionData], Awaitable[None]] = None
        self._on_disconnect: Callable[[WebSocketSessionData], Awaitable[None]] = None

        # WebSocket processing
        self.response_audio_chunk_size = response_audio_chunk_size
        self.barge_in_protected_tags = list(barge_in_protected_tags or [])

        # API Key
        self.api_key = api_key

        # Channel
        self.channel = channel

        # Mute immediately on barge-in. The rest of the response in progress, and of any
        # response accepted while the user keeps speaking, is dropped. Generation in the
        # pipeline continues, so the conversation context keeps the whole exchange.
        if mute_on_barge_in:
            @self.sts.vad.on_recording_started
            async def mute_on_barge_in(session_id: str):
                session_data = self.sessions.get(session_id)
                if session_data is None:
                    return
                # Mark before awaiting so that output arriving meanwhile is dropped.
                session_data.barge_in_active = True
                session_data.muted_transaction_id = session_data.active_transaction_id
                if self.debug:
                    logger.info(
                        "Mute on barge-in: session=%s, transaction=%s",
                        session_id, session_data.muted_transaction_id
                    )
                await self.stop_response(session_id, "")

        # Debug
        self.debug = debug
        self.last_response = None

        @self.sts.on_accepted(channels=self.channel)
        async def on_accepted(request: STSRequest):
            if not self.can_handle(request.session_id):
                return
            barge_in_enabled = self.sts.vad.get_session_data(request.session_id, "barge_in_enabled")
            if barge_in_enabled is not False:
                return
            if hasattr(self.sts.vad, "reset_session_audio_state"):
                self.sts.vad.reset_session_audio_state(request.session_id, clear_preroll=True)
                if self.debug:
                    logger.info(
                        "VAD session audio state reset on accepted (barge_in_enabled=False): %s",
                        request.session_id
                    )

        self.send_voiced = send_voiced

        @self.sts.vad.on_voiced
        async def on_voiced(session_id: str):
            if self.send_voiced and session_id in self.websockets:
                await self.send_response(AIAvatarResponse(
                    type="voiced",
                    session_id=session_id
                ))

        if include_interruption_context:
            self._register_interruption_context()

    def _register_interruption_context(self):
        vad = self.sts.vad

        @self.on_session_start
        async def initialize_interruption_state(_request, session):
            session.interruption_state = InterruptionState(debug=self.debug)

        @self.on_response
        async def remember_response(_response, response):
            session = self.sessions.get(response.session_id)
            if session is not None and session.interruption_state is not None:
                session.interruption_state.observe(response)

        @self.sts.on_before_llm(channels=self.channel)
        async def include_discarded_audio(request: STSRequest):
            session = self.sessions.get(request.session_id)
            if not request.text or session is None or session.interruption_state is None:
                return
            discarded_text = "\n".join(session.interruption_state.take_discarded_audio(request.context_id))
            if discarded_text:
                request.text = (
                    "$The following assistant speech was canceled before playback started.\n"
                    f"{discarded_text}\n\n{request.text}"
                )

        @vad.on_speech_started
        def capture_interruption(session_id: str, metadata: dict) -> None:
            session = self.sessions.get(session_id)
            if session is None or session.interruption_state is None:
                return
            interruption = session.interruption_state.snapshot(session.active_transaction_id)
            metadata["request_context"] = RequestContext(interruption=interruption, track_interruption=True)

    def set_context_resolver(
        self,
        resolver: Callable[[str], Awaitable[Optional[str]]],
        *,
        after_idle_seconds: float,
    ):
        """Register a user-to-context resolver for idle turns during setup."""
        if not math.isfinite(after_idle_seconds) or after_idle_seconds <= 0:
            raise ValueError("after_idle_seconds must be a positive finite number")

        @self.sts.on_accepted(channels=self.channel)
        async def refresh_context(request: STSRequest):
            session = self.sessions.get(request.session_id)
            if session is None or monotonic() - session.last_response_at < after_idle_seconds:
                return
            user_id = self.sts.vad.get_session_data(request.session_id, "user_id")
            if not user_id:
                return
            context_id = await resolver(user_id)
            if self.sessions.get(request.session_id) is not session:
                return
            request.user_id = user_id
            request.context_id = context_id
            self.sts.vad.set_session_data(request.session_id, "context_id", request.context_id)

    def get_config(self) -> dict:
        return {
            "response_audio_chunk_size": self.response_audio_chunk_size,
            "debug": self.debug,
        }

    def on_connect(self, func: Callable[[AIAvatarRequest, WebSocketSessionData], Awaitable[None]]):
        self._on_connect = func
        return func

    def on_disconnect(self, func: Callable[[WebSocketSessionData], Awaitable[None]]):
        self._on_disconnect = func
        return func

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

        # Callback for request
        for on_req in self._on_request_handlers:
            await on_req(request)

        if request.type == "start":
            self.websockets[request.session_id] = websocket
            session_data.id = request.session_id

            logger.info(f"WebSocket connected for session: {request.session_id}")

            for on_session_start in self._on_session_start_handlers:
                await on_session_start(request, session_data)

            # Store session data to initialize
            if request.user_id:
                self.sts.vad.set_session_data(request.session_id, "user_id", request.user_id, True)
            if request.context_id:
                self.sts.vad.set_session_data(request.session_id, "context_id", request.context_id, True)
            self.sts.vad.set_session_data(request.session_id, "channel", self.channel, True)
            if request.metadata:
                self._apply_session_config(request.session_id, request.metadata, create_session=True)
            session_data.data["metadata"] = request.metadata
            self.sessions[session_data.id] = session_data
            if session_data.interruption_state is not None:
                self.sts.start_interruption_tracking(session_data.id)

            await self.handle_response(STSResponse(
                type="connected",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id
            ))

            if self._on_connect:
                asyncio.create_task(self._on_connect(request, session_data))

        elif request.type == "invoke":
            if request.context_id:
                context_id = request.context_id
            else:
                context_id = self.sts.vad.get_session_data(request.session_id, "context_id")

            if request.metadata and "barge_in_enabled" in request.metadata:
                self.sts.vad.set_session_data(
                    request.session_id,
                    "barge_in_enabled",
                    bool(request.metadata.get("barge_in_enabled")),
                    True
                )

            request_context = None
            if (request.text and session_data.interruption_state is not None
                    and self.sessions.get(request.session_id) is session_data):
                request_context = RequestContext(track_interruption=True)

            async for r in self.sts.invoke(STSRequest(
                type=request.type,
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=context_id,
                text=request.text,
                audio_data=base64.b64decode(request.audio_data) if request.audio_data else None,
                files=request.files,
                system_prompt_params=request.system_prompt_params,
                allow_merge=request.allow_merge,
                wait_in_queue=request.wait_in_queue,
                channel=self.channel,
                metadata=request.metadata,
                request_context=request_context,
            )):
                if r.type == "start":
                    self.sts.vad.set_session_data(request.session_id, "context_id", r.context_id)
                await self.sts.handle_response(r)

        elif request.type == "data":
            audio_data = base64.b64decode(request.audio_data)
            is_recording = await self.sts.vad.process_samples(audio_data, request.session_id)
            data_session = self.sessions.get(request.session_id)
            if not is_recording and data_session is not None:
                # The user's speech has ended; requests accepted from now on are new.
                data_session.barge_in_active = False

        elif request.type == "playback":
            if (
                session_data.interruption_state is not None
                and self.sessions.get(request.session_id) is session_data
                and self.websockets.get(request.session_id) is websocket
            ):
                session_data.interruption_state.handle_playback_event(**(request.metadata or {}))

        elif request.type == "config":
            metadata = request.metadata or {}
            if metadata:
                self._apply_session_config(request.session_id, metadata, create_session=True)
            if hasattr(self.sts.vad, "volume_db_threshold"):
                volume_db_threshold = metadata.get("volume_db_threshold")
                if volume_db_threshold:
                    self.sts.vad.set_volume_db_threshold(request.session_id, volume_db_threshold)

        elif request.type == "stop":
            logger.info(f"WebSocket disconnect for session: {request.session_id}")
            await websocket.close()

    def _apply_session_config(self, session_id: str, metadata: dict, create_session: bool = False):
        if "barge_in_enabled" in metadata:
            value = bool(metadata.get("barge_in_enabled"))
            self.sts.vad.set_session_data(session_id, "barge_in_enabled", value, create_session=create_session)
            if self.debug:
                logger.info("Set session config: session=%s, barge_in_enabled=%s", session_id, value)

    # Response
    async def send_response(self, aiavatar_response: AIAvatarResponse):
        """Send a response, framing audio bytes and preserving encoded audio strings."""
        session_id = aiavatar_response.session_id
        session_data = self.sessions.get(session_id)
        if not session_data:
            return
        audio_data = aiavatar_response.audio_data
        raw_audio = audio_data if aiavatar_response.type == "chunk" and isinstance(audio_data, bytes) else None
        responses = iter((aiavatar_response,))
        protected_pcm = False
        if raw_audio:
            chunk_size = self.response_audio_chunk_size
            protected_pcm = chunk_size > 0 and (aiavatar_response.metadata or {}).get("protect_from_barge_in") is True
            responses = iter_audio_responses(aiavatar_response, raw_audio, chunk_size, debug=self.debug)

        # Keep a protected clip's descriptor and audio together under one send lock.
        batches = (responses,) if protected_pcm else ((response,) for response in responses)
        for batch in batches:
            if raw_audio and not self._is_deliverable(session_data, aiavatar_response):
                return
            async with session_data.send_lock:
                websocket = self.websockets.get(session_id)
                if websocket is None or self.sessions.get(session_id) is not session_data:
                    return
                if protected_pcm and not self._is_deliverable(session_data, aiavatar_response):
                    return

                for response in batch:
                    state = session_data.interruption_state
                    await websocket.send_text(response.model_dump_json())
                    if (self.sessions.get(session_id) is not session_data
                            or self.websockets.get(session_id) is not websocket):
                        return
                    if state is not None:
                        metadata = response.metadata or {}
                        # PCM payloads use the original audio clip's tag.
                        if ((response.type == "chunk" and response.audio_data
                             and aiavatar_response.voice_text_tag != "ack")
                                or (response.type == "final" and not metadata.get("interrupted"))):
                            state.mark_answered(response.transaction_id)

    def can_handle(self, session_id: str) -> bool:
        return self.sessions.get(session_id) is not None

    async def handle_response(self, response: STSResponse):
        session_data = self.sessions.get(response.session_id)
        if not session_data:
            logger.warning(f"Session not found for response (WebSocket): {response.session_id}")
            return

        # On accepted: activate the new transaction, then close the old one for the client
        # unless its own final has been sent. The state changes before the write, so output
        # of the old transaction handled meanwhile is dropped instead of following its final.
        if response.type == "accepted" and response.transaction_id and session_data:
            previous_transaction_id = session_data.active_transaction_id
            session_data.active_transaction_id = response.transaction_id
            if session_data.barge_in_active:
                # Accepted while the user keeps speaking: it answers a request from before the barge-in.
                session_data.muted_transaction_id = response.transaction_id
            if previous_transaction_id and previous_transaction_id != response.transaction_id:
                await self._send_interrupted_final(session_data, response, previous_transaction_id)

        # Skip responses of a transaction replaced by a newer one, and the content of one muted
        # on barge-in, whose unheard text and tags must not reach the client.
        if not self._is_deliverable(session_data, response):
            if self.debug:
                logger.info(
                    "Skip undeliverable response: type=%s, transaction=%s, active=%s",
                    response.type, response.transaction_id, session_data.active_transaction_id
                )
            if response.type == "final" and session_data.active_transaction_id == response.transaction_id:
                # Still open for the client (muted): close it with an empty interrupted final.
                session_data.active_transaction_id = None
                await self._send_interrupted_final(session_data, response, response.transaction_id)
            return

        if response.type == "start":
            session_data.last_response_at = monotonic()

        aiavatar_response = AIAvatarResponse(
            type=response.type,
            session_id=response.session_id,
            user_id=response.user_id,
            context_id=response.context_id,
            transaction_id=response.transaction_id,
            text=response.text,
            voice_text=response.voice_text,
            voice_text_tag=response.voice_text_tag,
            control_tags=(self.parse_control_tags(response.text) or None) if response.type == "chunk" else None,
            audio_data=response.audio_data,
            metadata=response.metadata or {},
            structured_content=response.structured_content
        )

        # Callback for each response chunk
        for on_resp in self._on_response_handlers:
            await on_resp(aiavatar_response, response)

        if response.type == "chunk":
            # Stop response if guardrail triggered
            if response.metadata.get("is_guardrail_triggered"):
                await self.stop_response(response.session_id, response.context_id, metadata={"force_stop": True})

            # Language
            aiavatar_response.language = response.language

            # Face and Animation
            aiavatar_response.avatar_control_request = self.parse_avatar_control_request(response.text)

            if aiavatar_response.audio_data and aiavatar_response.voice_text_tag in self.barge_in_protected_tags:
                aiavatar_response.metadata = {
                    **(aiavatar_response.metadata or {}),
                    "protect_from_barge_in": True,
                }

        elif response.type == "tool_call":
            aiavatar_response.metadata["tool_call"] = response.tool_call.to_dict()

        elif response.type == "final":
            vision_source = self.parse_vision_source(response.text)
            if vision_source:
                aiavatar_response.type = "vision"
                aiavatar_response.metadata = {"source": vision_source}

        elif response.type == "stop":
            await self.stop_response(response.session_id, response.context_id)

        # The response may have been muted or replaced while the handlers above were awaited.
        if not self._is_deliverable(session_data, response):
            if self.debug:
                logger.info(
                    "Skip undeliverable response: type=%s, transaction=%s, active=%s",
                    response.type, response.transaction_id, session_data.active_transaction_id
                )
            if response.type == "final" and session_data.active_transaction_id == response.transaction_id:
                # Still open for the client (muted): close it with an empty interrupted final.
                session_data.active_transaction_id = None
                await self._send_interrupted_final(session_data, response, response.transaction_id)
            return

        if response.type == "final" and response.transaction_id:
            # Its own final closes the transaction, so the next accepted does not close it again.
            session_data.active_transaction_id = None

        await self.send_response(aiavatar_response)

    def _is_deliverable(self, session_data: WebSocketSessionData, response: STSResponse | AIAvatarResponse) -> bool:
        """Return whether the response can be sent to this connection now.

        A response with a transaction ID is sent only while the client has that transaction
        open. A barge-in mute withholds only the content (chunk, tool_call, and final);
        accepted and start are still sent, so the client sees the transaction open before
        its interrupted final.
        """
        if self.sessions.get(response.session_id) is not session_data:
            return False  # The connection has been replaced or closed
        if not response.transaction_id:
            return True
        if response.transaction_id != session_data.active_transaction_id:
            return False  # Replaced by a newer transaction, or already closed
        if response.type in ("chunk", "tool_call", "final"):
            return response.transaction_id != session_data.muted_transaction_id
        return True

    async def _send_interrupted_final(self, session_data: WebSocketSessionData, response: STSResponse, transaction_id: str):
        """Close the transaction for the client with an empty final marked interrupted."""
        if self.sessions.get(response.session_id) is not session_data:
            return  # Not to a connection that has replaced this one
        await self.send_response(AIAvatarResponse(
            type="final",
            session_id=response.session_id,
            user_id=response.user_id,
            context_id=response.context_id,
            transaction_id=transaction_id,
            text="",
            voice_text="",
            metadata={"interrupted": True}
        ))

    async def stop_response(self, session_id: str, context_id: str, *, metadata: Optional[dict] = None):
        # Send stop message to client
        await self.send_response(AIAvatarResponse(
            type="stop",
            session_id=session_id,
            context_id=context_id,
            metadata=metadata,
        ))

    def _authenticate_websocket(self, websocket: WebSocket) -> Optional[str]:
        """Authenticate WebSocket connection using Authorization header or Sec-WebSocket-Protocol.

        Returns the accepted subprotocol name if authenticated via Sec-WebSocket-Protocol, or None.
        """
        # Check Authorization header (for native clients)
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer ") and auth_header[7:] == self.api_key:
            return None

        # Check Sec-WebSocket-Protocol (for browser clients)
        # Format: "Authorization.<base64_encoded_api_key>"
        for protocol in websocket.headers.get("sec-websocket-protocol", "").split(","):
            protocol = protocol.strip()
            if protocol.startswith("Authorization."):
                try:
                    b64_key = protocol[14:]
                    b64_key += "=" * (-len(b64_key) % 4)  # Restore padding stripped by browser
                    decoded_key = base64.b64decode(b64_key).decode("utf-8")
                    if decoded_key == self.api_key:
                        return protocol
                except Exception:
                    pass

        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid or missing API Key",
        )

    def get_session_by_user_id(self, user_id: str) -> Optional[WebSocketSessionData]:
        for session_id, session_data in reversed(self.sessions.items()):
            if self.sts.vad.get_session_data(session_id, "user_id") == user_id:
                return session_data
        return None

    def get_websocket_router(self, path: str = "/ws"):
        router = APIRouter()

        @router.get(path + "/sessions")
        async def get_session_by_user_id_endpoint(user_id: str, authorization: str = Header(default=None)):
            if self.api_key:
                if not authorization or not authorization.lower().startswith("bearer ") or authorization[7:] != self.api_key:
                    raise HTTPException(status_code=401, detail="Invalid or missing API Key")
            session_data = self.get_session_by_user_id(user_id)
            if session_data is None:
                return {"session_id": None}
            return {"session_id": session_data.id, "data": session_data.data}

        @router.websocket(path)
        async def websocket_endpoint(websocket: WebSocket):
            subprotocol = None
            if self.api_key:
                subprotocol = self._authenticate_websocket(websocket)
            await websocket.accept(subprotocol=subprotocol)
            session_data = WebSocketSessionData()

            try:
                while True:
                    await self.process_websocket(websocket, session_data)

            except Exception as ex:
                error_message = str(ex)

                if "WebSocket is not connected" in error_message:
                    logger.info(f"WebSocket disconnected (1): session_id={session_data.id}")
                elif "<CloseCode.NO_STATUS_RCVD: 1005>" in error_message:
                    logger.info(f"WebSocket disconnected (2): session_id={session_data.id}")
                else:
                    raise

            finally:
                if session_id := session_data.id:
                    try:
                        if self._on_disconnect:
                            await self._on_disconnect(session_data)
                    finally:
                        if self.websockets.get(session_id) is websocket:
                            await self.sts.finalize(session_id)
                            if self.websockets.get(session_id) is websocket:
                                self.websockets.pop(session_id, None)
                                self.sessions.pop(session_id, None)

        return router
