import asyncio
import audioop
import base64
from datetime import datetime
import io
import json
import logging
from typing import List, Dict, Callable, Awaitable, Optional
from uuid import uuid4
import wave
from fastapi import APIRouter, Request, WebSocket, Response
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
from ...database import PoolProvider
from ...sts.models import STSRequest, STSResponse
from ...sts.pipeline import STSPipeline
from ...sts.vad import SpeechDetector, SpeechDetectorDummy
from ...sts.stt import SpeechRecognizer, SpeechRecognizerDummy
from ...sts.stt.openai import OpenAISpeechRecognizer
from ...sts.llm import LLMService
from ...sts.llm.context_manager import ContextManager
from ...sts.tts import SpeechSynthesizer, SpeechSynthesizerDummy
from ...sts.session_state_manager import SessionStateManager
from ...sts.performance_recorder import PerformanceRecorder
from ...sts.voice_recorder import VoiceRecorder
from ..models import AIAvatarRequest, AIAvatarResponse
from ..base import Adapter

logger = logging.getLogger(__name__)


class TwilioSessionData:
    def __init__(self):
        self.call_sid: str = None
        self.stream_sid: str = None
        self.caller: str = ""
        self.direction: str = ""  # "inbound" or "outbound-api"
        self.websocket: WebSocket = None
        self.muted: bool = False

        self.last_mark: str = ""
        self.hang_mark: str = ""
        self.unmute_mark: str = ""
        self.is_first_utterance_timeout_invoked: bool = False
        self.max_turn: bool = False
        self.idling_start_at: datetime = datetime.now()
        self.data = {}


class TwilioSMSMessage:
    def __init__(self, message_sid: str, from_number: str, to_number: str, body: str):
        self.message_sid = message_sid
        self.from_number = from_number
        self.to_number = to_number
        self.body = body


class MakeCallRequest(BaseModel):
    to: Optional[str] = None
    user_id: Optional[str] = None
    text: Optional[str] = None
    call_reason: Optional[str] = None


class MakeCallResponse(BaseModel):
    call_sid: str


class SendSMSRequest(BaseModel):
    to: str
    body: str


class SendSMSResponse(BaseModel):
    message_sid: str


class HTMLResponse(Response):
    media_type = "text/html"


class AIAvatarTwilioServer(Adapter):
    def __init__(
        self,
        *,
        # Quick start
        volume_db_threshold: float = -50.0,
        silence_duration_threshold: float = 0.5,
        input_sample_rate: int = 16000,
        tts_sample_rate: int = 16000,
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-4.1",
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
        llm_model: str = "gpt-4.1",
        llm_system_prompt: str = None,
        llm_context_manager: ContextManager = None,
        tts_voicevox_url: str = "http://127.0.0.1:50021",
        tts_voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        wakeword_timeout: float = 60.0,
        merge_request_threshold: float = 0.0,
        merge_request_prefix: str = "$Previous user's request and your response have been canceled. Please respond again to the following request:\n\n",
        timestamp_interval_seconds: float = 0.0,
        timestamp_prefix: str = "$Current date and time: ",
        timestamp_timezone: str = "UTC",
        mute_on_barge_in: bool = False,
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

        # Twilio
        account_sid: str = None,
        auth_token: str = None,
        phone_number: str = None,
        webhook_base_url: str = None,
        channel: str = "phone",

        # SMS
        enable_sms: bool = False,
        sms_sts: STSPipeline = None,

        # Timeout operations
        first_utterance_timeout: float = 10.0,
        hangup_timeout: float = 40.0,
        on_first_utterance_timeout: Callable = None,
        on_hangup_timeout: Callable = None,

        # Max turn control
        max_turn_count: int = 0,
        max_turn_prompt_prefix: str = None,

        # Channel
        insert_channel_tag: bool = False,
        skip_tts_channels: List[str] = ["sms"],

        # Debug
        debug: bool = False,
    ):
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
            skip_tts_channels=skip_tts_channels,
            debug=debug
        )

        # Call base after self.sts is set
        super().__init__(self.sts)
        self.sessions: Dict[str, TwilioSessionData] = {}

        # Callbacks
        self._on_connect: Callable[[AIAvatarRequest, TwilioSessionData], Awaitable[None]] = None
        self._on_disconnect: Callable[[TwilioSessionData], Awaitable[None]] = None
        self._on_dtmf: Callable[[str, str], Awaitable[None]] = None
        self._on_sms_received: Callable[[TwilioSMSMessage], Awaitable[None]] = None
        self._resolve_phone_number: Callable[[str], Awaitable[Optional[str]]] = None

        # Audio
        self.tts_sample_rate = tts_sample_rate

        # Channel
        self.channel = channel

        # Twilio
        self.phone_number = phone_number
        self.twilio_client = Client(account_sid, auth_token) if account_sid and auth_token else None
        self.webhook_base_url: str = webhook_base_url
        self._outbound_call_data: Dict[str, dict] = {}

        # SMS pipeline
        if sms_sts:
            self.sms_sts = sms_sts
        elif enable_sms:
            self.sms_sts = STSPipeline(
                vad=SpeechDetectorDummy(),
                stt=SpeechRecognizerDummy(),
                llm=self.sts.llm,
                tts=SpeechSynthesizerDummy(),
                timestamp_interval_seconds=self.sts.timestamp_interval_seconds,
                timestamp_prefix=self.sts.timestamp_prefix,
                timestamp_timezone=self.sts.timestamp_timezone,
                db_pool_provider=self.sts.db_pool_provider,
                session_state_manager=self.sts.session_state_manager,
                performance_recorder=self.sts.performance_recorder,
                voice_recorder_enabled=False,
                debug=self.sts.debug,
            )
        else:
            self.sms_sts = None

        if self.sms_sts:
            self.sms_sts.handle_response = self._handle_sms_response
            self.sms_sts.stop_response = self._stop_sms_response

        # Timeout operations
        self.first_utterance_timeout = first_utterance_timeout
        self.hangup_timeout = hangup_timeout
        self._on_first_utterance_timeout: Callable[[str], Awaitable[None]] = on_first_utterance_timeout
        self._on_hangup_timeout: Callable[[str], Awaitable[None]] = on_hangup_timeout

        # Max turn control
        self.max_turn_count = max_turn_count
        self.max_turn_prompt_prefix = max_turn_prompt_prefix
        self._turn_counts: Dict[str, int] = {}

        if max_turn_count > 0:
            @self.sts.on_before_llm
            async def count_turns(request: STSRequest):
                context_id = request.context_id

                # Skip increment for merged requests (redo of cancelled turn)
                is_merged = (
                    self.sts.merge_request_threshold > 0
                    and request.text
                    and request.text.startswith(self.sts.merge_request_prefix)
                )

                if context_id not in self._turn_counts:
                    self._turn_counts[context_id] = 1
                elif not is_merged:
                    self._turn_counts[context_id] += 1

                turn_count = self._turn_counts[context_id]
                logger.info(f"Turn count: {turn_count}{' (merged)' if is_merged else ''} ({request.session_id})")

                if turn_count >= self.max_turn_count:
                    session_data = self.sessions.get(request.session_id)
                    if session_data:
                        session_data.max_turn = True
                    if self.max_turn_prompt_prefix:
                        request.text = f"{self.max_turn_prompt_prefix}\n\n{request.text}"

        # Mute immediately on barge-in
        if mute_on_barge_in:
            @self.sts.vad.on_recording_started
            async def mute_on_barge_in(session_id: str):
                # Reset marks so idle timer works correctly after barge-in
                if session_id in self.sessions:
                    self.sessions[session_id].last_mark = ""
                    self.sessions[session_id].hang_mark = ""
                await self.stop_response(session_id, "")

        # Debug
        self.debug = debug
        self.last_response = None

        @self.sts.on_accepted
        async def on_accepted(request: STSRequest):
            # Disable first utterance timeout once user has spoken
            if request.session_id in self.sessions:
                self.sessions[request.session_id].is_first_utterance_timeout_invoked = True

            barge_in_enabled = self.sts.vad.get_session_data(request.session_id, "barge_in_enabled")
            if barge_in_enabled is not False:
                # When barge-in is enabled, keep accumulated audio in VAD for the next request
                return
            if hasattr(self.sts.vad, "reset_session_audio_state"):
                # When barge-in is disabled, reset VAD audio so it doesn't carry over as the next request
                self.sts.vad.reset_session_audio_state(request.session_id, clear_preroll=True)
                if self.debug:
                    logger.info(
                        "VAD session audio state reset on accepted (barge_in_enabled=False): %s",
                        request.session_id
                    )

    def get_config(self) -> dict:
        return {
            "first_utterance_timeout": self.first_utterance_timeout,
            "hangup_timeout": self.hangup_timeout,
            "max_turn_count": self.max_turn_count,
            "debug": self.debug,
        }

    def on_connect(self, func: Callable[[AIAvatarRequest, TwilioSessionData], Awaitable[None]]):
        self._on_connect = func
        return func

    def on_disconnect(self, func: Callable[[TwilioSessionData], Awaitable[None]]):
        self._on_disconnect = func
        return func

    def on_dtmf(self, func: Callable[[str, str], Awaitable[None]]):
        """Register a callback for DTMF events. func(digit, session_id)"""
        self._on_dtmf = func
        return func

    def on_first_utterance_timeout(self, func: Callable[[str], Awaitable[None]]):
        self._on_first_utterance_timeout = func
        return func

    def on_hangup_timeout(self, func: Callable[[str], Awaitable[None]]):
        self._on_hangup_timeout = func
        return func

    def on_sms_received(self, func: Callable[[TwilioSMSMessage], Awaitable[None]]):
        self._on_sms_received = func
        return func

    def resolve_phone_number(self, func: Callable[[str], Awaitable[Optional[str]]]):
        """Register a callback to resolve user_id to phone number. func(user_id) -> phone_number"""
        self._resolve_phone_number = func
        return func

    def get_session_by_user_id(self, user_id: str) -> Optional[TwilioSessionData]:
        for session_id, session_data in reversed(self.sessions.items()):
            if self.sts.vad.get_session_data(session_id, "user_id") == user_id:
                return session_data
        return None

    # Request
    async def process_websocket(self, websocket: WebSocket, session_data: TwilioSessionData) -> TwilioSessionData:
        message_str = await websocket.receive_text()
        message = json.loads(message_str)
        event_type = message.get("event")

        if event_type == "start":
            stream_sid = message["start"]["streamSid"]
            call_sid = message["start"]["callSid"]

            session_data = self.sessions[call_sid]
            session_data.stream_sid = stream_sid
            session_data.websocket = websocket

            user_id = session_data.caller

            self.sts.vad.set_session_data(session_data.call_sid, "user_id", user_id, True)
            self.sts.vad.set_session_data(session_data.call_sid, "channel", self.channel)

            logger.info(f"WebSocket connected for stream_sid: {stream_sid} / call_sid: {call_sid} / user_id: {user_id}")

            # Callback for session start (base class)
            request = AIAvatarRequest(
                type="start",
                session_id=session_data.call_sid,
                user_id=user_id
            )
            for on_session_start in self._on_session_start_handlers:
                await on_session_start(request, session_data)

            if self._on_connect:
                asyncio.create_task(self._on_connect(request, session_data))

            # Auto-invoke pipeline for outbound calls with text
            outbound_text = session_data.data.get("text")
            if outbound_text:
                sts_request = STSRequest(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=request.context_id,
                    text=outbound_text,
                    channel=self.channel,
                )
                asyncio.create_task(self.invoke(sts_request))

        elif event_type == "media":
            if not session_data or session_data.muted:
                return session_data

            payload_b64 = message["media"]["payload"]
            mulaw_chunk = base64.b64decode(payload_b64)
            linear16_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
            # Convert from 8kHz to 16kHz
            resampled_chunk, _ = audioop.ratecv(linear16_chunk, 2, 1, 8000, 16000, None)
            is_recording = await self.sts.vad.process_samples(resampled_chunk, session_data.call_sid)
            if is_recording:
                # Reset idling if user is speaking
                session_data.idling_start_at = datetime.now()

        elif event_type == "mark":
            mark = message["mark"]["name"]
            if self.debug:
                logger.info(f"mark: {mark} ({session_data.call_sid})")

            if mark == session_data.last_mark:
                # Clear last mark if not overwritten by successive voice
                session_data.last_mark = ""

            if session_data.unmute_mark == mark:
                logger.info(f"Unmute: {mark} ({session_data.call_sid})")
                session_data.muted = False
                session_data.unmute_mark = ""

            if session_data.hang_mark == mark:
                logger.info(f"Hangup: {mark} ({session_data.call_sid})")
                await session_data.websocket.close()

        elif event_type == "dtmf":
            digit = message["dtmf"]["digit"]
            logger.info(f"dtmf: {digit} ({session_data.call_sid})")
            if self._on_dtmf:
                asyncio.create_task(self._on_dtmf(digit, session_data.call_sid))

        else:
            logger.info(f"event: {event_type}")

        return session_data

    # Response
    async def handle_response(self, response: STSResponse):
        aiavatar_response = AIAvatarResponse(
            type=response.type,
            session_id=response.session_id,
            user_id=response.user_id,
            context_id=response.context_id,
            text=response.text,
            voice_text=response.voice_text,
            audio_data=response.audio_data,
            metadata=response.metadata or {},
            structured_content=response.structured_content
        )

        # Callback for each response chunk (base class)
        for on_resp in self._on_response_handlers:
            await on_resp(aiavatar_response, response)

        session_data = self.sessions.get(response.session_id)
        if not session_data:
            logger.warning(f"Session not found for response: {response.session_id}")
            return

        # Reset idle timer during response processing to prevent timeout
        # from misfiring while LLM/TTS is working.
        # Set metadata["keep_idling"] = True to skip (e.g. for timeout prompts
        # where idle measurement should not be interrupted).
        if not (response.metadata and response.metadata.get("keep_idling")):
            session_data.idling_start_at = datetime.now()

        if response.type == "accepted":
            # Mute audio input while AI is speaking to block barge-in
            if response.metadata and response.metadata.get("block_barge_in"):
                session_data.muted = True

        elif response.type == "start":
            # Reset marks
            session_data.last_mark = ""
            session_data.hang_mark = ""

        elif response.type == "chunk":
            # Stop response if guardrail triggered
            if response.metadata and response.metadata.get("is_guardrail_triggered"):
                await self.stop_response(response.session_id, response.context_id)

            # Voice
            if response.audio_data:
                await self.send_voice(response.session_id, audio_data=response.audio_data)

        elif response.type == "final":
            # Schedule unmute after last audio chunk finishes playing
            if session_data.muted and session_data.last_mark:
                session_data.unmute_mark = session_data.last_mark

            if response.text and ("[operation:hangup]" in response.text or "<operation name=\"hangup\" />" in response.text):
                logger.info(f"Hangup after: {session_data.last_mark} ({session_data.call_sid})")
                session_data.hang_mark = session_data.last_mark
                if session_data.max_turn:
                    logger.info(f"Muted by hang_mark after max turn exceeded: {session_data.call_sid}")
                    session_data.muted = True

        elif response.type == "stop":
            await self.stop_response(response.session_id, response.context_id)

        if self.debug:
            self.last_response = aiavatar_response

    async def send_voice(self, session_id: str, *, text: str = None, audio_data: bytes = None):
        """Send voice audio to the Twilio WebSocket.

        Args:
            session_id: The stream SID to send audio to.
            text: Text for TTS synthesis. Used when audio_data is not provided.
            audio_data: 16kHz linear16 PCM audio bytes. If None, synthesized from text using pipeline TTS.
        """
        session_data = self.sessions.get(session_id)
        if not session_data or not session_data.websocket:
            logger.warning(f"WebSocket not found for session: {session_id}")
            return

        if audio_data is None:
            if not text:
                return
            audio_data = await self.sts.tts.synthesize(text)

        # Strip WAV header if present to get raw PCM samples
        if len(audio_data) >= 44 and audio_data[:4] == b"RIFF" and audio_data[8:12] == b"WAVE":
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())

        # Convert from tts_sample_rate linear16 to 8kHz mu-law
        if self.debug:
            conv_start = datetime.now()

        downsampled_data, _ = audioop.ratecv(audio_data, 2, 1, self.tts_sample_rate, 8000, None)
        mulaw_data = audioop.lin2ulaw(downsampled_data, 2)

        if self.debug:
            conv_ms = (datetime.now() - conv_start).total_seconds() * 1000
            logger.info(f"Audio conversion ({self.tts_sample_rate}Hz -> 8kHz mu-law): {conv_ms:.1f}ms ({session_id})")

        base64_encoded = base64.b64encode(mulaw_data).decode("utf-8")

        # Send media
        await session_data.websocket.send_json({
            "event": "media",
            "streamSid": session_data.stream_sid,
            "media": {
                "payload": base64_encoded
            }
        })

        # Send mark to track playback completion
        mark = f"{session_id}-{uuid4()}"
        await session_data.websocket.send_json({
            "event": "mark",
            "streamSid": session_data.stream_sid,
            "mark": {
                "name": mark
            }
        })

        session_data.last_mark = mark

    async def stop_response(self, session_id: str, context_id: str):
        session_data = self.sessions.get(session_id)
        if session_data and session_data.websocket:
            # Clear voice queue to stop speech
            await session_data.websocket.send_json({
                "event": "clear",
                "streamSid": session_data.stream_sid,
            })

    # Invoke pipeline
    async def invoke(self, request: STSRequest):
        try:
            async for response in self.sts.invoke(request):
                await self.sts.handle_response(response)
                if response.context_id:
                    self.sts.vad.set_session_data(request.session_id, "context_id", response.context_id)
        except Exception as ex:
            logger.exception(f"Error invoking pipeline: {ex}")

    async def make_call(self, to: str = None, from_: str = None, user_id: str = None, text: str = None, call_reason: str = None) -> str:
        # Resolve and validation
        if not to and user_id and self._resolve_phone_number:
            to = await self._resolve_phone_number(user_id)
        if not to:
            raise ValueError("Could not resolve phone number. Provide 'to' directly or register a resolve_phone_number callback.")
        from_number = from_ or self.phone_number
        if not from_number:
            raise ValueError("phone_number is required for making calls. Set it in the constructor or pass from_ parameter.")
        if not self.twilio_client:
            raise ValueError("twilio_client is not configured. Set account_sid and auth_token in the constructor.")
        if not self.webhook_base_url:
            raise ValueError("webhook_base_url is required for making calls. Set it in the constructor or call get_router first.")

        # Make call
        call = self.twilio_client.calls.create(
            to=to,
            from_=from_number,
            url=self.webhook_base_url + "/voice"
        )

        # Set outbound temporary data
        outbound_data = {}
        if text:
            outbound_data["text"] = text
        if call_reason:
            outbound_data["call_reason"] = call_reason
        if outbound_data:
            self._outbound_call_data[call.sid] = outbound_data

        return call.sid

    # Response (SMS)
    async def _handle_sms_response(self, response: STSResponse):
        pass

    async def _stop_sms_response(self, session_id: str, context_id: str):
        pass

    # Outbound SMS
    async def send_sms(self, to: str, body: str, from_: str = None) -> str:
        from_number = from_ or self.phone_number
        if not from_number:
            raise ValueError("phone_number is required for sending SMS. Set it in the constructor or pass from_ parameter.")
        if not self.twilio_client:
            raise ValueError("twilio_client is not configured. Set account_sid and auth_token in the constructor.")
        message = self.twilio_client.messages.create(body=body, from_=from_number, to=to)
        return message.sid

    # FastAPI Router
    def get_router(self):
        # Endpoints: /voice, /ws, /call/make, /sms, /sms/send
        websocket_url = self.webhook_base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

        router = APIRouter()

        # Phone
        @router.post("/voice")
        async def voice_webhook_endpoint(request: Request):
            form_data = await request.form()
            direction = form_data.get("Direction", "inbound")
            call_sid = form_data.get("CallSid", "")

            if direction == "outbound-api":
                caller = form_data.get("Called", "")
            else:
                caller = form_data.get("Caller", "")

            session_data = TwilioSessionData()
            session_data.call_sid = call_sid
            session_data.caller = caller
            session_data.direction = direction

            # Pick up outbound call data (text, etc.) stored by make_call
            outbound_data = self._outbound_call_data.pop(call_sid, None)
            if outbound_data:
                session_data.data.update(outbound_data)

            self.sessions[call_sid] = session_data

            logger.info(f"Call ({direction}) with: {caller} (CallSid: {call_sid})")

            response = VoiceResponse()
            connect = Connect()
            connect.stream(url=websocket_url)
            response.append(connect)
            return HTMLResponse(content=str(response), media_type="application/xml")

        @router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            session_data = None

            try:
                while True:
                    if session_data:
                        now = datetime.now()
                        if session_data.last_mark:
                            # Reset idling if AI is speaking
                            session_data.idling_start_at = now

                        # Timeout operations
                        idling_for = (now - session_data.idling_start_at).total_seconds()
                        if idling_for >= self.hangup_timeout and self._on_hangup_timeout is not None:
                            asyncio.create_task(self._on_hangup_timeout(session_data.call_sid))
                        elif (idling_for >= self.first_utterance_timeout and idling_for < self.hangup_timeout) and self._on_first_utterance_timeout is not None and not session_data.is_first_utterance_timeout_invoked:
                            session_data.is_first_utterance_timeout_invoked = True
                            asyncio.create_task(self._on_first_utterance_timeout(session_data.call_sid))

                    session_data = await self.process_websocket(websocket, session_data)

            except Exception as ex:
                error_message = str(ex)

                if "WebSocket is not connected" in error_message:
                    logger.info(f"WebSocket disconnected (1): session_id={session_data.call_sid}")
                elif "<CloseCode.NO_STATUS_RCVD: 1005>" in error_message:
                    logger.info(f"WebSocket disconnected (2): session_id={session_data.call_sid}")
                else:
                    raise

            finally:
                if session_data and session_data.call_sid:
                    if self._on_disconnect:
                        await self._on_disconnect(session_data)

                    # Clean up turn counts for this session's context
                    context_id = self.sts.vad.get_session_data(session_data.call_sid, "context_id")
                    if context_id and context_id in self._turn_counts:
                        del self._turn_counts[context_id]

                    await self.sts.finalize(session_data.call_sid)
                    if session_data.call_sid in self.sessions:
                        del self.sessions[session_data.call_sid]

        @router.post("/call/make", response_model=MakeCallResponse)
        async def make_outbound_call(request_body: MakeCallRequest):
            call_sid = await self.make_call(
                to=request_body.to,
                user_id=request_body.user_id,
                text=request_body.text,
                call_reason=request_body.call_reason
            )
            return MakeCallResponse(call_sid=call_sid)

        # SMS
        @router.post("/sms")
        async def incoming_sms(request: Request):
            form_data = await request.form()
            message = TwilioSMSMessage(
                message_sid=form_data.get("MessageSid", ""),
                from_number=form_data.get("From", ""),
                to_number=form_data.get("To", ""),
                body=form_data.get("Body", ""),
            )
            logger.info(f"Incoming SMS from: {message.from_number} (MessageSid: {message.message_sid})")
            if self._on_sms_received:
                asyncio.create_task(self._on_sms_received(message))
            return HTMLResponse(content="<Response></Response>", media_type="application/xml")

        @router.post("/sms/send", response_model=SendSMSResponse)
        async def send_sms_endpoint(request_body: SendSMSRequest):
            message_sid = await self.send_sms(to=request_body.to, body=request_body.body)
            return SendSMSResponse(message_sid=message_sid)

        return router
