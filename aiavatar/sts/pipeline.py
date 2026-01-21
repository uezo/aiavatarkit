import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import logging
import re
from time import time
import traceback
from typing import AsyncGenerator, Tuple, List, Optional
from uuid import uuid4
from ..database import PoolProvider
from .models import STSRequest, STSResponse
from .vad import SpeechDetector
from .vad.silero import SileroSpeechDetector
from .stt import SpeechRecognizer
from .stt.google import GoogleSpeechRecognizer
from .llm import LLMService, LLMResponse
from .llm.chatgpt import ChatGPTService
from .llm.context_manager import ContextManager
from .tts import SpeechSynthesizer
from .tts.voicevox import VoicevoxSpeechSynthesizer
from .performance_recorder import PerformanceRecord, PerformanceRecorder
from .performance_recorder.sqlite import SQLitePerformanceRecorder
from .voice_recorder import VoiceRecorder, RequestVoice, ResponseVoices
from .voice_recorder.file import FileVoiceRecorder
from .session_state_manager import SessionStateManager, SQLiteSessionStateManager

logger = logging.getLogger(__name__)


class STSPipeline:
    def __init__(
        self,
        *,
        vad: SpeechDetector = None,
        vad_volume_db_threshold: float = -90.0,
        vad_silence_duration_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        stt: SpeechRecognizer = None,
        stt_google_api_key: str = None,
        stt_sample_rate: int = 16000,
        llm: LLMService = None,
        llm_openai_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "gpt-4o-mini",
        llm_system_prompt: str = None,
        llm_context_manager: ContextManager = None,
        tts: SpeechSynthesizer = None,
        tts_voicevox_url: str = "http://127.0.0.1:50021",
        tts_voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        wakeword_timeout: float = 60.0,
        merge_request_threshold: float = 0.0,
        merge_request_prefix: str = "$Previous user's request and your response have been canceled. Please respond again to the following request:\n\n",
        # Japanese version
        # merge_request_prefix: str = "$直前のユーザーの要求とあなたの応答はキャンセルされました。以下の要求に対して、あらためて応答しなおしてください:\n\n"
        timestamp_interval_seconds: float = 0.0,
        timestamp_prefix: str = "$Current date and time: ",
        timestamp_timezone: str = "UTC",
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
        debug: bool = False
    ):
        self.debug = debug
        self.use_invoke_queue = use_invoke_queue

        # Database connection pool
        if db_pool_provider:
            if db_pool_provider.db_type == "postgresql":
                self.db_pool_provider = db_pool_provider
            else:
                raise ValueError(f"Unsupported db_type: {db_pool_provider.db_type}")
        elif db_connection_str.startswith("postgresql://"):
            from ..database.postgres import PostgreSQLPoolProvider
            self.db_pool_provider = PostgreSQLPoolProvider(connection_str=db_connection_str)
        else:
            self.db_pool_provider = None

        # Session state management
        if session_state_manager:
            self.session_state_manager = session_state_manager
        elif self.db_pool_provider:
            from .session_state_manager.postgres import PostgreSQLSessionStateManager
            self.session_state_manager = PostgreSQLSessionStateManager(get_pool=self.db_pool_provider.get_pool)
        else:
            self.session_state_manager = SQLiteSessionStateManager(db_path=db_connection_str)

        # VAD
        self.vad = vad or SileroSpeechDetector(
            volume_db_threshold=vad_volume_db_threshold,
            silence_duration_threshold=vad_silence_duration_threshold,
            sample_rate=vad_sample_rate,
            debug=debug
        )

        @self.vad.on_speech_detected
        async def on_speech_detected(data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
            async for response in self.invoke(STSRequest(
                session_id=session_id,
                user_id=self.vad.get_session_data(session_id, "user_id"),
                context_id=self.vad.get_session_data(session_id, "context_id"),
                text=text,
                audio_data=data,
                audio_duration=recorded_duration,
                system_prompt_params=self.vad.get_session_data(session_id, "system_prompt_params")
            )):
                if response.type == "start":
                    self.vad.set_session_data(session_id, "context_id", response.context_id)
                await self.handle_response(response)

        # Speech-to-Text
        self.stt = stt or GoogleSpeechRecognizer(
            google_api_key=stt_google_api_key,
            sample_rate=stt_sample_rate,
            debug=debug
        )

        # LLM
        if llm:
            self.llm = llm
        else:
            _context_managaer = None
            if llm_context_manager:
                _context_managaer = llm_context_manager
            else:
                if self.db_pool_provider:
                    from .llm.context_manager.postgres import PostgreSQLContextManager
                    _context_managaer = PostgreSQLContextManager(get_pool=self.db_pool_provider.get_pool)

            self.llm = ChatGPTService(
                openai_api_key=llm_openai_api_key,
                base_url=llm_base_url,
                model=llm_model,
                system_prompt=llm_system_prompt,
                context_manager=_context_managaer,
                db_connection_str=db_connection_str,
                debug=debug
            )

        # Text-to-Speech
        self.tts = tts or VoicevoxSpeechSynthesizer(
            base_url=tts_voicevox_url,
            speaker=tts_voicevox_speaker,
            debug=debug
        )

        # Wakeword
        self.wakewords = wakewords
        self.wakeword_timeout = wakeword_timeout

        # Merge consecutive requests
        self.merge_request_threshold = merge_request_threshold
        self.merge_request_prefix = merge_request_prefix

        # Timestamp
        self.timestamp_interval_seconds = timestamp_interval_seconds
        self.timestamp_timezone = timestamp_timezone
        self.timestamp_prefix = timestamp_prefix

        # Response handler
        self.handle_response = self.handle_response_default
        self.stop_response = self.stop_response_default
        self._process_llm_chunk = self.process_llm_chunk_default

        # Performance recorder
        if performance_recorder:
            self.performance_recorder = performance_recorder
        else:
            if self.db_pool_provider:
                from .performance_recorder.postgres import PostgreSQLPerformanceRecorder
                self.performance_recorder = PostgreSQLPerformanceRecorder(connection_str=self.db_pool_provider.connection_str)
            else:
                self.performance_recorder = SQLitePerformanceRecorder(db_path=db_connection_str)

        # Voice recorder
        self.voice_recorder = voice_recorder or FileVoiceRecorder(
            record_dir=voice_recorder_dir,
            sample_rate=stt_sample_rate
        )
        self.voice_recorder_enabled = voice_recorder_enabled
        self.voice_recorder_response_audio_format = "wav"

        # User custom logic
        self._on_before_llm = self.on_before_llm_default
        self._on_before_tts = self.on_before_tts_default
        self._on_finish = self.on_finish_default

        # Queue management for invoke_queued
        self._request_queues: dict[str, asyncio.Queue] = {}
        self._invoke_workers: dict[str, asyncio.Task] = {}
        self._response_queues: dict[str, dict[str, asyncio.Queue]] = {}
        self.invoke_queue_idle_timeout = invoke_queue_idle_timeout
        self.invoke_timeout = invoke_timeout

    def on_before_llm(self, func):
        self._on_before_llm = func
        return func

    def on_before_tts(self, func):
        self._on_before_tts = func
        return func
    
    def on_finish(self, func):
        self._on_finish = func
        return func

    async def on_before_llm_default(self, request: STSRequest):
        pass

    async def on_before_tts_default(self, request: STSRequest):
        pass

    async def on_finish_default(self, request: STSRequest, response: STSResponse):
        pass

    async def process_audio_samples(self, samples: bytes, context_id: str):
        await self.vad.process_samples(samples, context_id)

    def process_llm_chunk(self, func) -> dict:
        self._process_llm_chunk = func
        return func

    async def process_llm_chunk_default(self, response: STSResponse):
        return {}

    async def handle_response_default(self, response: STSResponse):
        logger.info(f"Handle response: {response}")

    async def stop_response_default(self, session_id: str, context_id: str):
        logger.info(f"Stop response: {session_id} / {context_id}")

    def is_awake(self, request: STSRequest, last_request_at: datetime) -> bool:
        now = datetime.now(timezone.utc)

        if not self.wakewords:
            # Always return True if no wakewords are registered
            return True

        if self.wakeword_timeout > (now - last_request_at).total_seconds():
            # Return True if not timeout
            return True

        for ww in self.wakewords:
            if ww in request.text:
                logger.info(f"Wake by '{ww}': {request.text}")
                return True

        return False

    async def is_transaction_active(self, session_id: str, transaction_id: str) -> Tuple[bool, Optional[str]]:
        state = await self.session_state_manager.get_session_state(session_id)
        return state.active_transaction_id == transaction_id, state.active_transaction_id

    async def invoke(self, request: STSRequest) -> AsyncGenerator[STSResponse, None]:
        if self.use_invoke_queue:
            async for response in self._invoke_queued(request):
                yield response
        else:
            async for response in self._invoke_direct(request):
                yield response

    async def _invoke_direct(self, request: STSRequest) -> AsyncGenerator[STSResponse, None]:
        try:
            if not request.session_id:
                raise ValueError("session_id is required but not provided")

            # Notify client that request is accepted (fire and forget to avoid blocking pipeline latency)
            asyncio.create_task(self.handle_response(STSResponse(
                type="accepted",
                session_id=request.session_id,
                metadata={"block_barge_in": request.block_barge_in}
            )))

            start_time = time()
            transaction_id = str(uuid4())

            performance = PerformanceRecord(
                transaction_id=transaction_id,
                user_id=request.user_id,
                stt_name=self.stt.__class__.__name__,
                llm_name=self.llm.__class__.__name__,
                tts_name=self.tts.__class__.__name__
            )

            # Record request voice
            if self.voice_recorder_enabled and request.audio_data:
                await self.voice_recorder.record(RequestVoice(transaction_id, request.audio_data))

            if request.text:
                # Use text if exist
                recognized_text = request.text
                if self.debug:
                    logger.info(f"Use text in request: {recognized_text}")
            elif request.audio_data:
                # Speech-to-Text
                recognized_text = (await self.stt.recognize(request.session_id, request.audio_data)).text
                if not recognized_text:
                    if self.debug:
                        logger.info("No speech recognized.")
                    return
                if self.debug:
                    logger.info(f"Recognized text from request: {recognized_text}")
            else:
                recognized_text = ""    # Request without both text and audio (e.g. image only)
            request.text = recognized_text

            performance.request_text = request.text
            performance.request_files = json.dumps(request.files or [], ensure_ascii=False)
            performance.voice_length = request.audio_duration
            performance.stt_time = time() - start_time

            # Get session state
            state = await self.session_state_manager.get_session_state(request.session_id)
            now = datetime.now(timezone.utc)

            # Merge consecutive requests
            if self.merge_request_threshold > 0 and request.allow_merge:
                if state.previous_request_timestamp:
                    requests_interval = (now - state.previous_request_timestamp).total_seconds()
                    if self.merge_request_threshold > requests_interval:
                        logger.info(f"Merge consecutive requests: Interval {requests_interval} < Threshold {self.merge_request_threshold}")
                        previous_request_text = (state.previous_request_text or "").replace(self.merge_request_prefix, "")
                        request.text = f"{self.merge_request_prefix}{previous_request_text}\n{request.text}"
                        request.files = request.files or state.previous_request_files
                await self.session_state_manager.update_previous_request(
                    request.session_id, now, request.text, request.files
                )

            last_created_at = await self.llm.context_manager.get_last_created_at(request.context_id)
            is_awake = self.is_awake(request, last_created_at)
            if is_awake:
                # Get context
                if request.context_id:
                    if last_created_at == datetime.min.replace(tzinfo=timezone.utc):
                        logger.info(f"Invalid context_id: {request.context_id}")
                        request.context_id = None

                if not request.context_id:
                    request.context_id = str(uuid4())
                    logger.info(f"Create new context_id: {request.context_id}")

                # Insert timestamp
                if self.timestamp_interval_seconds > 0 and (now - state.timestamp_inserted_at).total_seconds() > self.timestamp_interval_seconds:
                    now_str = datetime.now(ZoneInfo(self.timestamp_timezone)).strftime("%Y/%m/%d %H:%M:%S")
                    request.text = f"{self.timestamp_prefix}{now_str}\n\n{request.text}"
                    timestamp_inserted_at = now
                else:
                    timestamp_inserted_at = state.timestamp_inserted_at

                # Overwrite active transaction
                if self.debug:
                    logger.info(f"Start transaction: {transaction_id} {request.text} (previous: {state.active_transaction_id})")
                await self.session_state_manager.update_transaction(request.session_id, transaction_id, timestamp_inserted_at)
            else:
                # Clear request content to avoid LLM and TTS processing
                request.text = None
                request.files = {}

            performance.context_id = request.context_id

            # Stop on-going response before new response
            if is_awake and (not self.use_invoke_queue or not request.wait_in_queue):
                await self.stop_response(request.session_id, request.context_id)
            performance.stop_response_time = time() - start_time

            yield STSResponse(
                type="start",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"request_text": request.text, "recognized_text": recognized_text}
            )

            # LLM
            await self._on_before_llm(request)
            llm_stream = self.llm.chat_stream(request.context_id, request.user_id, request.text, request.files, request.system_prompt_params)

            # TTS
            async def synthesize_stream() -> AsyncGenerator[Tuple[bytes, LLMResponse], None]:
                voice_text = ""
                language = None
                async for llm_stream_chunk in llm_stream:
                    is_txn_active, active_txn = await self.is_transaction_active(request.session_id, transaction_id)
                    if not is_txn_active:
                        # Break when new transaction started in this session
                        if self.debug:
                            logger.info(f"Break llm_stream for new transaction: {active_txn} {request.text} (current: {transaction_id})")
                        break

                    # LLM performance
                    if performance.llm_first_chunk_time == 0:
                        performance.llm_first_chunk_time = time() - start_time

                    # ToolCall
                    if llm_stream_chunk.tool_call:
                        yield None, llm_stream_chunk, None, None
                        continue

                    # Voice
                    if llm_stream_chunk.voice_text:
                        voice_text += llm_stream_chunk.voice_text
                        if performance.llm_first_voice_chunk_time == 0:
                            performance.llm_first_voice_chunk_time = time() - start_time
                            await self._on_before_tts(request)
                    performance.llm_time = time() - start_time

                    # Parse language
                    if match := re.search(r"\[lang:([a-zA-Z-]+)\]", llm_stream_chunk.text):
                        language = match.group(1)

                    # Parse style info from LLM chunk
                    parsed_info = await self._process_llm_chunk(llm_stream_chunk)

                    audio_chunk = await self.tts.synthesize(
                        text=llm_stream_chunk.voice_text,
                        style_info={"styled_text": llm_stream_chunk.text, "info": parsed_info},
                        language=language
                    )

                    # TTS performance
                    if audio_chunk:
                        if performance.tts_first_chunk_time == 0:
                            performance.tts_first_chunk_time = time() - start_time
                        performance.tts_time = time() - start_time

                    yield audio_chunk, llm_stream_chunk, language, llm_stream_chunk.guradrail_name
                performance.response_voice_text = voice_text

            response_text = ""
            response_audios = []
            is_first_chunk = True
            async for audio_chunk, llm_stream_chunk, language, guradrail_name in synthesize_stream():
                is_txn_active, active_txn = await self.is_transaction_active(request.session_id, transaction_id)
                if not is_txn_active:
                    # Break when new transaction started in this session
                    if self.debug:
                        logger.info(f"Break synthesize_stream for new transaction: {active_txn} {request.text} (current: {transaction_id})")
                    break

                if llm_stream_chunk.tool_call:
                    yield STSResponse(
                        type="tool_call",
                        session_id=request.session_id,
                        user_id=request.user_id,
                        context_id=llm_stream_chunk.context_id,
                        tool_call=llm_stream_chunk.tool_call
                    )
                    continue

                response_text += llm_stream_chunk.text
                if audio_chunk:
                    response_audios.append(audio_chunk)

                yield STSResponse(
                    type="chunk",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=llm_stream_chunk.context_id,
                    text=llm_stream_chunk.text,
                    voice_text=llm_stream_chunk.voice_text,
                    language=language,
                    audio_data=audio_chunk,
                    metadata={"is_first_chunk": is_first_chunk, "is_guardrail_triggered": True if guradrail_name else False}
                )
                is_first_chunk = False

            performance.response_text = response_text
            performance.total_time = time() - start_time
            self.performance_recorder.record(performance)

            final_response = STSResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                text=response_text,
                voice_text=performance.response_voice_text
            )

            if self.voice_recorder_enabled:
                await self.voice_recorder.record(ResponseVoices(
                    transaction_id, response_audios, self.voice_recorder_response_audio_format
                ))
            await self._on_finish(request, final_response)
            yield final_response
        
        except Exception as iex:
            tb = traceback.format_exc()
            logger.error(f"Error at invoke: {iex}\n\n{tb}")

            yield STSResponse(
                type="error",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"error": "Error in processing Speech-to-Speech pipeline"}
            )

    async def _clear_queue(self, session_id: str):
        if session_id not in self._request_queues:
            return

        queue = self._request_queues[session_id]
        pending = self._response_queues.get(session_id, {})

        while not queue.empty():
            try:
                request_id, request = queue.get_nowait()
                response_queue = pending.get(request_id)
                if response_queue:
                    await response_queue.put(STSResponse(
                        type="cancelled",
                        session_id=session_id,
                        context_id=request.context_id
                    ))
                    await response_queue.put(None)
            except asyncio.QueueEmpty:
                break

    async def _process_queue(self, session_id: str):
        queue = self._request_queues[session_id]
        if self.debug:
            logger.info(f"Queue worker started: {session_id}")

        try:
            while True:
                try:
                    request_id, request = await asyncio.wait_for(
                        queue.get(), timeout=self.invoke_queue_idle_timeout
                    )
                except asyncio.TimeoutError:
                    if queue.empty():
                        if self.debug:
                            logger.info(f"Queue worker idle timeout, cleaning up: {session_id}")
                        self._cleanup_session_queue(session_id)
                        return
                    continue

                response_queue = self._response_queues.get(session_id, {}).get(request_id)
                try:
                    async with asyncio.timeout(self.invoke_timeout):
                        async for response in self._invoke_direct(request):
                            if response_queue:
                                await response_queue.put(response)
                except asyncio.TimeoutError:
                    logger.warning(f"invoke timed out: {session_id}")
                    if response_queue:
                        await response_queue.put(STSResponse(
                            type="error",
                            session_id=session_id,
                            context_id=request.context_id,
                            metadata={"error": "invoke timed out"}
                        ))
                except Exception as ex:
                    logger.error(f"invoke error in queue worker: {session_id} - {ex}")
                    if response_queue:
                        await response_queue.put(STSResponse(
                            type="error",
                            session_id=session_id,
                            context_id=request.context_id,
                            metadata={"error": "invoke error in queue worker"}
                        ))
                finally:
                    if response_queue:
                        await response_queue.put(None)
                    if session_id in self._response_queues:
                        self._response_queues[session_id].pop(request_id, None)

        except Exception as ex:
            logger.error(f"Queue worker crashed: {session_id} - {ex}")
            self._cleanup_session_queue(session_id)

    def _cleanup_session_queue(self, session_id: str):
        self._request_queues.pop(session_id, None)
        self._invoke_workers.pop(session_id, None)
        self._response_queues.pop(session_id, None)

    async def _invoke_queued(
        self,
        request: STSRequest
    ) -> AsyncGenerator[STSResponse, None]:
        session_id = request.session_id

        if session_id not in self._request_queues:
            self._request_queues[session_id] = asyncio.Queue()
            self._response_queues[session_id] = {}
            self._invoke_workers[session_id] = asyncio.create_task(
                self._process_queue(session_id)
            )

        if not request.wait_in_queue:
            await self._clear_queue(session_id)

        request_id = str(uuid4())
        response_queue: asyncio.Queue[STSResponse] = asyncio.Queue()
        self._response_queues[session_id][request_id] = response_queue

        await self._request_queues[session_id].put((request_id, request))

        while True:
            response = await response_queue.get()
            if response is None:
                break
            yield response

    async def finalize(self, context_id: str):
        await self.vad.finalize_session(context_id)

    async def shutdown(self):
        self.performance_recorder.close()
        await self.voice_recorder.stop()
