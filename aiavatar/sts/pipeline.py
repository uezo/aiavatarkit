from datetime import datetime, timezone
import json
import logging
from time import time
import traceback
from typing import AsyncGenerator, Tuple, List, Dict
from uuid import uuid4
from .models import STSRequest, STSResponse
from .vad import SpeechDetector, StandardSpeechDetector
from .stt import SpeechRecognizer
from .stt.google import GoogleSpeechRecognizer
from .llm import LLMService, LLMResponse
from .llm.chatgpt import ChatGPTService
from .tts import SpeechSynthesizer
from .tts.voicevox import VoicevoxSpeechSynthesizer
from .performance_recorder import PerformanceRecord, PerformanceRecorder
from .performance_recorder.sqlite import SQLitePerformanceRecorder
from .voice_recorder import VoiceRecorder, RequestVoice, ResponseVoices
from .voice_recorder.file import FileVoiceRecorder

logger = logging.getLogger(__name__)


class STSPipeline:
    def __init__(
        self,
        *,
        vad: SpeechDetector = None,
        vad_volume_db_threshold: float = -50.0,
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
        tts: SpeechSynthesizer = None,
        tts_voicevox_url: str = "http://127.0.0.1:50021",
        tts_voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        wakeword_timeout: float = 60.0,
        performance_recorder: PerformanceRecorder = None,
        voice_recorder: VoiceRecorder = None,
        voice_recorder_enabled: bool = True,
        debug: bool = False
    ):
        self.debug = debug

        # Cancellation
        self.active_transactions: Dict[str, str] = {}

        # VAD
        self.vad = vad or StandardSpeechDetector(
            volume_db_threshold=vad_volume_db_threshold,
            silence_duration_threshold=vad_silence_duration_threshold,
            sample_rate=vad_sample_rate,
            debug=debug
        )

        @self.vad.on_speech_detected
        async def on_speech_detected(data: bytes, recorded_duration: float, session_id: str):
            async for response in self.invoke(STSRequest(
                session_id=session_id,
                user_id=self.vad.get_session_data(session_id, "user_id"),
                context_id=self.vad.get_session_data(session_id, "context_id"),
                audio_data=data,
                audio_duration=recorded_duration
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
        self.llm = llm or ChatGPTService(
            openai_api_key=llm_openai_api_key,
            base_url=llm_base_url,
            model=llm_model,
            system_prompt=llm_system_prompt,
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

        # Response handler
        self.handle_response = self.handle_response_default
        self.stop_response = self.stop_response_default
        self._process_llm_chunk = self.process_llm_chunk_default

        # Performance recorder
        self.performance_recorder = performance_recorder or SQLitePerformanceRecorder()

        # Voice recorder
        self.voice_recorder = voice_recorder or FileVoiceRecorder(
            sample_rate=stt_sample_rate
        )
        self.voice_recorder_enabled = voice_recorder_enabled
        self.voice_recorder_response_audio_format = "wav"

        # User custom logic
        self._on_before_llm = self.on_before_llm_default
        self._on_before_tts = self.on_before_tts_default
        self._on_finish = self.on_finish_default

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

    def is_transaction_active(self, session_id: str, transaction_id: str) -> bool:
        return self.active_transactions.get(session_id) == transaction_id

    async def invoke(self, request: STSRequest) -> AsyncGenerator[STSResponse, None]:
        try:
            start_time = time()
            transaction_id = str(uuid4())

            performance = PerformanceRecord(
                transaction_id=transaction_id,
                user_id=request.user_id,
                stt_name=self.stt.__class__.__name__,
                llm_name=self.llm.__class__.__name__,
                tts_name=self.tts.__class__.__name__
            )

            if request.text:
                # Use text if exist
                recognized_text = request.text
                if self.debug:
                    logger.info(f"Use text in request: {recognized_text}")
            elif request.audio_data:
                if self.voice_recorder_enabled:
                    await self.voice_recorder.record(RequestVoice(transaction_id, request.audio_data))
                # Speech-to-Text
                recognized_text = await self.stt.transcribe(request.audio_data)
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

            last_created_at = await self.llm.context_manager.get_last_created_at(request.context_id)
            if self.is_awake(request, last_created_at):
                # Get context
                if request.context_id:
                    if last_created_at == datetime.min.replace(tzinfo=timezone.utc):
                        logger.info(f"Invalid context_id: {request.context_id}")
                        request.context_id = None

                if not request.context_id:
                    request.context_id = str(uuid4())
                    logger.info(f"Create new context_id: {request.context_id}")

                # Overwrite active transaction
                if self.debug:
                    logger.info(f"Start transaction: {transaction_id} {request.text} (previous: {self.active_transactions.get(request.session_id)})")
                self.active_transactions[request.session_id] = transaction_id
            else:
                # Clear request content to avoid LLM and TTS processing
                request.text = None
                request.files = {}

            performance.context_id = request.context_id

            # Stop on-going response before new response
            await self.stop_response(request.session_id, request.context_id)
            performance.stop_response_time = time() - start_time

            yield STSResponse(
                type="start",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"request_text": request.text}
            )

            # LLM
            await self._on_before_llm(request)
            llm_stream = self.llm.chat_stream(request.context_id, request.user_id, request.text, request.files, request.system_prompt_params)

            # TTS
            async def synthesize_stream() -> AsyncGenerator[Tuple[bytes, LLMResponse], None]:
                voice_text = ""
                language = None
                async for llm_stream_chunk in llm_stream:
                    if not self.is_transaction_active(request.session_id, transaction_id):
                        # Break when new transaction started in this session
                        if self.debug:
                            logger.info(f"Break llm_stream for new transaction: {self.active_transactions.get(request.session_id)} {request.text} (current: {transaction_id})")
                        break

                    # LLM performance
                    if performance.llm_first_chunk_time == 0:
                        performance.llm_first_chunk_time = time() - start_time

                    # ToolCall
                    if llm_stream_chunk.tool_call:
                        yield None, llm_stream_chunk, None
                        continue

                    # Voice
                    if llm_stream_chunk.voice_text:
                        voice_text += llm_stream_chunk.voice_text
                        if performance.llm_first_voice_chunk_time == 0:
                            performance.llm_first_voice_chunk_time = time() - start_time
                            await self._on_before_tts(request)
                    performance.llm_time = time() - start_time

                    # Parse info from LLM chunk (especially, language)
                    parsed_info = await self._process_llm_chunk(llm_stream_chunk)
                    language = parsed_info.get("language") or language

                    audio_chunk = await self.tts.synthesize(
                        text=llm_stream_chunk.voice_text,
                        style_info={"styled_text": llm_stream_chunk.text},
                        language=language
                    )

                    # TTS performance
                    if audio_chunk:
                        if performance.tts_first_chunk_time == 0:
                            performance.tts_first_chunk_time = time() - start_time
                        performance.tts_time = time() - start_time

                    yield audio_chunk, llm_stream_chunk, language
                performance.response_voice_text = voice_text

            response_text = ""
            response_audios = []
            is_first_chunk = True
            async for audio_chunk, llm_stream_chunk, language in synthesize_stream():
                if not self.is_transaction_active(request.session_id, transaction_id):
                    # Break when new transaction started in this session
                    if self.debug:
                        logger.info(f"Break synthesize_stream for new transaction: {self.active_transactions.get(request.session_id)} {request.text} (current: {transaction_id})")
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
                    metadata={"is_first_chunk": is_first_chunk}
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

    async def finalize(self, context_id: str):
        await self.vad.finalize_session(context_id)

    async def shutdown(self):
        self.performance_recorder.close()
        await self.voice_recorder.stop()
