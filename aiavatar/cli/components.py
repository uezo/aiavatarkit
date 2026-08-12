import inspect
import logging
from dataclasses import dataclass, field

from aiavatar.sts.llm import LLMService
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.llm.openai_responses_websocket import (
    OpenAIResponsesWebSocketService,
)
from aiavatar.sts.stt import SpeechRecognizer
from aiavatar.sts.stt.openai import OpenAISpeechRecognizer
from aiavatar.sts.tts import SpeechSynthesizer
from aiavatar.sts.vad import SpeechDetector
from aiavatar.sts.vad.filters import NearFieldAudioGate
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_end_gates.filler import FillerOnlyTurnEndGate
from aiavatar.sts.vad.turn_end_gates.namo_turn import NamoTurnEndGate

from .config import AppConfig
from .tts import build_default_tts


logger = logging.getLogger(__name__)


def websocket_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    base_url = base_url.rstrip("/").removesuffix("/v1")
    if base_url.startswith("https://"):
        return "wss://" + base_url.removeprefix("https://")
    if base_url.startswith("http://"):
        return "ws://" + base_url.removeprefix("http://")
    return base_url


def create_default_llm(config: AppConfig) -> LLMService:
    if not config.llm_openai.api_key:
        raise RuntimeError(
            "OpenAI API key is required for LLM; set OPENAI_API_KEY or "
            "AIAVATAR_LLM_OPENAI_API_KEY"
        )
    common_kwargs = {
        "openai_api_key": config.llm_openai.api_key,
        "model": config.llm_model,
        "reasoning_effort": config.llm_reasoning_effort,
        "extra_body": config.llm_extra_body,
        "system_prompt": config.llm_system_prompt,
        "voice_text_tag": config.llm_voice_text_tags,
        "debug": config.debug,
    }
    if config.llm_api == "chat-completions":
        return ChatGPTService(
            base_url=config.llm_openai.base_url,
            **common_kwargs,
        )
    if config.llm_api == "responses-websocket":
        return OpenAIResponsesWebSocketService(
            ws_url=websocket_base_url(config.llm_openai.base_url),
            **common_kwargs,
        )
    raise RuntimeError(
        "AIAVATAR_LLM_API must be 'responses-websocket' or 'chat-completions'"
    )


@dataclass
class ComponentSet:
    """The VAD, STT, LLM, and TTS selected for one application."""

    vad: SpeechDetector
    stt: SpeechRecognizer
    llm: LLMService
    tts: SpeechSynthesizer
    _managed_resources: list = field(default_factory=list, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __iter__(self):
        yield self.vad
        yield self.stt
        yield self.llm
        yield self.tts

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        llm_resource = getattr(self.llm, "_ws_pool", None)
        if llm_resource is None:
            llm_resource = getattr(self.llm, "openai_client", None)
        if llm_resource is None and (
            hasattr(self.llm, "close") or hasattr(self.llm, "aclose")
        ):
            llm_resource = self.llm

        resources = [
            llm_resource,
            self.stt,
            self.tts,
            *self._managed_resources,
        ]
        closed = set()
        for resource in resources:
            if resource is None or id(resource) in closed:
                continue
            closed.add(id(resource))
            close = getattr(resource, "close", None) or getattr(
                resource,
                "aclose",
                None,
            )
            if close is None:
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Error closing %s", resource.__class__.__name__)


def build_components(
    config: AppConfig | None = None,
    *,
    vad: SpeechDetector | None = None,
    stt: SpeechRecognizer | None = None,
    llm: LLMService | None = None,
    tts: SpeechSynthesizer | None = None,
) -> ComponentSet:
    """Build missing speech components without creating a pipeline or adapter.

    Explicitly supplied components are kept, and the selected STT is injected
    into the default streaming VAD when ``vad`` is omitted.
    """
    config = config or AppConfig.from_env()
    managed_resources = []

    if stt is None:
        if not config.stt_openai.api_key:
            raise RuntimeError(
                "OpenAI API key is required for STT; set OPENAI_API_KEY or "
                "AIAVATAR_STT_OPENAI_API_KEY"
            )
        stt = OpenAISpeechRecognizer(
            openai_api_key=config.stt_openai.api_key,
            base_url=config.stt_openai.base_url,
            model=config.stt_model,
            language=config.stt_language,
            debug=config.debug,
        )

    if vad is None:
        near_field_gate = NearFieldAudioGate(
            initial_ambient_db=config.near_field_initial_ambient_db,
        )
        filler_gate = FillerOnlyTurnEndGate(
            fillers=config.filler_phrases,
            timeout=config.filler_timeout,
        )
        namo_turn_gate = NamoTurnEndGate(
            language=config.namo_language,
            threshold=config.namo_threshold,
            timeout=config.namo_timeout,
            force_end_phrases=config.namo_force_end_phrases,
        )
        vad = SileroStreamSpeechDetector(
            speech_recognizer=stt,
            silence_duration_threshold=config.vad_silence_duration_threshold,
            segment_silence_threshold=config.vad_segment_silence_threshold,
            audio_filters=[near_field_gate],
            turn_end_gates=[filler_gate, namo_turn_gate],
            use_vad_iterator=config.vad_use_iterator,
            debug=config.debug,
        )

    if llm is None:
        llm = create_default_llm(config)

    if tts is None:
        tts, alphabet_to_kana = build_default_tts(config)
        if alphabet_to_kana is not None:
            managed_resources.append(alphabet_to_kana.http_client)

    return ComponentSet(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        _managed_resources=managed_resources,
    )
