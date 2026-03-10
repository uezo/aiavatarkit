import asyncio
import logging
import random
from time import time
from typing import Dict, List
from .pipeline import STSPipeline
from .models import STSRequest
from .llm import LLMService
from .tts import SpeechSynthesizer

DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX = "$The following is the user's utterance. Respond with a very short phrase of no more than 5 words that serves as an appropriate opening acknowledgment. The phrase must end with punctuation such as a period, comma, or exclamation mark. For this output only, do not use tag formats like <think> or <answer>—output only the phrase."
DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA = "$以下はユーザーの発話内容である。ユーザー発話を受け止めて、第一声として相応しい、10文字以内のごく短いフレーズを出力せよ。応答の末尾は「。」や「、」句読点や感嘆符とする。この出力に限っては<think>や<answer>のタグフォーマットは不要で、フレーズのみを出力すること。"
DEFAULT_REQUEST_PREFIX = "$For the following input, you have already output \"{quick_response_text}\"—do NOT repeat it or any similar expression. Output only the continuation. Follow the <think></think><answer></answer> format for this response:"
DEFAULT_REQUEST_PREFIX_JA = "$以下の入力に対して、既にあなたが出力済みの「{quick_response_text}」や類似の表現は再出力せず、その続きだけを出力せよ。今回は<think></think><answer></answer>などフォーマットに従うこと:"
DEFAULT_FALLBACK_PHRASES = ["I see.", "Right.", "Sure.", "Got it."]
DEFAULT_FALLBACK_PHRASES_JA = ["はい。"]

logger = logging.getLogger(__name__)


class QuickResponder:
    def __init__(
        self,
        llm: LLMService,
        tts: SpeechSynthesizer,
        *,
        inline_llm_params: Dict[str, any] = None,
        quick_response_prompt_prefix: str = None,
        request_prefix: str = None,
        timeout: float = 1.5,
        fallback_phrases: List[str] = None,
        debug: bool = False
    ):
        self.llm = llm
        self.tts = tts
        self.inline_llm_params = inline_llm_params or {"reasoning_effort": "none", "tools": [], "tool_choice": "none"}
        self.quick_response_prompt_prefix = quick_response_prompt_prefix or DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX
        self.request_prefix = request_prefix or DEFAULT_REQUEST_PREFIX
        self.timeout = timeout
        self.fallback_phrases = fallback_phrases or DEFAULT_FALLBACK_PHRASES
        self.debug = debug
        self.voice_cache = {}

    async def _generate_from_llm(self, request: STSRequest):
        start_time = time()

        qr_text = ""
        qr_voice_text = ""
        async for chunk in self.llm.chat_stream(
            context_id=request.context_id,
            user_id=request.user_id,
            text=f"{self.quick_response_prompt_prefix}\n\n{request.text}",
            inline_llm_params=self.inline_llm_params
        ):
            if t := chunk.text:
                qr_text += t
            if vt := chunk.voice_text:
                qr_voice_text += vt

        llm_time = time() - start_time

        if not qr_voice_text:
            qr_voice_text = qr_text

        if qr_voice := self.voice_cache.get(qr_text):
            tts_time = 0.0
        else:
            tts_start = time()
            qr_voice = await self.tts.synthesize(qr_text)
            self.voice_cache[qr_text] = qr_voice
            tts_time = time() - tts_start

        total_time = time() - start_time
        if self.debug:
            logger.info(f"Quick response: {qr_text} ({total_time:.3f}s = LLM {llm_time:.3f}s + TTS {tts_time:.3f}s)")

        return qr_text, qr_voice_text, qr_voice

    async def _generate_fallback(self):
        qr_text = random.choice(self.fallback_phrases)

        if qr_voice := self.voice_cache.get(qr_text):
            tts_time = 0.0
        else:
            tts_start = time()
            qr_voice = await self.tts.synthesize(qr_text)
            self.voice_cache[qr_text] = qr_voice
            tts_time = time() - tts_start

        if self.debug:
            logger.info(f"Quick response (fallback): {qr_text} (TTS {tts_time:.3f}s)")

        return qr_text, qr_text, qr_voice

    async def _generate(self, request: STSRequest):
        if self.timeout and self.timeout > 0:
            try:
                return await asyncio.wait_for(
                    self._generate_from_llm(request),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                if self.debug:
                    logger.warning(f"Quick response timed out ({self.timeout}s), using fallback")
                return await self._generate_fallback()
        else:
            return await self._generate_from_llm(request)

    async def respond(self, request: STSRequest):
        # Generate quick response
        qr_text, qr_voice_text, qr_voice = await self._generate(request)

        # Store quick response in request for pipeline to yield
        request.quick_response_text = qr_text
        request.quick_response_voice_text = qr_voice_text
        request.quick_response_audio = qr_voice

        # Overwrite request to avoid duplication
        prefix = self.request_prefix.format(quick_response_text=qr_text)
        request.text = f"{prefix}\n\n{request.text}"

    def clear_voice_cache(self):
        self.voice_cache.clear()
