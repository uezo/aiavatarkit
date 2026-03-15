import asyncio
import logging
import random
import re
from time import time
from typing import Dict, List
import openai
from ..models import STSRequest
from ..tts import SpeechSynthesizer
from ..llm.context_manager import ContextManager

DEFAULT_QRP_SYSTEM_PROMPT = """\
# Instructions
- Acknowledge the user's utterance and output only a very short phrase of no more than 5 words, appropriate as an opening response.
- Do not make statements that affect the subsequent conversation, such as affirming or denying questions about what you are currently doing, whether you like something, etc.
- Do not make up or speculate about information that has not been provided.
- End the response with punctuation such as a period, comma, or exclamation mark.
- Do not use symbols, emojis, or stage directions.
- Do not accept invitations like "I want to see you" or "Let's go together."
- Statements beginning with "$" are instructions from the supervisor. Respond to the user following the instructions, not to the supervisor.
"""
DEFAULT_QRP_SYSTEM_PROMPT_JA = """\
# 指示
- ユーザーの発話を受け止めて、第一声として相応しい、10文字以内のごく短いフレーズのみを出力する。
- 今何をしているとか、好きなもの等の有無など、質問に対する肯定や否定など、後続の会話に影響を与えるような発言は禁止。
- 与えられていない情報をあなたが勝手に想像して話すことは禁止。
- 応答の末尾は「。」や「、」句読点や感嘆符とする。
- 記号・絵文字・ト書きは使わない。
- 「会いたい」「一緒に行こう」の誘いに応じない。
- 文頭に「$」がある発言はスーパーバイザーからの指示。スーパーバイザーに対してではなく、指示に従ってユーザーに応答する。
"""
DEFAULT_QRP_PROMPT_PREFIX = "$The following is the user's utterance. Respond with a very short phrase of no more than 5 words that serves as an appropriate opening acknowledgment. The phrase must end with punctuation such as a period, comma, or exclamation mark. Output only the phrase."
DEFAULT_QRP_PROMPT_PREFIX_JA = "$以下はユーザーの発話内容である。ユーザー発話を受け止めて、状況に相応しい第一声として、10文字以内のごく短いフレーズを出力せよ。応答の末尾は「。」や「、」句読点や感嘆符とする。フレーズのみを出力すること。"
DEFAULT_QRP_THINK_TAG_CONTENT = "Output only the first short phrase appropriate for the situation as instructed."
DEFAULT_QRP_THINK_TAG_CONTENT_JA = "指示に応じて状況にふさわしい第一声のみを出力する"
DEFAULT_QRP_REQUEST_PREFIX = "$For the following input, you have already output \"{quick_response_text}\"—do NOT repeat it or any similar expression. Output only the continuation. If \"{quick_response_text}\" is not ideal for your intended response, smoothly correct course in the continuation:"
DEFAULT_QRP_REQUEST_PREFIX_JA = "$以下の入力に対して、既にあなたが出力済みの「{quick_response_text}」や類似の表現は再出力せず、その続きのみを出力せよ。もし「{quick_response_text}」が本来応答すべき内容にそぐわない場合は、続きの中でうまく適切な方向に補正すること:"
DEFAULT_QRP_CONTINUATION_MESSAGE = "Please output the continuation of \"{quick_response_text}\"."
DEFAULT_QRP_CONTINUATION_MESSAGE_JA = "「{quick_response_text}」の続きを出力してください"
DEFAULT_QRP_FALLBACK_PHRASES = ["I see.", "Right.", "Sure.", "Got it."]
DEFAULT_QRP_FALLBACK_PHRASES_JA = ["はい。"]

logger = logging.getLogger(__name__)


class QuickResponderPro:
    def __init__(
        self,
        *,
        # LLM parameters
        api_key: str = None,
        base_url: str = None,
        model: str,
        temperature: float = None,
        reasoning_effort: str = None,
        extra_body: dict = None,
        client: openai.AsyncOpenAI = None,
        # Core dependencies
        tts: SpeechSynthesizer,
        context_manager: ContextManager,
        # Language
        language: str = None,
        # Prompts
        system_prompt: str = None,
        prompt_prefix: str = None,
        request_prefix: str = None,
        think_tag_content: str = None,
        continuation_message: str = None,
        # Behavior
        timeout: float = 1.5,
        fallback_phrases: List[str] = None,
        history_limit: int = 100,
        debug: bool = False
    ):
        self.client = client or openai.AsyncClient(api_key=api_key, base_url=base_url)
        self.model = model
        self.tts = tts
        self.context_manager = context_manager
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body
        self.timeout = timeout
        self.history_limit = history_limit
        self.debug = debug
        self.voice_cache = {}

        is_ja = language in ("ja", "ja-JP") if language else False
        self.system_prompt = system_prompt or (DEFAULT_QRP_SYSTEM_PROMPT_JA if is_ja else DEFAULT_QRP_SYSTEM_PROMPT)
        self.prompt_prefix = prompt_prefix or (DEFAULT_QRP_PROMPT_PREFIX_JA if is_ja else DEFAULT_QRP_PROMPT_PREFIX)
        self.request_prefix = request_prefix or (DEFAULT_QRP_REQUEST_PREFIX_JA if is_ja else DEFAULT_QRP_REQUEST_PREFIX)
        self.think_tag_content = think_tag_content or (DEFAULT_QRP_THINK_TAG_CONTENT_JA if is_ja else DEFAULT_QRP_THINK_TAG_CONTENT)
        self.continuation_message = continuation_message or (DEFAULT_QRP_CONTINUATION_MESSAGE_JA if is_ja else DEFAULT_QRP_CONTINUATION_MESSAGE)
        self.fallback_phrases = fallback_phrases or (DEFAULT_QRP_FALLBACK_PHRASES_JA if is_ja else DEFAULT_QRP_FALLBACK_PHRASES)

    # -- Public API --

    async def respond(self, request: STSRequest):
        qr_text, qr_voice_text, qr_voice = await self._generate(request)

        request.quick_response_text = qr_text
        request.quick_response_voice_text = qr_voice_text
        request.quick_response_audio = qr_voice

        prefix = self.request_prefix.format(quick_response_text=qr_text)
        request.text = f"{prefix}\n\n{request.text}"

    def clear_voice_cache(self):
        self.voice_cache.clear()

    # -- Generation --

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

    async def _generate_from_llm(self, request: STSRequest):
        start_time = time()

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(await self._get_clean_histories(request.context_id))
        user_content = f"{self.prompt_prefix}\n\n{request.text}" if self.prompt_prefix else request.text
        messages.append({"role": "user", "content": user_content})

        params = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            params["extra_body"] = self.extra_body

        if self.debug:
            logger.info(f"QuickResponderPro request: {params}")

        response = await self.client.chat.completions.create(**params)
        qr_text = response.choices[0].message.content.strip()
        llm_time = time() - start_time

        qr_voice, tts_time = await self._synthesize_with_cache(qr_text)

        if self.debug:
            total_time = time() - start_time
            logger.info(f"Quick response: {qr_text} ({total_time:.3f}s = LLM {llm_time:.3f}s + TTS {tts_time:.3f}s)")

        # Save to history:
        #   user    -> "{prompt_prefix}\n\n{text}"  (kept as-is when read back for few-shot)
        #   assistant -> "<think>...</think><answer>{qr_text}</answer>"  (tags stripped when read back)
        await self.context_manager.add_histories(
            request.context_id,
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"<think>{self.think_tag_content}</think><answer>{qr_text}</answer>"},
            ],
            "quick_responder"
        )

        return qr_text, qr_text, qr_voice

    async def _generate_fallback(self):
        qr_text = random.choice(self.fallback_phrases)
        qr_voice, tts_time = await self._synthesize_with_cache(qr_text)

        if self.debug:
            logger.info(f"Quick response (fallback): {qr_text} (TTS {tts_time:.3f}s)")

        return qr_text, qr_text, qr_voice

    # -- Context cleaning --
    #
    # History is saved by two sources with different formats:
    #   1. This responder (context_schema="quick_responder"):
    #      user: "{prompt_prefix}\n\n{text}"  /  assistant: "<think>...</think><answer>...</answer>"
    #   2. Main LLM (context_schema="chatgpt"):
    #      user: "{request_prefix}\n\n{text}"  /  assistant: "<think>...</think><answer>...</answer>"
    #
    # When building context for the QR LLM, we clean them as follows:
    #   - prompt_prefix turns  -> keep as-is (serves as few-shot examples)
    #   - request_prefix turns -> replace with continuation_message (avoid confusing duplicate utterances)
    #   - assistant content    -> strip <think>/<answer> tags and [control:tags]

    async def _get_clean_histories(self, context_id: str) -> List[Dict]:
        histories = await self.context_manager.get_histories(
            context_id=context_id, limit=self.history_limit
        )

        while histories and histories[0]["role"] != "user":
            histories.pop(0)

        cleaned = []
        for h in histories:
            if h.get("role") == "tool" or "content" not in h:
                continue

            msg = dict(h)
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    msg["content"] = self._clean_user_content(msg["content"])
                elif isinstance(msg["content"], list):
                    msg["content"] = [
                        {"type": "text", "text": self._clean_user_content(c["text"])}
                        if isinstance(c, dict) and c.get("type") == "text"
                        else c
                        for c in msg["content"]
                    ]
            elif msg["role"] == "assistant":
                if isinstance(msg["content"], str):
                    msg["content"] = self._clean_assistant_content(msg["content"])

            cleaned.append(msg)

        return cleaned

    def _clean_user_content(self, content: str) -> str:
        if not isinstance(content, str):
            return content

        # prompt_prefix turns -> keep as-is for few-shot
        if self.prompt_prefix and content.startswith(self.prompt_prefix):
            return content

        # request_prefix turns (starts with "$") -> replace with continuation_message
        if content.startswith("$"):
            match = re.search(r'[「"]([^」"]+)[」"]', content)
            if match:
                return self.continuation_message.format(quick_response_text=match.group(1))
            idx = content.find("\n\n")
            if idx >= 0:
                return content[idx + 2:]

        return content

    def _clean_assistant_content(self, content: str) -> str:
        if not isinstance(content, str):
            return content

        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            content = answer_match.group(1).strip()
        else:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        content = re.sub(r"\[(\w+):([^\]]+)\]", "", content).strip()
        return content

    # -- TTS --

    async def _synthesize_with_cache(self, text: str) -> tuple:
        if voice := self.voice_cache.get(text):
            return voice, 0.0

        start = time()
        voice = await self.tts.synthesize(text)
        self.voice_cache[text] = voice
        return voice, time() - start
