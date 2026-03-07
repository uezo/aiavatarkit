from .pipeline import STSPipeline
from .models import STSRequest, STSResponse

DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX = "$The following is the user's utterance. Respond with a very short phrase of no more than 5 words that serves as an appropriate opening acknowledgment. The phrase must end with punctuation such as a period, comma, or exclamation mark. For this output only, do not use tag formats like <think> or <answer>—output only the phrase."
DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA = "$以下はユーザーの発話内容である。ユーザー発話を受け止めて、第一声として相応しい、10文字以内のごく短いフレーズを出力せよ。応答の末尾は「。」や「、」句読点や感嘆符とする。この出力に限っては<think>や<answer>のタグフォーマットは不要で、フレーズのみを出力すること。"
DEFAULT_REQUEST_PREFIX = "$For the following input, you have already output \"{quick_response_text}\"—do NOT repeat it or any similar expression. Output only the continuation. Follow the <think></think><answer></answer> format for this response:"
DEFAULT_REQUEST_PREFIX_JA = "$以下の入力に対して、既にあなたが出力済みの「{quick_response_text}」や類似の表現は再出力せず、その続きだけを出力せよ。今回は<think></think><answer></answer>などフォーマットに従うこと:"


class QuickResponder:
    def __init__(self, pipeline: STSPipeline, *, quick_response_prompt_prefix: str = None, request_prefix: str = None):
        self.pipeline = pipeline
        self.quick_response_prompt_prefix = quick_response_prompt_prefix or DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX
        self.request_prefix = request_prefix or DEFAULT_REQUEST_PREFIX
        self.voice_cache = {}

    async def _generate(self, request: STSRequest):
        qr_text = ""
        qr_voice_text = ""
        async for chunk in self.pipeline.llm.chat_stream(
            context_id=request.context_id,
            user_id=request.user_id,
            text=f"{self.quick_response_prompt_prefix}\n\n{request.text}",
        ):
            if t := chunk.text:
                qr_text += t
            if vt := chunk.voice_text:
                qr_voice_text += vt

        if not qr_voice_text:
            qr_voice_text = qr_text

        if qr_voice := self.voice_cache.get(qr_text):
            return qr_text, qr_voice_text, qr_voice
        else:
            qr_voice = await self.pipeline.tts.synthesize(qr_text)
            self.voice_cache[qr_text] = qr_voice
            return qr_text, qr_voice_text, qr_voice

    async def respond(self, request: STSRequest):
        # Respond quick response
        qr_text, qr_voice_text, qr_voice = await self._generate(request)
        await self.pipeline.handle_response(
            STSResponse(
                type="chunk",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                text=qr_text,
                voice_text=qr_voice_text,
                audio_data=qr_voice,
                metadata={}
            )
        )

        # Overwrite request to avoid duplication
        prefix = self.request_prefix.format(quick_response_text=qr_text)
        request.text = f"{prefix}\n\n{request.text}"

    def clear_voice_cache(self):
        self.voice_cache.clear()
