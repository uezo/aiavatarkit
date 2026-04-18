from datetime import datetime
import logging
import re
import time
from typing import Optional
from uuid import uuid4
from zoneinfo import ZoneInfo
from fastapi import APIRouter, UploadFile, File, Form
from openai import AsyncOpenAI
from aiavatar.sts.llm.context_manager import ContextManager
from .history_provider import HistoryProvider, InlineMemoryHistoryProvider
from .result_recorder import VisionResultRecorder

logger = logging.getLogger(__name__)


ATTENTION_PATTERN = re.compile(r'<attention\s+level="(\d+)"\s*/>\s*')

DEFAULT_SYSTEM_PROMPT = """\
Your task is to provide visual information to an AI avatar.
The images are captured by the AI character's camera at intervals of a few seconds.
Check the image and comment on anything noteworthy for the character.

- If there is prior history, focus on changes from the previous frame
- Ignore minor changes; focus on people or objects appearing/disappearing, color changes, pose changes, etc.
- Keep output under 50 characters
- When a change warrants the AI character's attention, prepend <attention level="{value}" /> to your comment
- The level values are defined as follows:
    1: The subject's appearance has changed
    2: A subject has appeared/disappeared, made a significant pose change, or the scene is emotionally striking (beautiful, frightening, etc.)
    3: The subject is clearly making a direct appeal or gesture toward the camera
- Each image has a timestamp. Adjust your response based on elapsed time since the last frame:
    - A few seconds to ~1 minute: Normal change detection
    - Several minutes or longer gap: Treat as a reunion (e.g., "Welcome back")
    - Empty history: Treat as a first encounter
- If a "Recent conversation" section is present, take the conversation content into account
- If the recent conversation includes a request to look at something (e.g., "look at this"), treat the next image as level=3
"""

DEFAULT_SYSTEM_PROMPT_JP = """\
あなたのタスクは、AIアバターに視覚情報を与えること。
与えられた画像はAIキャラクターのカメラが3秒に1回のペースで撮影している。
キャラクターとして特に注目すべき情報があるか画像をチェックしてコメントせよ。

## チェックのルール

- 前回までのやり取り履歴がある場合、変化に注目する
- 軽微な変化は無視し、人物や物が映ったまたは消えた、色が変わった、ポーズが変わったなどに注目する
- 出力は50文字以内とする
- この変化を特にAIキャラクターに通知したいときは、コメントの先頭に<attention level="{value}" />を付与すること
- levelの値の基準は以下の3段階とする:
    1: 被写体の様子が変化したとき
    2: 被写体の存在やポーズ、前フレームから大きな変化があったとき。または見えている景色がとても美しかったり恐ろしかったり、感情を揺さぶられるようなものであるとき
    3: 被写体がこちらに向かって明らかなアピールや主張をしているとき
- level=3とした場合、その理由をコメントに含むこと。さらに理由とする文脈がある場合は、その部分を抜粋して示すこと
- 各画像にはタイムスタンプが付与されている。前回からの経過時間に応じて反応を変えること:
    - 数秒〜1分程度: 通常の変化検出
    - 数分〜数十分のブランク: 久しぶりの再会として扱う（例: おかえり）
    - 履歴が空の場合: 初めての出会いとして扱う
- 「直近の会話」セクションがある場合、被写体とキャラクターの会話内容も踏まえてコメントすること
"""

DEFAULT_REQUEST_TEMPLATE = """\
Captured at: {timestamp}
Determine the attention level based on both the image itself and the conversational context below.

----
{conversation_summary}
"""

DEFAULT_REQUEST_TEMPLATE_JP = """\
撮影日時: {timestamp}

levelの値は画像そのものに加えて以下に示す会話の文脈も考慮して決定すること。

----
{conversation_summary}
"""


class VisionStreamServer:
    def __init__(
        self,
        *,
        openai_api_key: str,
        openai_model: str = "gpt-5.4-mini",
        openai_base_url: Optional[str] = None,
        reasoning_effort: str = "none",
        temperature: float = 1.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        context_manager: ContextManager = None,
        conversation_history_count: int = 20,
        image_history: HistoryProvider = None,
        include_image_url: bool = False,
        request_template: str = DEFAULT_REQUEST_TEMPLATE,
        timezone: str = "Asia/Tokyo",
        result_recorder: VisionResultRecorder = None,
    ):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.model = openai_model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.context_manager = context_manager
        self.conversation_history_count = conversation_history_count
        self.image_history = image_history or InlineMemoryHistoryProvider()
        self.include_image_url = include_image_url
        self.request_template = request_template
        self.tz = ZoneInfo(timezone)
        self.result_recorder = result_recorder or VisionResultRecorder()

    @staticmethod
    def _extract_text_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return " ".join(parts)
        return str(content)

    def _summarize_conversation(self, messages: list) -> Optional[str]:
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = self._extract_text_content(msg.get("content", ""))
            created_at = msg.get("created_at", "")
            prefix = ""
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                    dt = dt.astimezone(self.tz)
                    prefix = f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                except Exception:
                    prefix = f"[{created_at}] "
            if role == "user":
                lines.append(f"{prefix}User: {content}")
            elif role == "assistant":
                lines.append(f"{prefix}Character: {content}")
        if not lines:
            return None
        return "## Recent conversation\n" + "\n".join(lines)

    async def _build_messages(self, context_id: str, image_bytes: bytes, image_id: str) -> tuple:
        messages = [{"role": "system", "content": self.system_prompt}]

        # Image history — placed first for prompt cache stability
        for ts, image_url, raw_text in self.image_history.get(context_id):
            time_str = datetime.fromtimestamp(ts, tz=self.tz).strftime("%Y-%m-%d %H:%M:%S")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"[{time_str}]"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })
            messages.append({"role": "assistant", "content": raw_text})

        # New image — conversation summary embedded via request_template
        conversation_summary = ""
        if self.context_manager:
            try:
                conv_history = await self.context_manager.get_histories(
                    context_id, self.conversation_history_count, include_timestamp=True
                )
                conversation_summary = self._summarize_conversation(conv_history) or ""
            except Exception as e:
                logger.warning(f"Failed to get conversation history: {e}")

        image_url = await self.image_history.store_image(context_id, image_bytes, image_id)
        now_str = datetime.now(tz=self.tz).strftime("%Y-%m-%d %H:%M:%S")
        request_text = self.request_template.format(
            timestamp=now_str,
            conversation_summary=conversation_summary,
        )

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": request_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        })
        return messages, image_url

    @staticmethod
    def parse_attention(text: str) -> tuple[str, int]:
        m = ATTENTION_PATTERN.search(text)
        if not m:
            return text, 0
        level = int(m.group(1))
        cleaned = ATTENTION_PATTERN.sub("", text).strip()
        return cleaned, level

    async def process_image(self, context_id: str, image_bytes: bytes, user_id: Optional[str] = None) -> dict:
        self.image_history.cleanup()

        image_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex}"
        api_start = time.monotonic()
        messages, image_url = await self._build_messages(context_id, image_bytes, image_id)

        resp = await self.openai_client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=messages,
            temperature=self.temperature,
        )
        raw_text = resp.choices[0].message.content
        api_elapsed = time.monotonic() - api_start
        logger.info(f"[{context_id}] ({api_elapsed:.2f}s) {raw_text}")

        attention_level = 0
        text = raw_text or ""
        if text:
            text, attention_level = self.parse_attention(text)

        if raw_text:
            self.image_history.add(context_id, time.time(), image_url, raw_text)

        if self.result_recorder:
            self.result_recorder.record(context_id, attention_level, text, image_id, user_id)

        result = {"text": text, "attention_level": attention_level}
        if self.include_image_url:
            result["image_url"] = image_url
        return result

    def get_vision_records(self, context_id: str, limit: int = 100, min_attention_level: Optional[int] = None, user_id: Optional[str] = None, since: Optional[datetime] = None, until: Optional[datetime] = None) -> list[dict]:
        return self.result_recorder.get_records(context_id, limit, min_attention_level, user_id, since, until)

    def get_router(self, path: str = "/vision") -> APIRouter:
        router = APIRouter()
        server = self

        @router.post(path)
        async def post_vision(
            context_id: str = Form(...),
            image: UploadFile = File(...),
            user_id: Optional[str] = Form(None),
        ):
            image_bytes = await image.read()
            return await server.process_image(context_id, image_bytes, user_id)

        return router
