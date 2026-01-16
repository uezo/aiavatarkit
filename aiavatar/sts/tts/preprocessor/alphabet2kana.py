import json
import logging
import re
import time
import httpx
from . import TTSPreprocessor

logger = logging.getLogger(__name__)


class AlphabetToKanaPreprocessor(TTSPreprocessor):
    CONVERT_TOOL = {
        "type": "function",
        "function": {
            "name": "convert_alphabet_to_kana",
            "description": "Output the result of converting alphabet to katakana reading",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversions": {
                        "type": "array",
                        "description": "List of pairs of words to convert and their readings",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {
                                    "type": "string",
                                    "description": "Original alphabet notation before conversion"
                                },
                                "kana": {
                                    "type": "string",
                                    "description": "Katakana reading"
                                }
                            },
                            "required": ["original", "kana"]
                        }
                    }
                },
                "required": ["conversions"]
            }
        }
    }

    def __init__(
        self,
        *,
        openai_api_key: str,
        model: str = "gpt-4.1-mini",
        base_url: str = None,
        reasoning_effort: str = None,
        extra_body: dict = None,
        system_prompt: str = None,
        system_prompt_with_cache: str = None,
        alphabet_length: int = 3,
        special_chars: str = ".'-'−–",
        use_kana_map: bool = True,
        kana_map: dict[str, str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        self.openai_api_key = openai_api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body
        self.system_prompt = system_prompt or "与えられた文字列に含まれる外国語（アルファベット、中国語、ハングルなど）をカタカナ読みに変換してください。変換後の文字列で置換した文章全体を<converted>~</converted>に出力してください。"
        self.system_prompt_with_cache = system_prompt_with_cache or "与えられたアルファベット文字列をカタカナ読みに変換してください。convert_alphabet_to_kana関数で出力すること。originalにはアポストロフィーやピリオド等の記号も省略せずに出力すること。"
        self._alphabet_length = alphabet_length
        self.special_chars = special_chars
        self._build_alphabet_pattern()
        self.converted_pattern = re.compile(r"<converted>(.*?)</converted>", re.DOTALL)
        self.kana_map: dict[str, str] = kana_map.copy() if kana_map else {}
        self.use_kana_map = use_kana_map
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )
        self.debug = debug

    def _build_alphabet_pattern(self):
        escaped_chars = re.escape(self.special_chars)
        self.alphabet_pattern = re.compile(rf"[A-Za-z]+(?:[{escaped_chars}][A-Za-z]+)*")

    def _should_process(self, match: str) -> bool:
        for c in self.special_chars:
            if c in match:
                return True
        return len(match) >= self._alphabet_length

    @property
    def alphabet_length(self):
        return self._alphabet_length

    @alphabet_length.setter
    def alphabet_length(self, value: int):
        self._alphabet_length = value

    async def process(self, text: str, style_info: dict = None, language: str = None) -> str:
        if self.use_kana_map:
            return await self.process_with_kana_map(text=text, style_info=style_info, language=language)
        else:
            return await self.process_direct(text=text, style_info=style_info, language=language)

    async def process_direct(self, text: str, style_info: dict = None, language: str = None) -> str:
        if self.alphabet_pattern.search(text):
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": text},
                ],
            }
            if self.reasoning_effort:
                data["reasoning_effort"] = self.reasoning_effort
            if self.extra_body:
                data["extra_body"] = self.extra_body

            headers = {
                "Authorization": f"Bearer {self.openai_api_key}"
            }

            try:
                start_time = time.time()
                resp = await self.http_client.post(
                    f"{self.base_url}/chat/completions",
                    json=data,
                    headers=headers,
                    timeout=5.0,
                )
                resp.raise_for_status()
                resp_json = resp.json()
                elapsed = time.time() - start_time

                converted = resp_json["choices"][0]["message"]["content"]
                if self.debug:
                    logger.info(f"{self.__class__.__name__} [LLM]: {text} -> {converted} ({elapsed:.3f}sec)")

                m = self.converted_pattern.search(converted)
                return m.group(1) if m else text

            except Exception as ex:
                logger.error(f"Error at process_direct (text={text}): {ex}", exc_info=True)
                return text

        else:
            return text

    async def process_with_kana_map(self, text: str, style_info: dict = None, language: str = None) -> str:
        for original, kana in self.kana_map.items():
            new_text = re.sub(re.escape(original), kana, text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                if self.debug:
                    logger.info(f"{self.__class__.__name__} [KanaMap]: {original} -> {kana}")

        kana_map_lower = {k.lower() for k in self.kana_map}
        uncached = [
            m for m in self.alphabet_pattern.findall(text)
            if self._should_process(m) and m.lower() not in kana_map_lower
        ]
        if not uncached:
            return text

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt_with_cache},
                {"role": "user", "content": "\n".join(uncached)},
            ],
            "tools": [self.CONVERT_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "convert_alphabet_to_kana"}},
        }
        if self.reasoning_effort:
            data["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            data["extra_body"] = self.extra_body

        if "azure" in self.base_url:
            headers = {"api-key": self.openai_api_key}

        else:
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}

        try:
            start_time = time.time()
            resp = await self.http_client.post(
                f"{self.base_url}/chat/completions",
                json=data,
                headers=headers,
                timeout=5.0,
            )
            resp.raise_for_status()
            resp_json = resp.json()
            elapsed = time.time() - start_time

            tool_calls = resp_json["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                arguments = json.loads(tool_calls[0]["function"]["arguments"])
                conversions = arguments.get("conversions", [])

                for conv in conversions:
                    original = conv.get("original", "")
                    kana = conv.get("kana", "")
                    if original and kana:
                        self.kana_map[original] = kana
                        text = re.sub(re.escape(original), kana, text, flags=re.IGNORECASE)
                        if self.debug:
                            logger.info(f"{self.__class__.__name__} [LLM]: {original} -> {kana}")

            if self.debug:
                logger.info(f"{self.__class__.__name__} [LLM] completed ({elapsed:.3f}sec)")

            return text

        except Exception as ex:
            logger.error(f"Error at process_with_cache (text={text}): {ex}", exc_info=True)
            return text
