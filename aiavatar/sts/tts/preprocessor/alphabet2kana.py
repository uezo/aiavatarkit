import re
import logging
import httpx
from . import TTSPreprocessor


class AlphabetToKanaPreprocessor(TTSPreprocessor):
    def __init__(
        self,
        *,
        openai_api_key: str,
        model: str = "gpt-4.1-mini",
        system_prompt: str = None,
        alphabet_length: int = 3,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
    ):
        self.openai_api_key = openai_api_key
        self.model = model
        self.system_prompt = system_prompt or "与えられた文字列に含まれる外国語（アルファベット、中国語、ハングルなど）をカタカナ読みに変換してください。変換後の文字列で置換した文章全体を<converted>~</converted>に出力してください。"
        self._alphabet_length = alphabet_length
        self.alphabet_pattern = re.compile(rf"[A-Za-z]{{{self._alphabet_length},}}")
        self.converted_pattern = re.compile(r"<converted>(.*?)</converted>", re.DOTALL)
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )

    @property
    def alphabet_length(self):
        return self._alphabet_length
    
    @alphabet_length.setter
    def alphabet_length(self, value: int):
        self._alphabet_length = value
        self.alphabet_pattern = re.compile(rf"[A-Za-z]{{{self._alphabet_length},}}")

    async def process(self, text: str, style_info: dict = None, language: str = None) -> str:
        if self.alphabet_pattern.search(text):
            data = {
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": text},
                ],
            }
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}"
            }

            try:
                resp = await self.http_client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=data,
                    headers=headers,
                    timeout=5.0,
                )
                resp.raise_for_status()
                resp_json = resp.json()

                converted = resp_json["choices"][0]["message"]["content"]
                logging.debug(f"Convert: {text} -> {converted}")

                m = self.converted_pattern.search(converted)
                return m.group(1) if m else text

            except Exception as ex:
                logging.error(f"Error at convert_alphabet_to_kana (text={text}): {ex}", exc_info=True)
                return text

        else:
            return text
