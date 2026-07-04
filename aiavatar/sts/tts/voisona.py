import asyncio
import logging
import os
import tempfile
import uuid as uuid_mod
from typing import Dict, List, Optional

import aiofiles

from . import SpeechSynthesizer
from .preprocessor import TTSPreprocessor

logger = logging.getLogger(__name__)


class VoisonaSpeechSynthesizer(SpeechSynthesizer):
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:32766/api/talk/v1",
        username: str = None,
        password: str = None,
        speaker: str = None,
        voice_version: str = None,
        default_language: str = "ja_JP",
        global_parameters: dict = None,
        output_dir: str = None,
        poll_interval: float = 0.05,
        delete_request: bool = True,
        style_mapper: Dict[str, str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        preprocessors: List[TTSPreprocessor] = None,
        cache_dir: str = None,
        cache_ext: str = "wav",
        debug: bool = False
    ):
        super().__init__(
            style_mapper=style_mapper,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            preprocessors=preprocessors,
            cache_dir=cache_dir,
            cache_ext=cache_ext,
            debug=debug
        )
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.speaker = speaker
        self.voice_version = voice_version
        self.default_language = default_language
        self.global_parameters = global_parameters
        self.output_dir = output_dir or self._get_default_output_dir()
        self.poll_interval = poll_interval
        self.delete_request = delete_request
        self._voice_libraries = None
        self._voice_libraries_lock = asyncio.Lock()

    def get_config(self) -> dict:
        config = super().get_config()
        config["base_url"] = self.base_url
        config["speaker"] = self.speaker
        config["voice_version"] = self.voice_version
        config["default_language"] = self.default_language
        config["global_parameters"] = self.global_parameters
        config["output_dir"] = self.output_dir
        config["poll_interval"] = self.poll_interval
        config["delete_request"] = self.delete_request
        return config

    @staticmethod
    def _get_default_output_dir() -> str:
        shm_dir = "/dev/shm"
        if os.path.isdir(shm_dir) and os.access(shm_dir, os.W_OK):
            return shm_dir
        return tempfile.gettempdir()

    @property
    def _auth(self):
        if self.username is None and self.password is None:
            return None
        return (self.username or "", self.password or "")

    async def get_voice_libraries(self, refresh: bool = False) -> list:
        if self._voice_libraries is not None and not refresh:
            return self._voice_libraries

        async with self._voice_libraries_lock:
            if self._voice_libraries is not None and not refresh:
                return self._voice_libraries

            response = await self.http_client.get(f"{self.base_url}/voices", auth=self._auth)
            response.raise_for_status()
            self._voice_libraries = response.json().get("items", [])
            return self._voice_libraries

    def _find_voice_library(self, voice_libraries: list, speaker: Optional[str]) -> Optional[dict]:
        if not voice_libraries:
            return None

        if not speaker:
            if self.voice_version:
                for voice_library in voice_libraries:
                    if voice_library.get("voice_version") == self.voice_version:
                        return voice_library
            return voice_libraries[0]

        for voice_library in voice_libraries:
            if voice_library.get("voice_name") != speaker:
                continue
            if self.voice_version and voice_library.get("voice_version") != self.voice_version:
                continue
            return voice_library

        return None

    async def get_voice_library(self, speaker: Optional[str] = None) -> dict:
        voice_libraries = await self.get_voice_libraries()
        voice_library = self._find_voice_library(voice_libraries, speaker)
        if voice_library:
            return voice_library

        voice_libraries = await self.get_voice_libraries(refresh=True)
        voice_library = self._find_voice_library(voice_libraries, speaker)
        if voice_library:
            return voice_library

        if speaker:
            raise ValueError(f"Voisona voice library not found: {speaker}")
        raise ValueError("Voisona voice library not found")

    async def _wait_for_synthesis(self, request_uuid: str) -> dict:
        start = asyncio.get_running_loop().time()
        timeout = self.http_client.timeout.read
        while True:
            response = await self.http_client.get(
                f"{self.base_url}/speech-syntheses/{request_uuid}",
                auth=self._auth
            )
            response.raise_for_status()
            response_json = response.json()
            state = response_json.get("state")
            if state == "succeeded":
                return response_json
            if state in ("failed", "canceled"):
                raise RuntimeError(f"Voisona speech synthesis {state}: {response_json}")
            if timeout is not None and asyncio.get_running_loop().time() - start > timeout:
                raise TimeoutError("Voisona speech synthesis timed out")
            await asyncio.sleep(self.poll_interval)

    async def _delete_synthesis_request(self, request_uuid: str):
        response = await self.http_client.delete(
            f"{self.base_url}/speech-syntheses/{request_uuid}",
            auth=self._auth
        )
        response.raise_for_status()

    def _make_output_path(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"aiavatar-voisona-{uuid_mod.uuid4().hex}.wav")

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        if not text or not text.strip():
            return bytes()

        if self.debug:
            logger.info(f"Speech synthesize: {text}")

        processed_text = await self.preprocess(text, style_info, language)

        speaker = self.speaker
        if style := self.parse_style(style_info):
            speaker = style
            logger.info(f"Apply style: {speaker}")

        voice_library = await self.get_voice_library(speaker)
        synthesis_language = language or self.default_language
        if synthesis_language not in voice_library.get("languages", []):
            synthesis_language = (voice_library.get("languages") or [self.default_language])[0]

        url = f"{self.base_url}/speech-syntheses"
        cache_payload = {
            "text": processed_text,
            "language": synthesis_language,
            "voice_name": voice_library["voice_name"],
            "voice_version": voice_library["voice_version"],
            "global_parameters": self.global_parameters,
        }
        cache_key = self.make_cache_key(
            url=url,
            json_body={k: v for k, v in cache_payload.items() if v is not None}
        )
        if cached := await self.read_cache(cache_key):
            return cached

        output_file_path = self._make_output_path()
        request_uuid = None
        try:
            payload = {
                **{k: v for k, v in cache_payload.items() if v is not None},
                "can_overwrite_file": True,
                "destination": "file",
                "output_file_path": output_file_path,
            }
            response = await self.http_client.post(url, auth=self._auth, json=payload)
            response.raise_for_status()
            request_uuid = response.json()["uuid"]

            await self._wait_for_synthesis(request_uuid)

            async with aiofiles.open(output_file_path, "rb") as f:
                audio = await f.read()

            await self.write_cache(cache_key, audio)
            return audio
        finally:
            if request_uuid and self.delete_request:
                try:
                    await self._delete_synthesis_request(request_uuid)
                except Exception:
                    logger.exception("Failed to delete Voisona speech synthesis request")
            try:
                os.remove(output_file_path)
            except FileNotFoundError:
                pass
