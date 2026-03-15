import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import aiofiles


class CharacterLoader:
    def __init__(
        self,
        source: str = "system_prompt.md",
        *,
        split_initial_messages: bool = False,
        lang: str = "ja",
        user_names: Dict[str, str] = None,
        default_user_name: str = None,
    ):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Character source not found: {source_path}")

        if source_path.is_file():
            if split_initial_messages:
                raise ValueError("split_initial_messages=True requires a directory path")
            self.character_dir: Optional[Path] = None
            self._system_prompt_file: Optional[Path] = source_path
        else:
            self.character_dir = source_path
            self._system_prompt_file = None

        self.split_initial_messages = split_initial_messages
        self.lang = lang
        self.user_names = user_names or {}
        self.default_user_name = default_user_name
        self._get_user_name = self._get_user_name_default
        self._format_messages = None

        # Cache: {filename: (mtime, content)}
        self._file_cache: Dict[str, Tuple[float, str]] = {}
        self._system_prompt: Optional[str] = None
        self._initial_messages: Optional[List[Dict[str, str]]] = None
        self._templates: Optional[dict] = None
        self._templates_mtime: Optional[float] = None

    async def _read_cached(self, path: Path) -> Tuple[str, bool]:
        key = str(path)
        if not path.exists():
            if key in self._file_cache:
                del self._file_cache[key]
                return "", True
            return "", False

        mtime = os.path.getmtime(path)
        cached = self._file_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1], False

        async with aiofiles.open(path, encoding="utf-8") as f:
            content = (await f.read()).strip()
        self._file_cache[key] = (mtime, content)
        return content, True

    async def _read(self, filename: str) -> str:
        path = self.character_dir / filename
        if path.exists():
            async with aiofiles.open(path, encoding="utf-8") as f:
                return (await f.read()).strip()
        return ""

    async def _read_templates(self) -> Tuple[dict, bool]:
        path = self.character_dir / "message_templates.json"
        if not path.exists():
            if self._templates is not None:
                self._templates = None
                self._templates_mtime = None
                return {}, True
            return {}, False

        mtime = os.path.getmtime(path)
        if self._templates is not None and self._templates_mtime == mtime:
            return self._templates, False

        async with aiofiles.open(path, encoding="utf-8") as f:
            self._templates = json.loads(await f.read())
        self._templates_mtime = mtime
        return self._templates, True

    async def read(self, filename: str) -> str:
        if self.character_dir is not None:
            path = self.character_dir / filename
        else:
            path = Path(filename)
        content, _ = await self._read_cached(path)
        return content

    async def get_system_prompt(self) -> str:
        if self._system_prompt_file is not None:
            content, _ = await self._read_cached(self._system_prompt_file)
            return content

        if not self.split_initial_messages:
            sp_path = self.character_dir / "system_prompt.md"
            _, changed = await self._read_cached(sp_path)
            if self._system_prompt is None or changed:
                self._system_prompt = self._file_cache.get(str(sp_path), (0, ""))[1]
        else:
            char_path = self.character_dir / "character.md"
            ri_path = self.character_dir / "response_instructions.md"
            _, char_changed = await self._read_cached(char_path)
            _, ri_changed = await self._read_cached(ri_path)

            if self._system_prompt is None or char_changed or ri_changed:
                content = self._file_cache.get(str(char_path), (0, ""))[1]
                ri = self._file_cache.get(str(ri_path), (0, ""))[1]
                self._system_prompt = content + ("\n" + ri if ri else "")

        return self._system_prompt

    async def get_initial_messages(self) -> List[Dict[str, str]]:
        templates, tmpl_changed = await self._read_templates()
        defs = templates.get("initial_message_defs", {}).get(self.lang, {})
        prefixes = templates.get("prefixes", {}).get(self.lang, {})

        any_changed = tmpl_changed
        keys = [k for k in defs if k != "self_intro"]
        for key in keys:
            _, changed = await self._read_cached(self.character_dir / f"{key}.md")
            if changed:
                any_changed = True

        if self._initial_messages is None or any_changed:
            messages: List[Dict[str, str]] = []
            for key in keys:
                content = self._file_cache.get(str(self.character_dir / f"{key}.md"), (0, ""))[1]
                if not content:
                    continue
                prefix = prefixes.get(key, "")
                messages.append({"role": "user", "content": "$" + prefix + content})
                messages.append({"role": "assistant", "content": defs[key]})
            self._initial_messages = messages

        return self._initial_messages

    # --- Sync wrappers (safe to call with or without a running event loop) ---

    def _run_sync(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Already in an event loop; run in a new thread to avoid conflict
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def read_sync(self, filename: str) -> str:
        return self._run_sync(self.read(filename))

    def get_system_prompt_sync(self) -> str:
        return self._run_sync(self.get_system_prompt())

    def get_initial_messages_sync(self) -> List[Dict[str, str]]:
        return self._run_sync(self.get_initial_messages())

    def _get_user_name_default(self, user_id: str) -> Optional[str]:
        return self.user_names.get(user_id) or self.default_user_name

    def get_user_name(self, func):
        self._get_user_name = func
        return func

    def format_messages(self, func):
        self._format_messages = func
        return func

    def bind(self, llm_service):
        @llm_service.get_system_prompt
        async def _get_system_prompt(context_id, user_id, system_prompt_params):
            # Prompt body is cached; user-specific parts are assembled per call
            prompt = await self.get_system_prompt()
            if not self.split_initial_messages and self.character_dir is not None:
                templates, _ = await self._read_templates()
                tmpl = templates.get("self_intro_template", {}).get(self.lang, "")
                username = self._get_user_name(user_id)
                if username and tmpl:
                    prompt += "\n\n" + tmpl.lstrip("$").format(username=username)
            return prompt

        if self.split_initial_messages:
            @llm_service.get_initial_messages
            async def _get_initial_messages(context_id, user_id, system_prompt_params):
                # Message bodies are cached; user-specific parts are assembled per call
                messages = []
                templates, _ = await self._read_templates()
                defs = templates.get("initial_message_defs", {}).get(self.lang, {})
                tmpl = templates.get("self_intro_template", {}).get(self.lang, "")
                username = self._get_user_name(user_id)
                if username and tmpl:
                    messages.append({"role": "user", "content": tmpl.format(username=username)})
                    if "self_intro" in defs:
                        messages.append({"role": "assistant", "content": defs["self_intro"].format(username=username)})
                messages.extend(await self.get_initial_messages())
                if self._format_messages:
                    messages = self._format_messages(messages)
                return messages
