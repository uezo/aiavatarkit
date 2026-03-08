import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional


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

    def _read_cached(self, filename: str) -> Tuple[str, bool]:
        path = self.character_dir / filename
        if not path.exists():
            if filename in self._file_cache:
                del self._file_cache[filename]
                return "", True
            return "", False

        mtime = os.path.getmtime(path)
        cached = self._file_cache.get(filename)
        if cached and cached[0] == mtime:
            return cached[1], False

        content = path.read_text(encoding="utf-8").strip()
        self._file_cache[filename] = (mtime, content)
        return content, True

    def _read(self, filename: str) -> str:
        path = self.character_dir / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    def _read_templates(self) -> Tuple[dict, bool]:
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

        self._templates = json.loads(path.read_text(encoding="utf-8"))
        self._templates_mtime = mtime
        return self._templates, True

    def get_system_prompt(self) -> str:
        if self._system_prompt_file is not None:
            path = self._system_prompt_file
            mtime = os.path.getmtime(path)
            cached = self._file_cache.get(str(path))
            if cached and cached[0] == mtime:
                return cached[1]
            content = path.read_text(encoding="utf-8").strip()
            self._file_cache[str(path)] = (mtime, content)
            return content

        if not self.split_initial_messages:
            _, changed = self._read_cached("system_prompt.md")
            if self._system_prompt is None or changed:
                self._system_prompt = self._file_cache.get("system_prompt.md", (0, ""))[1]
        else:
            _, char_changed = self._read_cached("character.md")
            _, ri_changed = self._read_cached("response_instructions.md")

            if self._system_prompt is None or char_changed or ri_changed:
                content = self._file_cache.get("character.md", (0, ""))[1]
                ri = self._file_cache.get("response_instructions.md", (0, ""))[1]
                self._system_prompt = content + ("\n" + ri if ri else "")

        return self._system_prompt

    def get_initial_messages(self) -> List[Dict[str, str]]:
        templates, tmpl_changed = self._read_templates()
        defs = templates.get("initial_message_defs", {}).get(self.lang, {})
        prefixes = templates.get("prefixes", {}).get(self.lang, {})

        any_changed = tmpl_changed
        keys = [k for k in defs if k != "self_intro"]
        for key in keys:
            _, changed = self._read_cached(f"{key}.md")
            if changed:
                any_changed = True

        if self._initial_messages is None or any_changed:
            messages: List[Dict[str, str]] = []
            for key in keys:
                content = self._file_cache.get(f"{key}.md", (0, ""))[1]
                if not content:
                    continue
                prefix = prefixes.get(key, "")
                messages.append({"role": "user", "content": "$" + prefix + content})
                messages.append({"role": "assistant", "content": defs[key]})
            self._initial_messages = messages

        return self._initial_messages

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
            prompt = self.get_system_prompt()
            if not self.split_initial_messages and self.character_dir is not None:
                templates, _ = self._read_templates()
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
                templates, _ = self._read_templates()
                defs = templates.get("initial_message_defs", {}).get(self.lang, {})
                tmpl = templates.get("self_intro_template", {}).get(self.lang, "")
                username = self._get_user_name(user_id)
                if username and tmpl:
                    messages.append({"role": "user", "content": tmpl.format(username=username)})
                    if "self_intro" in defs:
                        messages.append({"role": "assistant", "content": defs["self_intro"].format(username=username)})
                messages.extend(self.get_initial_messages())
                if self._format_messages:
                    messages = self._format_messages(messages)
                return messages
