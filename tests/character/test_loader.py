import os
import pytest
from pathlib import Path
from aiavatar.character.loader import CharacterLoader

CHARACTER_DIR = os.environ["CHARACTER_DIR"]


# --- Init ---

class TestInit:
    def test_dir_mode(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        assert loader.character_dir == Path(CHARACTER_DIR)
        assert loader._system_prompt_file is None

    def test_file_mode(self):
        loader = CharacterLoader(source=os.path.join(CHARACTER_DIR, "system_prompt.md"))
        assert loader.character_dir is None
        assert loader._system_prompt_file is not None

    def test_source_not_found(self):
        with pytest.raises(FileNotFoundError):
            CharacterLoader(source="/nonexistent/path")

    def test_split_requires_dir(self):
        with pytest.raises(ValueError, match="requires a directory"):
            CharacterLoader(
                source=os.path.join(CHARACTER_DIR, "system_prompt.md"),
                split_initial_messages=True,
            )


# --- read (public) ---

class TestRead:
    @pytest.mark.asyncio
    async def test_dir_mode(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        content = await loader.read("character.md")
        assert "くろは" in content

    @pytest.mark.asyncio
    async def test_dir_mode_missing_file(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        assert await loader.read("nonexistent.md") == ""

    @pytest.mark.asyncio
    async def test_file_mode_by_path(self):
        loader = CharacterLoader(source=os.path.join(CHARACTER_DIR, "system_prompt.md"))
        content = await loader.read(os.path.join(CHARACTER_DIR, "character.md"))
        assert "くろは" in content

    @pytest.mark.asyncio
    async def test_file_mode_missing(self):
        loader = CharacterLoader(source=os.path.join(CHARACTER_DIR, "system_prompt.md"))
        assert await loader.read("/nonexistent/file.md") == ""

    @pytest.mark.asyncio
    async def test_uses_cache(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        content1 = await loader.read("character.md")
        content2 = await loader.read("character.md")
        assert content1 == content2
        key = str(Path(CHARACTER_DIR) / "character.md")
        assert key in loader._file_cache

    @pytest.mark.asyncio
    async def test_file_mode_uses_cache(self):
        path = os.path.join(CHARACTER_DIR, "character.md")
        loader = CharacterLoader(source=os.path.join(CHARACTER_DIR, "system_prompt.md"))
        content1 = await loader.read(path)
        content2 = await loader.read(path)
        assert content1 == content2
        assert str(Path(path)) in loader._file_cache


# --- get_system_prompt ---

class TestGetSystemPrompt:
    @pytest.mark.asyncio
    async def test_file_mode(self):
        loader = CharacterLoader(source=os.path.join(CHARACTER_DIR, "system_prompt.md"))
        prompt = await loader.get_system_prompt()
        assert "くろは" in prompt

    @pytest.mark.asyncio
    async def test_file_mode_cache(self):
        loader = CharacterLoader(source=os.path.join(CHARACTER_DIR, "system_prompt.md"))
        p1 = await loader.get_system_prompt()
        p2 = await loader.get_system_prompt()
        assert p1 == p2

    @pytest.mark.asyncio
    async def test_dir_mode_no_split(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        prompt = await loader.get_system_prompt()
        assert "くろは" in prompt

    @pytest.mark.asyncio
    async def test_dir_mode_split(self):
        loader = CharacterLoader(source=CHARACTER_DIR, split_initial_messages=True)
        prompt = await loader.get_system_prompt()
        assert "くろは" in prompt  # character.md content
        assert "思考" in prompt  # response_instructions.md content

    @pytest.mark.asyncio
    async def test_dir_mode_split_cache(self):
        loader = CharacterLoader(source=CHARACTER_DIR, split_initial_messages=True)
        p1 = await loader.get_system_prompt()
        p2 = await loader.get_system_prompt()
        assert p1 == p2


# --- get_initial_messages ---

class TestGetInitialMessages:
    @pytest.mark.asyncio
    async def test_basic(self):
        loader = CharacterLoader(source=CHARACTER_DIR, split_initial_messages=True)
        messages = await loader.get_initial_messages()
        assert len(messages) > 0
        assert all(m["role"] in ("user", "assistant") for m in messages)
        # Should contain content from md files (episode, attribute, etc.)
        user_contents = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "マスター" in user_contents

    @pytest.mark.asyncio
    async def test_excludes_self_intro(self):
        loader = CharacterLoader(source=CHARACTER_DIR, split_initial_messages=True)
        messages = await loader.get_initial_messages()
        # self_intro is handled separately, should not appear here
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        for c in user_contents:
            assert "{username}" not in c

    @pytest.mark.asyncio
    async def test_cache(self):
        loader = CharacterLoader(source=CHARACTER_DIR, split_initial_messages=True)
        msgs1 = await loader.get_initial_messages()
        msgs2 = await loader.get_initial_messages()
        assert msgs1 is msgs2  # Same object = cached

    @pytest.mark.asyncio
    async def test_no_templates(self):
        # dir mode without message_templates.json -> empty
        loader = CharacterLoader(source=CHARACTER_DIR)
        loader.character_dir = Path(CHARACTER_DIR).parent  # parent has no templates
        assert await loader.get_initial_messages() == []


# --- _read_cached ---

class TestReadCached:
    @pytest.mark.asyncio
    async def test_returns_content_and_changed(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        path = Path(CHARACTER_DIR) / "character.md"
        content, changed = await loader._read_cached(path)
        assert "くろは" in content
        assert changed is True

    @pytest.mark.asyncio
    async def test_second_call_not_changed(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        path = Path(CHARACTER_DIR) / "character.md"
        await loader._read_cached(path)
        content, changed = await loader._read_cached(path)
        assert "くろは" in content
        assert changed is False

    @pytest.mark.asyncio
    async def test_missing_file(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        path = Path(CHARACTER_DIR) / "nonexistent.md"
        content, changed = await loader._read_cached(path)
        assert content == ""
        assert changed is False

    @pytest.mark.asyncio
    async def test_absolute_path(self):
        loader = CharacterLoader(source=CHARACTER_DIR)
        path = Path(CHARACTER_DIR) / "episode.md"
        content, changed = await loader._read_cached(path)
        assert "マスター" in content
        assert changed is True


# --- User name ---

class TestUserName:
    def test_default_user_name(self):
        loader = CharacterLoader(source=CHARACTER_DIR, default_user_name="太郎")
        assert loader._get_user_name("unknown_user") == "太郎"

    def test_user_names_dict(self):
        loader = CharacterLoader(source=CHARACTER_DIR, user_names={"u1": "花子"})
        assert loader._get_user_name("u1") == "花子"
        assert loader._get_user_name("u2") is None

    def test_custom_get_user_name(self):
        loader = CharacterLoader(source=CHARACTER_DIR)

        @loader.get_user_name
        def custom(user_id):
            return f"User-{user_id}"

        assert loader._get_user_name("abc") == "User-abc"
