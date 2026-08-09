import pytest

from aiavatar.sts.tts import SpeechSynthesizer, SpeechSynthesizerRouter


class RecordingSynthesizer(SpeechSynthesizer):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.generate_calls = []
        self.close_calls = 0

    async def generate(self, text, style_info=None, language=None):
        self.generate_calls.append((text, style_info, language))
        return f"{self.name}:{text}".encode()

    async def close(self):
        self.close_calls += 1
        await super().close()


@pytest.mark.asyncio
async def test_routes_and_forwards_synthesis_arguments():
    japanese = RecordingSynthesizer("ja")
    english = RecordingSynthesizer("en")
    router = SpeechSynthesizerRouter({
        "japanese": japanese,
        "english": english,
    })
    route_calls = []

    @router.route
    def select_synthesizer(text, style_info=None, language=None):
        route_calls.append((text, style_info, language))
        return "english" if language == "en-US" else "japanese"

    style_info = {"styled_text": "Hello!"}
    try:
        result = await router.synthesize(
            "Hello",
            style_info=style_info,
            language="en-US",
        )
    finally:
        await router.close()

    assert result == b"en:Hello"
    assert route_calls == [("Hello", style_info, "en-US")]
    assert japanese.generate_calls == []
    assert english.generate_calls == [("Hello", style_info, "en-US")]


@pytest.mark.asyncio
async def test_uses_default_when_route_returns_none():
    japanese = RecordingSynthesizer("ja")
    router = SpeechSynthesizerRouter(
        {"japanese": japanese},
        default="japanese",
    )

    @router.route
    def select_synthesizer(text, style_info=None, language=None):
        return None

    try:
        result = await router.synthesize("こんにちは")
    finally:
        await router.close()

    assert result == "ja:こんにちは".encode()
    assert japanese.generate_calls == [("こんにちは", None, None)]


@pytest.mark.asyncio
async def test_uses_registered_synthesizer_cache(tmp_path):
    cache_dir = tmp_path / "actual-tts-cache"
    japanese = RecordingSynthesizer("ja", cache_dir=str(cache_dir))
    router = SpeechSynthesizerRouter({"japanese": japanese})
    route_calls = 0

    @router.route
    def select_synthesizer(text, style_info=None, language=None):
        nonlocal route_calls
        route_calls += 1
        return "japanese"

    try:
        first = await router.synthesize("cache me", language="ja-JP")
        second = await router.synthesize("cache me", language="ja-JP")
    finally:
        await router.close()

    assert first == second == b"ja:cache me"
    assert route_calls == 2
    assert japanese.generate_calls == [("cache me", None, "ja-JP")]
    assert len(list(cache_dir.iterdir())) == 1
    assert router.cache_dir is None


@pytest.mark.asyncio
async def test_requires_route_function():
    router = SpeechSynthesizerRouter({"japanese": RecordingSynthesizer("ja")})

    try:
        with pytest.raises(
            RuntimeError,
            match="TTS route function is not configured",
        ):
            await router.synthesize("こんにちは")
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_rejects_unknown_route():
    router = SpeechSynthesizerRouter({"japanese": RecordingSynthesizer("ja")})

    @router.route
    def select_synthesizer(text, style_info=None, language=None):
        return "missing"

    try:
        with pytest.raises(ValueError, match="Unknown TTS route: missing"):
            await router.synthesize("hello")
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_rejects_none_without_default():
    router = SpeechSynthesizerRouter({"japanese": RecordingSynthesizer("ja")})

    @router.route
    def select_synthesizer(text, style_info=None, language=None):
        return None

    try:
        with pytest.raises(
            ValueError,
            match="TTS route returned None and no default is configured",
        ):
            await router.synthesize("hello")
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_close_closes_shared_synthesizer_once():
    shared = RecordingSynthesizer("shared")
    router = SpeechSynthesizerRouter({
        "japanese": shared,
        "english": shared,
    })

    await router.close()

    assert shared.close_calls == 1


@pytest.mark.asyncio
async def test_rejects_unknown_default():
    synthesizer = RecordingSynthesizer("ja")

    try:
        with pytest.raises(ValueError, match="Unknown default TTS route: english"):
            SpeechSynthesizerRouter(
                {"japanese": synthesizer},
                default="english",
            )
    finally:
        await synthesizer.close()


@pytest.mark.asyncio
async def test_rejects_async_route_function():
    router = SpeechSynthesizerRouter({"japanese": RecordingSynthesizer("ja")})

    async def select_synthesizer(text, style_info=None, language=None):
        return "japanese"

    try:
        with pytest.raises(TypeError, match="route must be a synchronous function"):
            router.route(select_synthesizer)
    finally:
        await router.close()
