import pytest

from aiavatar.sts.tts import SpeechSynthesizer
from aiavatar.sts.tts.postprocessor import TTSPostprocessor
from aiavatar.sts.tts.preprocessor import TTSPreprocessor


class LegacyPreprocessor(TTSPreprocessor):
    def __init__(self):
        self.call_count = 0

    async def process(self, text, style_info=None, language=None):
        self.call_count += 1
        return f"{text}-legacy"


class SynthesizerAwarePreprocessor(TTSPreprocessor):
    def __init__(self):
        self.synthesizer = None

    async def process(
        self,
        text,
        style_info=None,
        language=None,
        *,
        synthesizer=None,
    ):
        self.synthesizer = synthesizer
        return f"{text}-aware"


class RecordingPostprocessor(TTSPostprocessor):
    def __init__(self):
        self.synthesizer = None
        self.call_count = 0

    async def process(self, audio, *, synthesizer):
        self.synthesizer = synthesizer
        self.call_count += 1
        return audio + b"-postprocessed"


class RecordingSynthesizer(SpeechSynthesizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generate_calls = []

    async def make_synthesis_cache_key(
        self,
        text,
        style_info=None,
        language=None,
    ):
        return self.make_cache_key(
            url="test://tts",
            json_body={
                "text": text,
                "style_info": style_info,
                "language": language,
            },
        )

    async def generate(self, text, style_info=None, language=None):
        self.generate_calls.append((text, style_info, language))
        return text.encode()


class GenerateOnlySynthesizer(SpeechSynthesizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generate_calls = []

    async def generate(self, text, style_info=None, language=None):
        self.generate_calls.append((text, style_info, language))
        return text.encode()


@pytest.mark.asyncio
async def test_generate_only_synthesizer_uses_default_cache_key(tmp_path):
    synthesizer = GenerateOnlySynthesizer(cache_dir=str(tmp_path))
    style_info = {"style": "happy"}

    try:
        first = await synthesizer.synthesize(
            "hello",
            style_info=style_info,
            language="en-US",
        )
        second = await synthesizer.synthesize(
            "hello",
            style_info=style_info,
            language="en-US",
        )
        await synthesizer.synthesize(
            "hello",
            style_info=style_info,
            language="ja-JP",
        )
    finally:
        await synthesizer.close()

    assert first == second == b"hello"
    assert synthesizer.generate_calls == [
        ("hello", style_info, "en-US"),
        ("hello", style_info, "ja-JP"),
    ]
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.asyncio
async def test_synthesize_owns_preprocess_generate_and_postprocess_flow():
    legacy = LegacyPreprocessor()
    aware = SynthesizerAwarePreprocessor()
    postprocessor = RecordingPostprocessor()
    synthesizer = RecordingSynthesizer(
        sample_rate=16000,
        preprocessors=[legacy, aware],
        postprocessors=[postprocessor],
    )

    try:
        result = await synthesizer.synthesize(
            "hello",
            style_info={"style": "happy"},
            language="en-US",
        )
    finally:
        await synthesizer.close()

    assert result == b"hello-legacy-aware-postprocessed"
    assert synthesizer.generate_calls == [
        ("hello-legacy-aware", {"style": "happy"}, "en-US")
    ]
    assert legacy.call_count == 1
    assert aware.synthesizer is synthesizer
    assert postprocessor.synthesizer is synthesizer


@pytest.mark.asyncio
async def test_cache_contains_postprocessed_audio(tmp_path):
    legacy = LegacyPreprocessor()
    postprocessor = RecordingPostprocessor()
    synthesizer = RecordingSynthesizer(
        preprocessors=[legacy],
        postprocessors=[postprocessor],
        cache_dir=str(tmp_path),
    )

    try:
        first = await synthesizer.synthesize("hello")
        second = await synthesizer.synthesize("hello")
    finally:
        await synthesizer.close()

    assert first == second == b"hello-legacy-postprocessed"
    assert legacy.call_count == 2
    assert len(synthesizer.generate_calls) == 1
    assert postprocessor.call_count == 1
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.asyncio
async def test_empty_text_skips_entire_synthesis_flow():
    legacy = LegacyPreprocessor()
    postprocessor = RecordingPostprocessor()
    synthesizer = RecordingSynthesizer(
        preprocessors=[legacy],
        postprocessors=[postprocessor],
    )

    try:
        result = await synthesizer.synthesize("   ")
    finally:
        await synthesizer.close()

    assert result == b""
    assert legacy.call_count == 0
    assert synthesizer.generate_calls == []
    assert postprocessor.call_count == 0
