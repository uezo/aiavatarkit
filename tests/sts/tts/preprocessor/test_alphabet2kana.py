import os
import pytest
from aiavatar.sts.tts.preprocessor.alphabet2kana import AlphabetToKanaPreprocessor

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Alphabet pattern tests

def test_alphabet_pattern_simple():
    """Test simple alphabet word detection."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        alphabet_length=3
    )
    # Should match
    assert preproc.alphabet_pattern.search("Hello World")
    assert preproc.alphabet_pattern.search("これはTestです")
    # Should match (pattern matches, length check is separate)
    assert preproc.alphabet_pattern.search("AB") is not None
    # Should not match
    assert preproc.alphabet_pattern.search("こんにちは") is None
    assert preproc.alphabet_pattern.search("12345") is None


def test_alphabet_pattern_with_special_chars():
    """Test alphabet pattern with special characters like period and apostrophe."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        special_chars=".'-'−–"
    )
    # Should match as single word
    matches = preproc.alphabet_pattern.findall("Mr. Smith said You're right")
    assert "Mr." in matches or "Mr" in matches
    assert any("You're" in m or "You" in m for m in matches)


def test_alphabet_pattern_with_hyphen():
    """Test alphabet pattern with hyphen."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        special_chars=".'-'−–"
    )
    matches = preproc.alphabet_pattern.findall("Wi-Fi is available")
    assert "Wi-Fi" in matches


# _should_process tests

def test_should_process_by_length():
    """Test that words are processed based on alphabet_length."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        alphabet_length=3
    )
    # Length >= 3 should be processed
    assert preproc._should_process("Hello") is True
    assert preproc._should_process("abc") is True
    # Length < 3 should not be processed
    assert preproc._should_process("AB") is False
    assert preproc._should_process("a") is False


def test_should_process_by_special_chars():
    """Test that words with special chars are always processed."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        alphabet_length=3,
        special_chars=".'-'−–"
    )
    # Words with special chars should be processed regardless of length
    assert preproc._should_process("Mr.") is True
    assert preproc._should_process("I'm") is True
    assert preproc._should_process("Wi-Fi") is True
    # Short words without special chars should not be processed
    assert preproc._should_process("AB") is False


# kana_map application tests

@pytest.mark.asyncio
async def test_kana_map_applied():
    """Test that kana_map mappings are applied."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        kana_map={"Hello": "ハロー", "World": "ワールド"},
        debug=True
    )
    result = await preproc.process("Hello World")
    assert "ハロー" in result
    assert "ワールド" in result


@pytest.mark.asyncio
async def test_kana_map_case_insensitive():
    """Test that kana_map matching is case-insensitive."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        kana_map={"hello": "ハロー"},
        debug=True
    )
    # Should match regardless of case
    result_lower = await preproc.process("hello world")
    result_upper = await preproc.process("HELLO world")
    result_mixed = await preproc.process("Hello world")
    assert "ハロー" in result_lower
    assert "ハロー" in result_upper
    assert "ハロー" in result_mixed


@pytest.mark.asyncio
async def test_kana_map_not_modified():
    """Test that initial kana_map is copied and not modified externally."""
    initial_map = {"Hello": "ハロー"}
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        kana_map=initial_map,
        debug=True
    )
    # Modify original map
    initial_map["World"] = "ワールド"
    # Preprocessor's map should not be affected
    assert "World" not in preproc.kana_map


# process_with_kana_map tests (with actual OpenAI API)

@pytest.mark.asyncio
async def test_process_simple_word():
    """Test processing a simple alphabet word."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        debug=True
    )
    result = await preproc.process("こんにちは、Pythonを学んでいます。")
    # Should convert Python to katakana
    assert "Python" not in result
    assert "パイソン" in result

    # Check kana_map is updated
    assert "Python" in preproc.kana_map

    await preproc.http_client.aclose()


@pytest.mark.asyncio
async def test_process_multiple_words():
    """Test processing multiple alphabet words."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        debug=True
    )
    result = await preproc.process("AppleとGoogleは有名な会社です。")
    # Should convert both words
    assert "Apple" not in result
    assert "Google" not in result
    # Check conversions are reasonable (katakana)
    assert "アップル" in result or "アプル" in result
    assert "グーグル" in result

    await preproc.http_client.aclose()


@pytest.mark.asyncio
async def test_process_with_special_chars():
    """Test processing words with special characters."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        special_chars=".'-'−–",
        debug=True
    )
    result = await preproc.process("Wi-Fiに接続してください。")
    # Should convert Wi-Fi
    assert "Wi-Fi" not in result
    # Check conversion is reasonable
    assert "ワイファイ" in result

    await preproc.http_client.aclose()


@pytest.mark.asyncio
async def test_process_cached_word_not_called_again():
    """Test that cached words are not sent to LLM again."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        kana_map={"Python": "パイソン"},
        debug=True
    )
    # Process text with cached word
    result = await preproc.process("Pythonは楽しい。")
    # Should use cached value
    assert "パイソン" in result
    # kana_map should not have new entries (only Python was in text)
    assert len(preproc.kana_map) == 1

    await preproc.http_client.aclose()


@pytest.mark.asyncio
async def test_process_no_alphabet():
    """Test processing text without alphabet."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        debug=True
    )
    input_text = "こんにちは、今日はいい天気ですね。"
    result = await preproc.process(input_text)
    # Should return unchanged
    assert result == input_text

    await preproc.http_client.aclose()


@pytest.mark.asyncio
async def test_process_short_alphabet_ignored():
    """Test that short alphabet words are ignored."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        alphabet_length=3,
        debug=True
    )
    input_text = "私はAIが好きです。"  # AI is only 2 chars
    result = await preproc.process(input_text)
    # Should return unchanged (AI is too short)
    assert result == input_text

    await preproc.http_client.aclose()


# process_direct tests (with actual OpenAI API)

@pytest.mark.asyncio
async def test_process_direct_simple():
    """Test process_direct method."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        use_kana_map=False,
        debug=True
    )
    result = await preproc.process("Pythonを学んでいます。")
    # Should convert Python to katakana
    assert "Python" not in result
    assert "パイソン" in result

    await preproc.http_client.aclose()


@pytest.mark.asyncio
async def test_process_direct_no_alphabet():
    """Test process_direct with no alphabet."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        use_kana_map=False,
        debug=True
    )
    input_text = "こんにちは"
    result = await preproc.process(input_text)
    # Should return unchanged
    assert result == input_text

    await preproc.http_client.aclose()


# alphabet_length property tests

def test_alphabet_length_getter():
    """Test alphabet_length getter."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        alphabet_length=5
    )
    assert preproc.alphabet_length == 5


def test_alphabet_length_setter():
    """Test alphabet_length setter."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        alphabet_length=3
    )
    preproc.alphabet_length = 10
    assert preproc.alphabet_length == 10
    assert preproc._alphabet_length == 10


# kana_map persistence tests

@pytest.mark.asyncio
async def test_kana_map_grows_with_usage():
    """Test that kana_map grows as new words are processed."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        debug=True
    )
    initial_size = len(preproc.kana_map)
    assert initial_size == 0

    # Process text with new word
    await preproc.process("Anthropicは素晴らしい会社です。")
    # kana_map should have new entry
    assert len(preproc.kana_map) > initial_size
    assert "Anthropic" in preproc.kana_map

    await preproc.http_client.aclose()


@pytest.mark.asyncio
async def test_kana_map_can_be_exported():
    """Test that kana_map can be exported as dict."""
    preproc = AlphabetToKanaPreprocessor(
        openai_api_key=OPENAI_API_KEY,
        kana_map={"Hello": "ハロー", "World": "ワールド"},
        debug=True
    )
    # Export kana_map
    exported = preproc.kana_map.copy()
    assert isinstance(exported, dict)
    assert exported == {"Hello": "ハロー", "World": "ワールド"}
