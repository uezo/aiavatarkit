# TTS preprocessing

Preprocessors rewrite text on its way to the synthesizer. They exist mostly to fix
pronunciation: Japanese engines read `AIAvatarKit` letter by letter, product names come out
wrong, and numbers need units attached. A preprocessor turns the written form into
something the engine says correctly, without changing what is stored in conversation
history.

Attach them with the `preprocessors` argument, which every synthesizer accepts.

AIAvatarKit provides text preprocessing functionality that transforms text before Text-to-Speech processing. This enables improved speech quality and conversion of specific text patterns.

## Alphabet to Katakana Conversion

A preprocessor that converts alphabet text to katakana using LLM. Supports kana_map for storing word-to-reading mappings to reduce latency on repeated words.

```python
from aiavatar.sts.tts.preprocessor.alphabet2kana import AlphabetToKanaPreprocessor

# Create preprocessor with kana_map for pre-registered word-reading mappings
alphabet2kana_preproc = AlphabetToKanaPreprocessor(
    openai_api_key=OPENAI_API_KEY,
    alphabet_length=3,                        # Minimum alphabet length to convert (default: 3)
    special_chars=".'-'−–",                   # Characters that connect words (default: ".'-'−–")
    use_kana_map=True,                        # Enable kana_map mode (default: True)
    kana_map={"GitHub": "ギットハブ"},         # Pre-registered word-reading mappings (optional)
    debug=True,                               # Enable debug logging (default: False)
)

# Add to TTS
tts.preprocessors.append(alphabet2kana_preproc)

# Words converted by LLM are automatically added to kana_map
# You can persist and restore kana_map for future sessions:
import json
# Save
with open("kana_map.json", "w") as f:
    json.dump(alphabet2kana_preproc.kana_map, f, ensure_ascii=False)
# Load
with open("kana_map.json") as f:
    kana_map = json.load(f)
```

Preprocessors may optionally accept a keyword-only `synthesizer` argument to access shared TTS configuration such as `sample_rate`. Existing preprocessors with the original `process(text, style_info, language)` signature remain supported.

Key features:
- **kana_map**: Pre-register known word-reading mappings and automatically add LLM results to avoid repeated API calls
- **special_chars**: Words containing these characters (e.g., `Mr.`, `You're`, `Wi-Fi`) are always processed regardless of `alphabet_length`
- **Case-insensitive**: Matches `API`, `api`, and `Api` with a single kana_map entry
- **debug mode**: Logs `[KanaMap]` for cached hits and `[LLM]` for new readings with elapsed time

## Pattern Match Conversion

You can also use regular expressions and string patterns for conversion:

```python
import re
from aiavatar.sts.tts.preprocessor.patternmatch import PatternMatchPreprocessor

# Create pattern match preprocessor
pattern_preproc = PatternMatchPreprocessor(patterns=[
    ("API", "エーピーアイ"),                        # Fixed string replacement
    ("URL", "ユーアールエル"),
    (re.compile(r"\d+"), lambda m: "number"),      # Regex: must be pre-compiled
])

# Add common patterns
pattern_preproc.add_number_dash_pattern()  # Number-dash patterns (e.g., 12-34 → イチニの サンヨン)
pattern_preproc.add_phonenumber_pattern()  # Phone number patterns

# Add to TTS
tts.preprocessors.append(pattern_preproc)
```

**Strings in `patterns` are treated as literals.** The constructor runs every plain string
through `re.escape()`, so passing `r"\d+"` matches the three characters `\d+` — not a number.
Hand regular expressions in already compiled, as above, or add them afterwards with the
explicit flag:

```python
pattern_preproc.add_pattern(r"\d+", lambda m: "number", regex=True)
```

## Creating Custom Preprocessors

You can create your own preprocessors by implementing the `TTSPreprocessor` interface:

```python
from aiavatar.sts.tts.preprocessor import TTSPreprocessor

class CustomPreprocessor(TTSPreprocessor):
    def __init__(self, custom_dict: dict = None):
        self.custom_dict = custom_dict or {}
    
    async def process(self, text: str, style_info: dict = None, language: str = None) -> str:
        # Custom conversion logic
        processed_text = text
        
        # Dictionary-based replacement
        for original, replacement in self.custom_dict.items():
            processed_text = processed_text.replace(original, replacement)
        
        # Language-specific conversions
        if language == "ja-JP":
            processed_text = processed_text.replace("OK", "オーケー")
        
        return processed_text

# Use custom preprocessor
custom_preproc = CustomPreprocessor(custom_dict={
    "GitHub": "ギットハブ",
    "Python": "パイソン",
    "Docker": "ドッカー"
})

tts.preprocessors.append(custom_preproc)
```

## Combining Preprocessors

Multiple preprocessors can be used together. They are executed in the order they were registered:

```python
# Combine multiple preprocessors
tts.preprocessors.extend([
    pattern_preproc,        # 1. Pattern match conversion
    alphabet2kana_preproc,  # 2. Alphabet to katakana conversion
    custom_preproc          # 3. Custom conversion
])
```

## See also

- [Text-to-Speech](tts.md) — the synthesizers preprocessors attach to
- [Instant TTS](tts-instant.md) — wrapping an arbitrary HTTP endpoint

---

[← Documentation index](../README.md#-documentation)
