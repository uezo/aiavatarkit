import json
import os
from dataclasses import dataclass, field

from aiavatar.sts.vad.turn_end_gates.filler import (
    DEFAULT_EN_FILLERS,
    DEFAULT_JA_FILLERS,
)


OPENAI_TTS_INSTRUCTIONS = (
    "Speak naturally in the language of the input text, using a clear, "
    "friendly, conversational tone and a moderate pace."
)

SYSTEM_PROMPT = """\
Follow the speaking style and output rules below.

## Language Switching
When switching to a different language, insert a tag such as <language code="en-US" />.
Required output and language tags are control tags and are exempt from the output constraints below.

## Acknowledgment, opening response, and reasoning
Your output must consist of an acknowledgment or opening response, followed by reasoning, and then the main response.

### Format

<ack><language code="en-US" />Acknowledgment or opening response</ack>
<think>Reasoning</think>
<answer><language code="en-US" />Main response</answer>

The language tags shown above are examples. Omit them when speaking Japanese,
which is the default language, and replace the language code as appropriate.

### Content

- Acknowledgment or opening response: Include a brief affirmative, negative, filler, or similar opening. It must end with punctuation such as a period or exclamation mark. When speaking a language other than Japanese, begin the content with the appropriate language tag, such as <language code="en-US" />.
- Reasoning: State what should be considered and covered in the response. Always reason first, even for a very short response.
- Main response: State what will ultimately be conveyed to the user. Do not repeat the acknowledgment or opening response; continue naturally from it. If the opening response conflicts with the main response, correct course and treat the main response as authoritative. When speaking a language other than Japanese, begin the content with the appropriate language tag, such as <language code="en-US" />.

## Expressions
neutral / joy / angry / sorrow / fun / surprise
When expressing emotion, insert a tag such as <face name="joy" />. Default is neutral.

## Supervisor instructions
Any message beginning with "$" is an instruction from the supervisor program.
Do not respond directly to the supervisor program. Follow its instructions when producing the response for the user.

## Output constraints
- The output will be synthesized as speech, so do not use emoji, symbols, stage directions, URLs, or similar content.
- Do not use Markdown or other formatting syntax.
- Use natural, fluent spoken language.
- Keep the combined spoken content inside <ack> and <answer> to roughly 10 seconds of speech: approximately 50 Japanese characters, 25 English words, or an equivalent amount in other languages. Do not count control tags or <think> content toward this limit.

## Additional considerations
- The user's input comes from speech recognition and may contain transcription errors. Infer the intended meaning from the context.
"""


def load_json_object(name: str, value: str | None = None) -> dict | None:
    value = os.getenv(name) if value is None else value
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"{name} must be a valid JSON object") from ex
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return parsed


def load_string_list(name: str, default: list) -> list:
    value = os.getenv(name)
    if not value:
        return list(default)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"{name} must be a valid JSON array of strings") from ex
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise RuntimeError(f"{name} must be a JSON array of strings")
    return parsed


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as ex:
        raise RuntimeError(f"{name} must be a number") from ex


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as ex:
        raise RuntimeError(f"{name} must be an integer") from ex


def env_nullable_string(
    name: str,
    default: str | None,
    *,
    null_values: set[str] | None = None,
) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in (null_values or {"", "null", "none"}):
        return None
    return value


def resolve_reasoning_effort(extra_body: dict | None) -> str | None:
    if not os.getenv("AIAVATAR_LLM_REASONING_EFFORT"):
        return None if extra_body else "none"
    return env_nullable_string(
        "AIAVATAR_LLM_REASONING_EFFORT",
        "none",
        null_values={"", "null", "omit"},
    )


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None

    @classmethod
    def from_env(cls, component: str):
        prefix = f"AIAVATAR_{component.upper()}_OPENAI"
        return cls(
            api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv(f"{prefix}_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
        )


@dataclass(frozen=True)
class TTSRouteConfig:
    name: str
    provider: str
    config_json: str | None
    alphabet_to_kana_default: bool


@dataclass(frozen=True)
class AppConfig:
    stt_openai: OpenAIConfig
    llm_openai: OpenAIConfig
    tts_openai: OpenAIConfig
    debug: bool

    stt_model: str
    stt_language: str | None

    near_field_initial_ambient_db: float
    filler_phrases: list
    filler_timeout: float
    namo_language: str | None
    namo_threshold: float
    namo_timeout: float
    namo_force_end_phrases: list[str]
    vad_silence_duration_threshold: float
    vad_segment_silence_threshold: float
    vad_use_iterator: bool

    llm_api: str
    llm_model: str
    llm_system_prompt: str
    llm_extra_body: dict | None
    llm_reasoning_effort: str | None
    llm_voice_text_tags: list[str]

    ja_tts: TTSRouteConfig
    multi_tts: TTSRouteConfig
    voicevox_url: str
    voicevox_speaker: int
    voicevox_cache_dir: str | None
    openai_tts_model: str
    openai_tts_speaker: str
    openai_tts_instructions: str
    openai_tts_cache_dir: str | None

    timestamp_interval_seconds: float
    timestamp_timezone: str
    merge_request_threshold: float
    use_invoke_queue: bool
    mute_on_barge_in: bool

    @classmethod
    def from_env(cls):
        llm_extra_body = load_json_object("AIAVATAR_LLM_EXTRA_BODY")
        return cls(
            stt_openai=OpenAIConfig.from_env("stt"),
            llm_openai=OpenAIConfig.from_env("llm"),
            tts_openai=OpenAIConfig.from_env("tts"),
            debug=env_bool("AIAVATAR_DEBUG", True),
            stt_model=os.getenv("AIAVATAR_STT_MODEL") or "gpt-transcribe",
            stt_language=env_nullable_string(
                "AIAVATAR_STT_LANGUAGE",
                None,
                null_values={"", "auto", "null", "none"},
            ),
            near_field_initial_ambient_db=env_float(
                "AIAVATAR_NEAR_FIELD_INITIAL_AMBIENT_DB",
                -30.0,
            ),
            filler_phrases=load_string_list(
                "AIAVATAR_FILLER_PHRASES",
                DEFAULT_JA_FILLERS + DEFAULT_EN_FILLERS,
            ),
            filler_timeout=env_float("FILLER_TURN_GATE_TIMEOUT", 3.0),
            namo_language=env_nullable_string(
                "AIAVATAR_NAMO_LANGUAGE",
                None,
                null_values={"", "auto", "null", "none"},
            ),
            namo_threshold=env_float("NAMO_TURN_THRESHOLD", 0.5),
            namo_timeout=env_float("NAMO_TURN_GATE_TIMEOUT", 1.0),
            namo_force_end_phrases=load_string_list(
                "AIAVATAR_NAMO_FORCE_END_PHRASES",
                ["こんにちは。"],
            ),
            vad_silence_duration_threshold=env_float(
                "AIAVATAR_VAD_SILENCE_DURATION_THRESHOLD",
                0.5,
            ),
            vad_segment_silence_threshold=env_float(
                "AIAVATAR_VAD_SEGMENT_SILENCE_THRESHOLD",
                0.05,
            ),
            vad_use_iterator=env_bool("AIAVATAR_VAD_USE_ITERATOR", True),
            llm_api=os.getenv(
                "AIAVATAR_LLM_API",
                "responses-websocket",
            ).strip().lower(),
            llm_model=os.getenv("AIAVATAR_LLM_MODEL") or "gpt-5.6-terra",
            llm_system_prompt=os.getenv(
                "AIAVATAR_LLM_SYSTEM_PROMPT",
                SYSTEM_PROMPT,
            ),
            llm_extra_body=llm_extra_body,
            llm_reasoning_effort=resolve_reasoning_effort(llm_extra_body),
            llm_voice_text_tags=load_string_list(
                "AIAVATAR_LLM_VOICE_TEXT_TAGS",
                ["ack", "answer"],
            ),
            ja_tts=TTSRouteConfig(
                name="ja",
                provider=os.getenv("AIAVATAR_JA_TTS", "voicevox").strip().lower(),
                config_json=os.getenv("AIAVATAR_JA_TTS_CONFIG"),
                alphabet_to_kana_default=True,
            ),
            multi_tts=TTSRouteConfig(
                name="multi",
                provider=os.getenv("AIAVATAR_MULTI_TTS", "openai").strip().lower(),
                config_json=os.getenv("AIAVATAR_MULTI_TTS_CONFIG"),
                alphabet_to_kana_default=False,
            ),
            voicevox_url=os.getenv(
                "AIAVATAR_VOICEVOX_URL",
                "http://127.0.0.1:50021",
            ),
            voicevox_speaker=env_int("AIAVATAR_VOICEVOX_SPEAKER", 46),
            voicevox_cache_dir=env_nullable_string(
                "AIAVATAR_VOICEVOX_CACHE_DIR",
                "ttscache/voicevox",
            ),
            openai_tts_model=os.getenv("AIAVATAR_OPENAI_TTS_MODEL")
            or "gpt-4o-mini-tts",
            openai_tts_speaker=os.getenv("AIAVATAR_OPENAI_TTS_SPEAKER", "sage"),
            openai_tts_instructions=os.getenv(
                "AIAVATAR_OPENAI_TTS_INSTRUCTIONS",
                OPENAI_TTS_INSTRUCTIONS,
            ),
            openai_tts_cache_dir=env_nullable_string(
                "AIAVATAR_OPENAI_TTS_CACHE_DIR",
                "ttscache/openai",
            ),
            timestamp_interval_seconds=env_float(
                "AIAVATAR_TIMESTAMP_INTERVAL_SECONDS",
                600.0,
            ),
            timestamp_timezone=os.getenv(
                "AIAVATAR_TIMESTAMP_TIMEZONE",
                "Asia/Tokyo",
            ),
            merge_request_threshold=env_float(
                "AIAVATAR_MERGE_REQUEST_THRESHOLD",
                3.0,
            ),
            use_invoke_queue=env_bool("AIAVATAR_USE_INVOKE_QUEUE", True),
            mute_on_barge_in=env_bool("AIAVATAR_MUTE_ON_BARGE_IN", True),
        )
