from aiavatar.sts.tts import SpeechSynthesizerRouter, create_instant_synthesizer
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.sts.tts.preprocessor.alphabet2kana import AlphabetToKanaPreprocessor
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer

from .config import AppConfig, TTSRouteConfig, load_json_object


TTS_PROVIDERS = {"voicevox", "openai", "instant"}


def prepare_tts_route(
    app_config: AppConfig,
    route: TTSRouteConfig,
) -> tuple[dict, bool]:
    name = f"AIAVATAR_{route.name.upper()}_TTS"
    config_name = f"{name}_CONFIG"
    if route.provider not in TTS_PROVIDERS:
        raise RuntimeError(
            f"{name} must be 'voicevox', 'openai', or 'instant'"
        )

    config = dict(load_json_object(config_name, route.config_json) or {})
    alphabet_to_kana = config.pop(
        "alphabet_to_kana",
        route.alphabet_to_kana_default,
    )
    if not isinstance(alphabet_to_kana, bool):
        raise RuntimeError(
            f"{config_name}.alphabet_to_kana must be true or false"
        )

    if route.provider == "voicevox":
        allowed_keys = {
            "base_url",
            "speaker",
            "style_mapper",
            "max_connections",
            "max_keepalive_connections",
            "timeout",
            "sample_rate",
            "cache_dir",
            "cache_ext",
        }
        defaults = {
            "base_url": app_config.voicevox_url,
            "speaker": app_config.voicevox_speaker,
            "cache_dir": app_config.voicevox_cache_dir,
        }
    elif route.provider == "openai":
        allowed_keys = {
            "base_url",
            "speaker",
            "model",
            "instructions",
            "style_mapper",
            "audio_format",
            "sample_rate",
            "max_connections",
            "max_keepalive_connections",
            "timeout",
            "cache_dir",
            "cache_ext",
        }
        defaults = {
            "base_url": app_config.tts_openai.base_url
            or "https://api.openai.com/v1",
            "model": app_config.openai_tts_model,
            "speaker": app_config.openai_tts_speaker,
            "instructions": app_config.openai_tts_instructions,
            "audio_format": "wav",
            "cache_dir": app_config.openai_tts_cache_dir,
        }
    else:
        allowed_keys = {
            "method",
            "url",
            "params",
            "headers",
            "json",
            "style_mapper",
            "max_connections",
            "max_keepalive_connections",
            "timeout",
            "sample_rate",
            "follow_redirects",
            "cache_dir",
            "cache_ext",
        }
        defaults = {}

    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise RuntimeError(
            f"{config_name} has unsupported keys for {route.provider}: "
            + ", ".join(unknown_keys)
        )

    resolved = {**defaults, **config}
    if route.provider == "voicevox" and (
        not isinstance(resolved["speaker"], int)
        or isinstance(resolved["speaker"], bool)
    ):
        raise RuntimeError(
            f"{config_name}.speaker must be an integer for voicevox"
        )
    if route.provider == "openai" and resolved.get("audio_format") != "wav":
        raise RuntimeError(
            f"{config_name}.audio_format must be 'wav' for the default app"
        )
    if route.provider == "instant":
        if not resolved.get("method"):
            raise RuntimeError(f"{config_name}.method is required for instant")
        if not resolved.get("url"):
            raise RuntimeError(f"{config_name}.url is required for instant")

    return resolved, alphabet_to_kana


def create_tts(
    provider: str,
    config: dict,
    *,
    openai_api_key: str | None,
    preprocessors: list,
    debug: bool,
):
    kwargs = {
        **config,
        "preprocessors": preprocessors,
        "debug": debug,
    }
    if provider == "voicevox":
        return VoicevoxSpeechSynthesizer(**kwargs)
    if provider == "openai":
        return OpenAISpeechSynthesizer(
            openai_api_key=openai_api_key,
            **kwargs,
        )
    return create_instant_synthesizer(**kwargs)


def build_default_tts(
    config: AppConfig,
) -> tuple[SpeechSynthesizerRouter, AlphabetToKanaPreprocessor | None]:
    ja_config, ja_alphabet_to_kana = prepare_tts_route(config, config.ja_tts)
    multi_config, multi_alphabet_to_kana = prepare_tts_route(
        config,
        config.multi_tts,
    )
    openai_required = (
        config.ja_tts.provider == "openai"
        or config.multi_tts.provider == "openai"
        or ja_alphabet_to_kana
        or multi_alphabet_to_kana
    )
    if openai_required and not config.tts_openai.api_key:
        raise RuntimeError(
            "OpenAI API key is required for TTS; set OPENAI_API_KEY or "
            "AIAVATAR_TTS_OPENAI_API_KEY"
        )

    alphabet_to_kana = None
    if ja_alphabet_to_kana or multi_alphabet_to_kana:
        alphabet_to_kana = AlphabetToKanaPreprocessor(
            openai_api_key=config.tts_openai.api_key,
            base_url=config.tts_openai.base_url,
            debug=config.debug,
        )

    tts_ja = create_tts(
        config.ja_tts.provider,
        ja_config,
        openai_api_key=config.tts_openai.api_key,
        preprocessors=[alphabet_to_kana] if ja_alphabet_to_kana else [],
        debug=config.debug,
    )
    tts_multi = create_tts(
        config.multi_tts.provider,
        multi_config,
        openai_api_key=config.tts_openai.api_key,
        preprocessors=[alphabet_to_kana] if multi_alphabet_to_kana else [],
        debug=config.debug,
    )
    router = SpeechSynthesizerRouter(
        synthesizers={
            "ja": tts_ja,
            "multi": tts_multi,
        }
    )

    @router.route
    def tts_route(text, style_info, language):
        language_code = (language or "").strip().lower().split("-", 1)[0]
        return "ja" if not language_code or language_code == "ja" else "multi"

    return router, alphabet_to_kana
