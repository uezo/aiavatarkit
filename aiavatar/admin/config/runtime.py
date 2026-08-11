import inspect
import json
from typing import Any, get_args, get_origin

from fastapi import APIRouter, Body, HTTPException

from ...sts.tts import SpeechSynthesizerRouter


# Values that need resource recreation, derived-state rebuilding, or hook
# registration are not suitable for this direct, process-local editor.
_EXCLUDED = {
    # Injected graph and callbacks
    "llm", "stt", "sts", "tts", "vad", "speech_recognizer",
    "context_manager", "session_state_manager", "response_id_store",
    "performance_recorder", "voice_recorder", "db_pool_provider",
    "audio_filters", "guardrails", "postprocessors", "preprocessors",
    "turn_end_gates", "on_recording_started", "to_linear16",
    # Values coupled to resources, derived state, or registered hooks
    "audio_format", "channel", "channels", "insert_channel_tag",
    "db_connection_str",
    "follow_redirects", "hub_cache_path", "model_path", "model_pool_size",
    "max_connection_age", "max_connections", "max_keepalive_connections",
    "option_split_chars", "sample_rate", "split_chars", "timeout",
    "use_invoke_queue", "use_vad_iterator", "voice_recorder_enabled",
}
_SECRET_WORDS = ("api_key", "password", "secret", "subscription_key", "token")
_SECRET_SUFFIXES = tuple(f"_{word}" for word in _SECRET_WORDS)
_KINDS = {
    bool: "boolean",
    int: "number",
    float: "number",
    str: "string",
    list: "json",
    dict: "json",
}


def _kind(parameter: inspect.Parameter, value: Any) -> str | None:
    if value is not None:
        return _KINDS.get(type(value))
    annotation = parameter.annotation
    candidates = get_args(annotation) or (annotation,)
    for candidate in candidates:
        kind = _KINDS.get(get_origin(candidate) or candidate)
        if kind:
            return kind
    return None


def _nullable(parameter: inspect.Parameter) -> bool:
    return (
        parameter.default is None
        or type(None) in get_args(parameter.annotation)
    )


def _fields(target: Any) -> list[dict]:
    try:
        parameters = inspect.signature(type(target)).parameters
    except (TypeError, ValueError):
        return []

    fields = []
    members = vars(target)
    for name, parameter in parameters.items():
        if name in _EXCLUDED or name not in members:
            continue
        value = members[name]
        kind = _kind(parameter, value)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        if not kind:
            continue
        lower_name = name.lower()
        secret = lower_name in _SECRET_WORDS or lower_name.endswith(_SECRET_SUFFIXES)
        fields.append({
            "name": name,
            "label": name.replace("_", " "),
            "kind": kind,
            "value": None if secret else value,
            "configured": bool(value) if secret else None,
            "secret": secret,
            "nullable": _nullable(parameter),
        })
    return fields


def _valid(field: dict, value: Any) -> bool:
    if value is None:
        return field["nullable"]
    current = field["value"]
    if field["kind"] == "number" and isinstance(current, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if field["kind"] == "json" and isinstance(current, (list, dict)):
        return isinstance(value, type(current))
    return {
        "string": lambda: isinstance(value, str),
        "boolean": lambda: isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "json": lambda: isinstance(value, (list, dict)),
    }[field["kind"]]()


def create_runtime_config_router(adapters: dict[str, Any]) -> APIRouter:
    router = APIRouter()

    def targets():
        pipeline = next(iter(adapters.values())).sts
        result = {
            "pipeline": ("Pipeline", pipeline),
            "vad": ("VAD", pipeline.vad),
            "stt": ("STT", pipeline.stt),
            "llm": ("LLM", pipeline.llm),
        }
        if isinstance(pipeline.tts, SpeechSynthesizerRouter):
            result.update({
                f"tts:{route}": (f"TTS · {route}", synthesizer)
                for route, synthesizer in pipeline.tts.synthesizers.items()
            })
        else:
            result["tts"] = ("TTS", pipeline.tts)
        result.update({
            f"adapter:{name}": (f"Adapter · {name}", adapter)
            for name, adapter in adapters.items()
        })
        return result

    @router.get("/config/runtime", tags=["Config"])
    async def get_runtime_config():
        return {"sections": [
            {
                "name": name,
                "title": title,
                "component": type(target).__name__ if name != "pipeline" else None,
                "fields": _fields(target),
            }
            for name, (title, target) in targets().items()
        ]}

    @router.post("/config/runtime/{section}", tags=["Config"])
    async def post_runtime_config(section: str, config: dict = Body(embed=True)):
        try:
            target = targets()[section][1]
        except KeyError as ex:
            raise HTTPException(status_code=404, detail=f"Unknown section: {section}") from ex

        fields = {field["name"]: field for field in _fields(target)}
        changes = {
            name: value
            for name, value in config.items()
            if not (fields.get(name, {}).get("secret") and value == "")
        }
        unknown = changes.keys() - fields.keys()
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown setting: {sorted(unknown)[0]}")
        for name, value in changes.items():
            if not _valid(fields[name], value):
                raise HTTPException(status_code=400, detail=f"Invalid value for '{name}'")
        for name, value in changes.items():
            setattr(target, name, value)
        return {"updated": list(changes)}

    return router
