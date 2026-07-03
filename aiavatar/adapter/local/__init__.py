def _raise_local_audio_import_error(exc: ModuleNotFoundError):
    if exc.name == "pyaudio" or "pyaudio" in str(exc).lower():
        raise ImportError(
            "aiavatar.adapter.local is deprecated and requires PyAudio. "
            "Install local audio dependencies with `pip install aiavatar[local-audio]`, "
            "or use AIAvatarWebSocketClient for local Python audio clients."
        ) from exc
    raise exc


def __getattr__(name: str):
    try:
        if name == "AIAvatar":
            from .client import AIAvatar
            return AIAvatar
        if name == "AIAvatarLocalServer":
            from .server import AIAvatarLocalServer
            return AIAvatarLocalServer
    except ModuleNotFoundError as exc:
        _raise_local_audio_import_error(exc)

    raise AttributeError(f"module 'aiavatar.adapter.local' has no attribute '{name}'")
