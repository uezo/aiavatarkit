from abc import ABC, abstractmethod


class AudioFilter(ABC):
    """Transforms audio samples (16-bit linear PCM) before VAD processing.

    Filters are chained in order via SpeechDetector's `audio_filters`.
    All downstream processing (VAD, recording, recognition) sees the
    filtered audio, so the VAD logic itself stays untouched. Use this
    for acoustic transforms such as gating, noise suppression, gain
    normalization or target speech extraction.

    A filter may hold audio back internally (e.g. a lookahead buffer)
    and return fewer or more bytes than it received, including b"" while
    warming up. The only requirement is that audio is conserved over the
    session.
    """

    @abstractmethod
    def process(self, samples: bytes, session_id: str) -> bytes:
        pass

    def reset_session(self, session_id: str):
        """Called when the session is deleted. Override to release per-session state."""
        pass

    def get_config(self) -> dict:
        return {}

    def set_config(self, config: dict) -> dict:
        allowed_keys = self.get_config().keys()
        updated = {}
        for k, v in config.items():
            if v is None:
                continue
            if k not in allowed_keys:
                continue
            try:
                setattr(self, k, v)
                updated[k] = v
            except Exception:
                pass
        return updated
