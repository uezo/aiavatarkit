import logging
import math
from typing import Dict, Optional

import numpy as np

from .base import AudioFilter

logger = logging.getLogger(__name__)


class AGCState:
    def __init__(self):
        self.gain_db: float = 0.0
        self.last_applied_gain: float = 1.0
        # Diagnostics
        self.last_rms_db: Optional[float] = None
        self.last_gain_db: float = 0.0


class AGCFilter(AudioFilter):
    """Automatic gain control that normalizes speech towards a target level.

    Slowly raises the gain of quiet audio (e.g. telephony) towards
    `target_rms_db` and quickly backs off when the input gets loud. Gain
    adaptation only happens on chunks above `silence_rms_db`, so silence
    (or audio attenuated by an upstream gate) does not pump the gain up.

    A per-chunk peak limiter caps the applied gain so samples never clip,
    and gain changes are ramped across each chunk to avoid zipper noise.

    Place this AFTER NearFieldAudioGate: putting it before would amplify
    far-field audio and break the gate's absolute level threshold
    (min_rms_db).
    """

    def __init__(
        self,
        *,
        target_rms_db: float = -20.0,
        max_gain_db: float = 18.0,
        min_gain_db: float = 0.0,
        up_db_per_sec: float = 10.0,
        down_db_per_sec: float = 60.0,
        silence_rms_db: float = -55.0,
        limiter_headroom_db: float = 1.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.target_rms_db = float(target_rms_db)
        self.max_gain_db = float(max_gain_db)
        self.min_gain_db = float(min_gain_db)
        self.up_db_per_sec = max(0.0, float(up_db_per_sec))
        self.down_db_per_sec = max(0.0, float(down_db_per_sec))
        self.silence_rms_db = float(silence_rms_db)
        self.limiter_headroom_db = max(0.0, float(limiter_headroom_db))
        self.sample_rate = sample_rate
        self.channels = max(1, int(channels))
        self._states: Dict[str, AGCState] = {}

    def get_config(self) -> dict:
        return {
            "target_rms_db": self.target_rms_db,
            "max_gain_db": self.max_gain_db,
            "min_gain_db": self.min_gain_db,
            "up_db_per_sec": self.up_db_per_sec,
            "down_db_per_sec": self.down_db_per_sec,
            "silence_rms_db": self.silence_rms_db,
            "limiter_headroom_db": self.limiter_headroom_db,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }

    def get_diagnostics(self, session_id: str) -> Optional[dict]:
        state = self._states.get(session_id)
        if state is None:
            return None
        return {
            "rms_db": state.last_rms_db,
            "gain_db": state.last_gain_db,
        }

    def _get_state(self, session_id: str) -> AGCState:
        state = self._states.get(session_id)
        if state is None:
            state = AGCState()
            self._states[session_id] = state
        return state

    def process(self, samples: bytes, session_id: str) -> bytes:
        usable_len = len(samples) - (len(samples) % 2)
        if usable_len <= 0:
            return samples

        state = self._get_state(session_id)
        audio = np.frombuffer(samples[:usable_len], dtype=np.int16).astype(np.float32)
        duration = (len(audio) / self.channels) / self.sample_rate

        rms = float(np.sqrt(np.mean(np.square(audio))))
        rms_db = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
        state.last_rms_db = rms_db

        # Adapt gain only while there is actual signal; hold during silence
        if rms_db >= self.silence_rms_db:
            desired_gain_db = min(self.max_gain_db, max(self.min_gain_db, self.target_rms_db - rms_db))
            if desired_gain_db > state.gain_db:
                state.gain_db = min(desired_gain_db, state.gain_db + self.up_db_per_sec * duration)
            else:
                state.gain_db = max(desired_gain_db, state.gain_db - self.down_db_per_sec * duration)

        gain = 10.0 ** (state.gain_db / 20.0)

        # Per-chunk peak limiter: never let the applied gain clip the samples
        peak = float(np.abs(audio).max())
        if peak > 0.0:
            max_gain = (32767.0 * 10.0 ** (-self.limiter_headroom_db / 20.0)) / peak
            gain = min(gain, max_gain)
        state.last_gain_db = 20.0 * math.log10(max(gain, 1e-6))

        # Ramp between the previous chunk's gain and this one to avoid zipper noise
        if gain == state.last_applied_gain:
            if gain != 1.0:
                audio *= gain
        else:
            audio *= np.linspace(state.last_applied_gain, gain, audio.size, dtype=np.float32)
        state.last_applied_gain = gain

        processed = np.clip(audio, -32768.0, 32767.0).astype(np.int16)
        return processed.tobytes() + samples[usable_len:]

    def reset_session(self, session_id: str):
        self._states.pop(session_id, None)
