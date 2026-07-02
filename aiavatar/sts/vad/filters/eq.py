import logging
import math
from typing import Dict, List

import numpy as np

from .base import AudioFilter

logger = logging.getLogger(__name__)


class HighShelfFilter(AudioFilter):
    """Biquad high-shelf EQ that boosts (or cuts) frequencies above the cutoff.

    Intended to recover intelligibility of fricatives (sibilants like /s/)
    on band-limited audio such as telephony: what little energy remains in
    the 2-3.4 kHz range gets emphasized so the recognizer can use it.

    Note: this can only emphasize what is present in the signal. Telephony
    audio carries nothing above about 3.4 kHz, and no EQ can restore it.

    Coefficients follow the RBJ Audio EQ Cookbook. The filter is stateful
    per session so chunk boundaries are seamless.
    """

    def __init__(
        self,
        *,
        gain_db: float = 6.0,
        cutoff_hz: float = 2000.0,
        slope: float = 1.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.gain_db = float(gain_db)
        self.cutoff_hz = float(cutoff_hz)
        self.slope = float(slope)
        self.sample_rate = sample_rate
        self.channels = max(1, int(channels))
        # (z1, z2) per channel per session
        self._states: Dict[str, List[List[float]]] = {}
        self._update_coefficients()

    def _update_coefficients(self):
        A = 10.0 ** (self.gain_db / 40.0)
        w0 = 2.0 * math.pi * self.cutoff_hz / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / self.slope - 1.0) + 2.0)
        sqrt_a2_alpha = 2.0 * math.sqrt(A) * alpha

        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + sqrt_a2_alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - sqrt_a2_alpha)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + sqrt_a2_alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - sqrt_a2_alpha

        self._b0 = b0 / a0
        self._b1 = b1 / a0
        self._b2 = b2 / a0
        self._a1 = a1 / a0
        self._a2 = a2 / a0

    def get_config(self) -> dict:
        return {
            "gain_db": self.gain_db,
            "cutoff_hz": self.cutoff_hz,
            "slope": self.slope,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }

    def set_config(self, config: dict) -> dict:
        updated = super().set_config(config)
        if updated:
            self._update_coefficients()
        return updated

    def _get_state(self, session_id: str) -> List[List[float]]:
        state = self._states.get(session_id)
        if state is None:
            state = [[0.0, 0.0] for _ in range(self.channels)]
            self._states[session_id] = state
        return state

    def process(self, samples: bytes, session_id: str) -> bytes:
        usable_len = len(samples) - (len(samples) % 2)
        if usable_len <= 0:
            return samples

        state = self._get_state(session_id)
        audio = np.frombuffer(samples[:usable_len], dtype=np.int16).astype(np.float64).tolist()

        b0, b1, b2, a1, a2 = self._b0, self._b1, self._b2, self._a1, self._a2
        channels = self.channels
        # Direct Form II Transposed, per channel state for interleaved samples
        for ch in range(channels):
            z1, z2 = state[ch]
            for i in range(ch, len(audio), channels):
                x = audio[i]
                y = b0 * x + z1
                z1 = b1 * x - a1 * y + z2
                z2 = b2 * x - a2 * y
                audio[i] = y
            state[ch][0] = z1
            state[ch][1] = z2

        filtered = np.clip(np.asarray(audio), -32768.0, 32767.0).astype(np.int16)
        return filtered.tobytes() + samples[usable_len:]

    def reset_session(self, session_id: str):
        self._states.pop(session_id, None)
