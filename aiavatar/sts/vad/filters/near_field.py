from collections import deque
import logging
import math
from typing import Dict, Optional

import numpy as np

from .base import AudioFilter

logger = logging.getLogger(__name__)


class NearFieldGateState:
    def __init__(self, closed_gain: float, initial_ambient_db: float):
        # Lookahead delay line: [samples, duration, gain] entries not yet emitted
        self.pending: deque = deque()
        self.pending_duration: float = 0.0
        # Ambient noise floor estimation: (rms_db, duration) entries
        self.rms_history: deque = deque()
        self.rms_history_duration: float = 0.0
        self.elapsed: float = 0.0
        self.is_open: bool = False
        self.open_streak_duration: float = 0.0
        self.not_passing_duration: float = 0.0
        self.last_emitted_gain: float = closed_gain
        # Diagnostics
        self.last_rms_db: Optional[float] = None
        self.last_ambient_db: float = initial_ambient_db
        self.last_snr_db: Optional[float] = None
        self.last_gate_reason: str = "not_checked"


class NearFieldAudioGate(AudioFilter):
    """Acoustic gate that attenuates far-field audio before it reaches the VAD.

    Instead of overriding the VAD verdict, this filter controls what audio
    flows downstream: audio judged as coming from the main (near-field)
    speaker passes through unchanged, everything else is attenuated by
    `closed_gain`. The VAD and recording pipeline behave exactly as usual,
    but they only "hear" the main speaker — so far-field speech neither
    starts a recording nor contaminates the audio sent to the recognizer.

    The near/far decision compares each chunk's RMS level against a
    dynamically estimated ambient noise floor (a low percentile of recent
    quiet-chunk RMS history). The gate opens when the SNR stays above
    `open_snr_db_threshold` for `open_min_duration`, and closes when the
    open condition keeps failing for `close_min_duration`.

    Opening the gate takes `open_min_duration`, so the filter delays its
    output by `lookahead_duration`: when the gate opens, chunks still in
    the delay line — including the speech onset — are released at full
    gain. Keep `lookahead_duration >= open_min_duration` so the beginning
    of utterances is never clipped. The whole pipeline is delayed by
    `lookahead_duration` (120 ms by default) as a trade-off.

    Gain transitions are crossfaded over one chunk to avoid clicks.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        sample_rate: int = 16000,
        channels: int = 1,
        closed_gain: float = 0.05,
        lookahead_duration: float = 0.12,
        min_rms_db: Optional[float] = -42.0,
        open_snr_db_threshold: float = 12.0,
        close_snr_db_threshold: float = 6.0,
        open_min_duration: float = 0.06,
        close_min_duration: float = 0.4,
        ambient_window_duration: float = 2.5,
        ambient_percentile: float = 25.0,
        initial_ambient_db: float = -65.0,
        calibration_duration: float = 0.0,
        update_ambient_with_rejected_speech: bool = True,
        ambient_max_rise_db_per_update: Optional[float] = 3.0,
        debug: bool = False,
    ):
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.channels = channels
        self.closed_gain = max(0.0, min(1.0, float(closed_gain)))
        self.lookahead_duration = max(0.0, float(lookahead_duration))
        self.min_rms_db = min_rms_db
        self.open_snr_db_threshold = float(open_snr_db_threshold)
        self.close_snr_db_threshold = float(close_snr_db_threshold)
        self.open_min_duration = max(0.0, float(open_min_duration))
        self.close_min_duration = max(0.0, float(close_min_duration))
        self.ambient_window_duration = max(0.1, float(ambient_window_duration))
        self.ambient_percentile = float(ambient_percentile)
        self.initial_ambient_db = float(initial_ambient_db)
        self.calibration_duration = max(0.0, float(calibration_duration))
        self.update_ambient_with_rejected_speech = update_ambient_with_rejected_speech
        self.ambient_max_rise_db_per_update = ambient_max_rise_db_per_update
        self.debug = debug
        self._states: Dict[str, NearFieldGateState] = {}

    def get_config(self) -> dict:
        return {
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "closed_gain": self.closed_gain,
            "lookahead_duration": self.lookahead_duration,
            "min_rms_db": self.min_rms_db,
            "open_snr_db_threshold": self.open_snr_db_threshold,
            "close_snr_db_threshold": self.close_snr_db_threshold,
            "open_min_duration": self.open_min_duration,
            "close_min_duration": self.close_min_duration,
            "ambient_window_duration": self.ambient_window_duration,
            "ambient_percentile": self.ambient_percentile,
            "initial_ambient_db": self.initial_ambient_db,
            "calibration_duration": self.calibration_duration,
            "update_ambient_with_rejected_speech": self.update_ambient_with_rejected_speech,
            "ambient_max_rise_db_per_update": self.ambient_max_rise_db_per_update,
            "debug": self.debug,
        }

    def get_diagnostics(self, session_id: str) -> Optional[dict]:
        state = self._states.get(session_id)
        if state is None:
            return None
        return {
            "rms_db": state.last_rms_db,
            "ambient_db": state.last_ambient_db,
            "snr_db": state.last_snr_db,
            "is_open": state.is_open,
            "gate_reason": state.last_gate_reason,
            "pending_duration": state.pending_duration,
        }

    def _get_state(self, session_id: str) -> NearFieldGateState:
        state = self._states.get(session_id)
        if state is None:
            state = NearFieldGateState(self.closed_gain, self.initial_ambient_db)
            self._states[session_id] = state
        return state

    def _rms_dbfs(self, samples: bytes) -> float:
        usable_len = len(samples) - (len(samples) % 2)
        if usable_len <= 0:
            return -120.0

        audio = np.frombuffer(samples[:usable_len], dtype=np.int16)
        if audio.size == 0:
            return -120.0

        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
        return 20.0 * math.log10(max(rms, 1.0) / 32768.0)

    def _ambient_db(self, state: NearFieldGateState) -> float:
        if not state.rms_history:
            return self.initial_ambient_db
        percentile = min(100.0, max(0.0, self.ambient_percentile))
        values = np.asarray([v for v, _ in state.rms_history], dtype=np.float32)
        return float(np.percentile(values, percentile))

    def _update_ambient(self, state: NearFieldGateState, rms_db: float, duration: float):
        ambient_db = self._ambient_db(state)
        if self.ambient_max_rise_db_per_update is not None:
            max_rms_db = ambient_db + float(self.ambient_max_rise_db_per_update)
            rms_db = min(float(rms_db), max_rms_db)

        state.rms_history.append((float(rms_db), duration))
        state.rms_history_duration += duration
        while state.rms_history and state.rms_history_duration > self.ambient_window_duration:
            _, dropped_duration = state.rms_history.popleft()
            state.rms_history_duration -= dropped_duration
        state.last_ambient_db = self._ambient_db(state)

    def _apply_gain(self, samples: bytes, from_gain: float, to_gain: float) -> bytes:
        if from_gain == 1.0 and to_gain == 1.0:
            return samples

        usable_len = len(samples) - (len(samples) % 2)
        if usable_len <= 0:
            return samples

        audio = np.frombuffer(samples[:usable_len], dtype=np.int16).astype(np.float32)
        if from_gain == to_gain:
            audio *= to_gain
        else:
            # Crossfade over the chunk to avoid clicks at gate transitions
            audio *= np.linspace(from_gain, to_gain, audio.size, dtype=np.float32)
        return np.clip(audio, -32768.0, 32767.0).astype(np.int16).tobytes() + samples[usable_len:]

    def _emit(self, state: NearFieldGateState, flush: bool = False) -> bytes:
        output = bytearray()
        while state.pending:
            if not flush and state.pending_duration - state.pending[0][1] < self.lookahead_duration:
                # Keep at least lookahead_duration buffered
                break
            samples, duration, gain = state.pending.popleft()
            state.pending_duration -= duration
            output.extend(self._apply_gain(samples, state.last_emitted_gain, gain))
            state.last_emitted_gain = gain
        return bytes(output)

    def process(self, samples: bytes, session_id: str) -> bytes:
        state = self._get_state(session_id)

        if not self.enabled:
            state.last_gate_reason = "disabled"
            # Flush any buffered audio, then pass through without delay
            return self._emit(state, flush=True) + samples

        duration = (len(samples) / 2) / (self.sample_rate * self.channels)
        state.elapsed += duration

        rms_db = self._rms_dbfs(samples)
        ambient_db = self._ambient_db(state)
        snr_db = rms_db - ambient_db

        state.last_rms_db = rms_db
        state.last_ambient_db = ambient_db
        state.last_snr_db = snr_db

        in_calibration = state.elapsed <= self.calibration_duration
        passes_min_rms = self.min_rms_db is None or rms_db >= self.min_rms_db
        passes_snr = snr_db >= self.open_snr_db_threshold
        passes = passes_min_rms and passes_snr and not in_calibration
        is_quiet = snr_db < self.close_snr_db_threshold

        if state.is_open:
            if passes:
                state.not_passing_duration = 0.0
            else:
                state.not_passing_duration += duration
                if is_quiet:
                    # Keep the ambient estimate updated with quiet chunks
                    # between the main speaker's words
                    self._update_ambient(state, rms_db, duration)
                if state.not_passing_duration >= self.close_min_duration:
                    state.is_open = False
                    state.open_streak_duration = 0.0
            state.last_gate_reason = "open" if state.is_open else "closed"
        else:
            if in_calibration:
                self._update_ambient(state, rms_db, duration)
                state.open_streak_duration = 0.0
                state.last_gate_reason = "calibrating"
            elif passes:
                state.open_streak_duration += duration
                if state.open_streak_duration >= self.open_min_duration:
                    state.is_open = True
                    state.not_passing_duration = 0.0
                    # Retroactively release the buffered chunks — they contain
                    # the onset of the utterance
                    for entry in state.pending:
                        entry[2] = 1.0
                    state.last_gate_reason = "open"
                else:
                    state.last_gate_reason = "waiting_open_duration"
            else:
                state.open_streak_duration = 0.0
                if is_quiet or self.update_ambient_with_rejected_speech:
                    self._update_ambient(state, rms_db, duration)
                if not passes_min_rms:
                    state.last_gate_reason = "below_min_rms"
                elif not passes_snr:
                    state.last_gate_reason = "below_snr"
                else:
                    state.last_gate_reason = "closed"

        gain = 1.0 if state.is_open else self.closed_gain
        state.pending.append([samples, duration, gain])
        state.pending_duration += duration

        output = self._emit(state)

        if self.debug:
            logger.debug(
                "near-field audio gate: open=%s reason=%s rms_db=%.1f ambient_db=%.1f snr_db=%.1f pending=%.3fs out_bytes=%s session=%s",
                state.is_open,
                state.last_gate_reason,
                rms_db,
                ambient_db,
                snr_db,
                state.pending_duration,
                len(output),
                session_id,
            )

        return output

    def flush(self, session_id: str) -> bytes:
        """Emit all buffered audio for the session without waiting for lookahead."""
        state = self._states.get(session_id)
        if state is None:
            return b""
        return self._emit(state, flush=True)

    def reset_session(self, session_id: str):
        self._states.pop(session_id, None)
