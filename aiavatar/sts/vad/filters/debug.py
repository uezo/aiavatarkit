from datetime import datetime
import logging
import os
import re
import time
from typing import Dict, Iterable, Optional, Set
import wave

from .base import AudioFilter

logger = logging.getLogger(__name__)


class _SessionWavWriter:
    """Writes one label's audio for one session, rotating files by duration."""

    def __init__(self, directory: str, label: str, sample_rate: int, channels: int, max_file_duration: Optional[float]):
        self.directory = directory
        self.label = label
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_file_duration = max_file_duration
        self.part = 0
        self.frames_written = 0
        self._writer: Optional[wave.Wave_write] = None

    def _open_next(self):
        self.part += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.directory, f"{self.label}_{timestamp}_{self.part:03d}.wav")
        self._writer = wave.open(path, "wb")
        self._writer.setnchannels(self.channels)
        self._writer.setsampwidth(2)
        self._writer.setframerate(self.sample_rate)
        self.frames_written = 0
        logger.info(f"SessionAudioRecorder({self.label}) recording to {path}")

    def write(self, samples: bytes):
        if self._writer is None:
            self._open_next()
        elif self.max_file_duration is not None and self.frames_written / self.sample_rate >= self.max_file_duration:
            self.close()
            self._open_next()
        self._writer.writeframes(samples)
        self.frames_written += len(samples) // (2 * self.channels)

    def close(self):
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                logger.error(f"SessionAudioRecorder({self.label}) failed to close writer", exc_info=True)
            self._writer = None


class _RecorderTap(AudioFilter):
    """Pass-through tap that feeds audio into a SessionAudioRecorder."""

    def __init__(self, recorder: "SessionAudioRecorder", label: str):
        self.recorder = recorder
        self.label = label

    def process(self, samples: bytes, session_id: str) -> bytes:
        self.recorder.write(session_id, self.label, samples)
        return samples

    def reset_session(self, session_id: str):
        # Detector's delete_session reaches here: the session is over
        self.recorder.finalize_session(session_id)


class SessionAudioRecorder:
    """Debug recorder that captures per-session audio continuously to WAV files.

    Unlike RecordingSession (which is reset per utterance), this records for
    the whole lifetime of a session ID. Place taps at any points of the
    audio_filters chain to capture and compare the audio at those points,
    e.g. raw vs gated:

        recorder = SessionAudioRecorder("debug_audio")
        detector = SileroStreamSpeechDetector(
            ...,
            audio_filters=[
                recorder.tap("raw"),
                NearFieldAudioGate(),
                recorder.tap("gated"),
            ]
        )

    Files are written under `directory/<session_id>/<label>_<timestamp>_<part>.wav`.
    Audio is streamed to disk incrementally (not buffered in memory), so memory
    usage stays constant regardless of session length.

    Session end must be explicit: it happens automatically when the detector's
    delete_session / finalize_session runs (e.g. on WebSocket disconnect), or
    call finalize_session() / close() directly. As a safety net, sessions idle
    longer than `session_ttl` seconds are finalized opportunistically so file
    handles cannot leak when a session ends without cleanup.

    Note: taps placed after a lookahead filter (e.g. NearFieldAudioGate)
    receive audio delayed by its lookahead duration, so the "gated" file is
    time-shifted and slightly shorter than the "raw" file.

    Set `target_session_ids` to record only specific sessions (None records
    all). The set can be mutated at runtime to start/stop capturing a
    problematic session in production.
    """

    def __init__(
        self,
        directory: str,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        target_session_ids: Optional[Iterable[str]] = None,
        max_file_duration: Optional[float] = None,
        session_ttl: float = 3600.0,
    ):
        self.directory = directory
        self.sample_rate = sample_rate
        self.channels = channels
        self.target_session_ids: Optional[Set[str]] = set(target_session_ids) if target_session_ids is not None else None
        self.max_file_duration = max_file_duration
        self.session_ttl = session_ttl
        self._sessions: Dict[str, Dict[str, _SessionWavWriter]] = {}
        self._last_activity: Dict[str, float] = {}
        self._last_sweep: float = time.time()
        os.makedirs(self.directory, exist_ok=True)

    def tap(self, label: str) -> AudioFilter:
        """Create a pass-through AudioFilter that records audio under the given label."""
        return _RecorderTap(self, label)

    def _session_directory(self, session_id: str) -> str:
        safe_session_id = re.sub(r"[^\w.-]", "_", session_id)
        path = os.path.join(self.directory, safe_session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def write(self, session_id: str, label: str, samples: bytes):
        if self.target_session_ids is not None and session_id not in self.target_session_ids:
            return
        if not samples:
            return

        try:
            now = time.time()
            self._last_activity[session_id] = now

            writers = self._sessions.get(session_id)
            if writers is None:
                writers = {}
                self._sessions[session_id] = writers

            writer = writers.get(label)
            if writer is None:
                writer = _SessionWavWriter(
                    self._session_directory(session_id),
                    label,
                    self.sample_rate,
                    self.channels,
                    self.max_file_duration
                )
                writers[label] = writer

            writer.write(samples)

            self._sweep(now)

        except Exception:
            logger.error("SessionAudioRecorder failed to write audio", exc_info=True)

    def _sweep(self, now: float):
        # Opportunistic safety net: finalize sessions that ended without cleanup
        if now - self._last_sweep < min(60.0, self.session_ttl):
            return
        self._last_sweep = now
        for session_id, last_activity in list(self._last_activity.items()):
            if now - last_activity >= self.session_ttl:
                logger.warning(f"SessionAudioRecorder finalizing idle session: {session_id}")
                self.finalize_session(session_id)

    def finalize_session(self, session_id: str):
        """Close and flush all files for the session. Called automatically via
        the detector's delete_session / finalize_session."""
        writers = self._sessions.pop(session_id, None)
        self._last_activity.pop(session_id, None)
        if writers:
            for writer in writers.values():
                writer.close()

    def close(self):
        """Finalize all sessions."""
        for session_id in list(self._sessions.keys()):
            self.finalize_session(session_id)
