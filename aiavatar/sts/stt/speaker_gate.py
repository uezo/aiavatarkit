import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav    # pip install resemblyzer

@dataclass
class MainSpeakerGateResult:
    accepted: bool
    confidence: Optional[float]
    main_locked: bool

    def to_dict(self):
        return asdict(self)


@dataclass
class SessionState:
    last_called_ts: float
    prev_emb: Optional[np.ndarray] = None   # last embedding (for 2-in-a-row lock)
    main_emb: Optional[np.ndarray] = None   # locked main speaker embedding


class MainSpeakerGate:
    def __init__(
        self,
        accept_threshold: float = 0.55,    # post-lock acceptance threshold (looser)
        pair_lock_threshold: float = 0.72, # pre-lock "two-in-a-row" threshold (stricter)
    ):
        self.accept_threshold = float(accept_threshold)
        self.pair_lock_threshold = float(pair_lock_threshold)
        self._enc = VoiceEncoder()
        self._sessions: Dict[str, SessionState] = {}

    async def evaluate(self, session_id: str, audio_bytes: bytes, sample_rate: int) -> MainSpeakerGateResult:
        # Get embeddings
        emb = await asyncio.to_thread(self._embed_pcm, audio_bytes, sample_rate)

        # Get or create session
        now = time.time()
        st = self._sessions.get(session_id)
        if st is None:
            st = SessionState(last_called_ts=now)
            self._sessions[session_id] = st
        st.last_called_ts = now

        # If main is already locked: decide by accept_threshold.
        if st.main_emb is not None:
            sim = self._cosine(st.main_emb, emb)
            accepted = sim >= self.accept_threshold
            return MainSpeakerGateResult(accepted=accepted, confidence=sim, main_locked=True)

        # Main not locked yet: always accept (pass-through).
        accepted = True
        confidence = None
        main_locked = False

        # First embedding observed -> store and pass.
        if st.prev_emb is None:
            st.prev_emb = emb
            return MainSpeakerGateResult(accepted=accepted, confidence=confidence, main_locked=main_locked)

        # Compare with previous embedding to check two-in-a-row lock.
        sim_prev = self._cosine(st.prev_emb, emb)
        if sim_prev >= self.pair_lock_threshold:
            # Lock main as the normalized mean of the two.
            main = self._normalize((st.prev_emb + emb) / 2.0)
            st.main_emb = main
            st.prev_emb = None
            # Optionally expose current similarity to the newly locked main.
            sim_main = self._cosine(main, emb)
            return MainSpeakerGateResult(accepted=True, confidence=sim_main, main_locked=True)
        else:
            # Not similar enough -> update prev and keep passing through.
            st.prev_emb = emb
            return MainSpeakerGateResult(accepted=accepted, confidence=confidence, main_locked=main_locked)

    def _embed_pcm(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        """
        Convert RAW PCM (int16, mono) bytes to float32 waveform, preprocess, and embed.
        """
        wav_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
        wav_f32 = wav_i16.astype(np.float32) / 32768.0
        wav_proc = preprocess_wav(wav_f32, source_sr=sample_rate)
        emb = self._enc.embed_utterance(wav_proc)
        return emb.astype(np.float32, copy=False)

    def delete_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def delete_sessions(self, max_age_seconds: float) -> int:
        now = time.time()
        to_delete = [sid for sid, st in self._sessions.items()
                     if now - st.last_called_ts > max_age_seconds]
        for sid in to_delete:
            self._sessions.pop(sid, None)
        return len(to_delete)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32, copy=False)
        v /= (np.linalg.norm(v) + 1e-9)
        return v

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = MainSpeakerGate._normalize(a)
        b = MainSpeakerGate._normalize(b)
        return float(np.dot(a, b))
