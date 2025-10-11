from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import json
import logging
import os
from typing import Dict, Optional, Any, Tuple, Iterable, List
import uuid
import numpy as np
try:
    from resemblyzer import VoiceEncoder, preprocess_wav  # pip install resemblyzer
except:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single candidate item from Top-K search."""
    speaker_id: str
    similarity: float
    metadata: Dict[str, Any]
    is_new: bool = False  # For chosen only; candidates from store are always existing (False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchTopKResult:
    """
    Top-K matching outcome with decision folded into chosen.is_new.
    - chosen: Top-1 if matched; or newly registered id (similarity=1.0, empty metadata) if not matched.
              chosen.is_new == True iff a new speaker was registered.
    - candidates: nearest existing neighbors for reference.
        * When chosen.is_new == False: excludes chosen (starts from rank-2).
        * When chosen.is_new == True: includes the original Top-1 as candidates[0].
    """
    chosen: Candidate
    candidates: List[Candidate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen": self.chosen.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
        }


class BaseSpeakerStore(ABC):
    """Abstract storage for speaker embeddings and metadata."""

    @abstractmethod
    def upsert(self, speaker_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert or update an L2-normalized embedding and its metadata."""
        ...

    @abstractmethod
    def get(self, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        ...

    @abstractmethod
    def set_metadata(self, speaker_id: str, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def get_metadata(self, speaker_id: str, key: str, default: Any = None) -> Any:
        ...

    @abstractmethod
    def all_items(self) -> Iterable[Tuple[str, np.ndarray, Dict[str, Any]]]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def topk_similarity(self, q_norm: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Return top-k (speaker_id, cosine_similarity) against normalized query q_norm.
        All stored embeddings must be L2-normalized (cosine = dot).
        """
        ...

    # Optional file persistence (in-memory only)
    def save_to_file(self, path: str) -> None:
        raise NotImplementedError

    def load_from_file(self, path: str) -> None:
        raise NotImplementedError


class InMemoryStore(BaseSpeakerStore):
    """
    In-memory store with BLAS-backed top-k via a dense (N, D) matrix.
    Provides optional file persistence ({base}.npz for ids+embeddings, {base}.json for metadata).
    """
    def __init__(self, data_path: Optional[str] = None):
        # {sid: {"embedding": np.ndarray(float32, L2-normalized), "metadata": dict}}
        self._store: Dict[str, Dict[str, Any]] = {}
        # Fast search cache
        self._id_list: List[str] = []                 # Row order aligned with _emb_matrix
        self._emb_matrix: Optional[np.ndarray] = None # (N, D) float32
        self.data_path = data_path
        self.load_from_file()

    def upsert(self, speaker_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        is_new = speaker_id not in self._store
        if is_new:
            self._store[speaker_id] = {"embedding": embedding.astype(np.float32, copy=False),
                                       "metadata": (metadata or {})}
            self._id_list.append(speaker_id)
        else:
            self._store[speaker_id]["embedding"] = embedding.astype(np.float32, copy=False)
            if metadata:
                self._store[speaker_id]["metadata"].update(metadata)
        self._emb_matrix = None  # Invalidate cache
        self.save_to_file()

    def get(self, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        slot = self._store.get(speaker_id)
        if slot is None:
            raise KeyError(f"Unknown speaker_id: {speaker_id}")
        return slot["embedding"], slot["metadata"]

    def set_metadata(self, speaker_id: str, key: str, value: Any) -> None:
        slot = self._store.get(speaker_id)
        if slot is None:
            raise KeyError(f"Unknown speaker_id: {speaker_id}")
        slot["metadata"][key] = value
        self.save_to_file()

    def get_metadata(self, speaker_id: str, key: str, default: Any = None) -> Any:
        slot = self._store.get(speaker_id)
        if slot is None:
            raise KeyError(f"Unknown speaker_id: {speaker_id}")
        return slot["metadata"].get(key, default)

    def all_items(self) -> Iterable[Tuple[str, np.ndarray, Dict[str, Any]]]:
        for sid, payload in self._store.items():
            yield sid, payload["embedding"], payload["metadata"]

    def count(self) -> int:
        return len(self._store)

    def _ensure_matrix(self) -> None:
        """Rebuild (N, D) matrix and aligned id list."""
        if self._emb_matrix is not None:
            return
        if not self._store:
            self._emb_matrix = np.zeros((0, 0), dtype=np.float32)
            return

        new_ids: List[str] = []
        rows: List[np.ndarray] = []
        # Keep stable order; drop missing
        for sid in self._id_list:
            payload = self._store.get(sid)
            if payload is None:
                continue
            new_ids.append(sid)
            rows.append(payload["embedding"])
        # Append any newly inserted ids that were not in _id_list yet
        for sid, payload in self._store.items():
            if sid not in new_ids:
                new_ids.append(sid)
                rows.append(payload["embedding"])

        self._id_list = new_ids
        self._emb_matrix = np.vstack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, 0), np.float32)

    def topk_similarity(self, q_norm: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Compute top-k using a single matvec and argpartition."""
        self._ensure_matrix()
        if self._emb_matrix.size == 0:
            raise RuntimeError("InMemoryStore is empty.")
        sims = self._emb_matrix @ q_norm  # (N,)
        k = max(1, min(k, sims.shape[0]))
        # Partial selection (O(N)) then sort k candidates descending
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(self._id_list[i], float(sims[i])) for i in idx]

    def save_to_file(self) -> None:
        """Write {path}.npz (ids + embeddings) and {path}.json (metadata)."""
        if not self.data_path:
            return

        self._emb_matrix = None  # Force rebuild
        self._ensure_matrix()
        metas = {sid: self._store[sid]["metadata"] for sid in self._id_list}
        if self._emb_matrix is None or self._emb_matrix.size == 0:
            np.savez_compressed(f"{self.data_path}.npz",
                                ids=np.array([], dtype=object),
                                embeddings=np.zeros((0, 0), dtype=np.float32))
        else:
            np.savez_compressed(f"{self.data_path}.npz",
                                ids=np.array(self._id_list, dtype=object),
                                embeddings=self._emb_matrix.astype(np.float32, copy=False))
        with open(f"{self.data_path}.json", "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)

    def load_from_file(self) -> None:
        """Load from {path}.npz and {path}.json and rebuild cache."""
        if not self.data_path:
            return

        npz_path = f"{self.data_path}.npz"
        json_path = f"{self.data_path}.json"
        if not os.path.exists(npz_path) or not os.path.exists(json_path):
            self._store.clear()
            self._id_list = []
            self._emb_matrix = np.zeros((0, 0), dtype=np.float32)
            return

        data = np.load(npz_path, allow_pickle=True)
        ids = data["ids"].tolist()
        embs = data["embeddings"]  # (N, D)
        with open(json_path, "r", encoding="utf-8") as f:
            metas = json.load(f)

        self._store.clear()
        self._id_list = []
        for i, sid in enumerate(ids):
            emb = np.asarray(embs[i], dtype=np.float32)
            md = metas.get(sid, {})
            self._store[sid] = {"embedding": emb, "metadata": md}
            self._id_list.append(sid)

        self._emb_matrix = np.asarray(embs, dtype=np.float32) if len(ids) > 0 else np.zeros((0, 0), np.float32)


class SpeakerRegistry:
    def __init__(self, match_threshold: float = 0.72, store: Optional[BaseSpeakerStore] = None, data_path: Optional[str] = None):
        self.match_threshold = float(match_threshold)
        try:
            self._enc = VoiceEncoder()
        except:
            self._enc = None
            logger.warning("SpeakerRegistry doesn't work because resemblyzer is not installed.")
        self._store: BaseSpeakerStore = store or InMemoryStore(data_path=data_path)

    def match_topk_from_embedding(
        self,
        embedding: np.ndarray,
        k: int = 3,
        candidate_min_sim: float = 0.0,
    ) -> MatchTopKResult:
        """
        Return Top-K nearest speakers and encode decision as chosen.is_new.
        Threshold applies ONLY to Top-1. Candidates are for reference.
        """
        q = self._normalize(embedding)

        if self._store.count() == 0:
            # First entry: register and return no candidates
            new_sid = self._new_speaker_id()
            self._store.upsert(new_sid, q, metadata={})
            return MatchTopKResult(
                chosen=Candidate(new_sid, 1.0, {}, is_new=True),
                candidates=[],
            )

        topk: List[Tuple[str, float]] = self._store.topk_similarity(q, max(1, k))
        best_sid, best_sim = topk[0]

        if best_sim >= self.match_threshold:
            # Match to existing: chosen = Top-1 existing
            _, md = self._store.get(best_sid)
            chosen = Candidate(best_sid, float(best_sim), dict(md), is_new=False)
            others = [
                Candidate(sid, float(sim), self._store.get(sid)[1], is_new=False)
                for (sid, sim) in topk[1:]
                if sim >= candidate_min_sim
            ]
            return MatchTopKResult(chosen=chosen, candidates=others)

        # Below threshold: register new; candidates include original Top-1 at index 0
        new_sid = self._new_speaker_id()
        self._store.upsert(new_sid, q, metadata={})
        refs = [
            Candidate(sid, float(sim), self._store.get(sid)[1], is_new=False)
            for (sid, sim) in topk
            if sim >= candidate_min_sim
        ]
        return MatchTopKResult(
            chosen=Candidate(new_sid, 1.0, {}, is_new=True),
            candidates=refs,
        )

    def match_topk_from_pcm(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        k: int = 3,
        candidate_min_sim: float = 0.0,
    ) -> MatchTopKResult:
        """Top-K matching directly from raw PCM (int16 mono)."""
        emb = self._embed_pcm(audio_bytes, sample_rate)
        return self.match_topk_from_embedding(
            emb,
            k=k,
            candidate_min_sim=candidate_min_sim,
        )

    def set_metadata(self, speaker_id: str, key: str, value: Any) -> None:
        self._store.set_metadata(speaker_id, key, value)

    def get_metadata(self, speaker_id: str, key: str, default: Any = None) -> Any:
        return self._store.get_metadata(speaker_id, key, default)

    def _embed_pcm(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        """Convert raw PCM (int16 mono) to float32 waveform, preprocess, and embed."""
        wav_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
        wav_f32 = wav_i16.astype(np.float32, copy=False) / 32768.0
        wav_proc = preprocess_wav(wav_f32, source_sr=sample_rate)
        emb = self._enc.embed_utterance(wav_proc)
        return emb.astype(np.float32, copy=False)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32, copy=False)
        n = np.linalg.norm(v) + 1e-9
        return v / n

    @staticmethod
    def _new_speaker_id() -> str:
        return f"spk_{uuid.uuid4().hex[:12]}"
