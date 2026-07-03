import logging
import unicodedata
from typing import Any, Iterable, Optional

from .base import TurnEndDecision, TurnEndGate

logger = logging.getLogger(__name__)


DEFAULT_JA_FILLERS = [
    "あ",
    "あの",
    "あのね",
    "あのさ",
    "え",
    "ええ",
    "えー",
    "えっと",
    "えと",
    "ええと",
    "えーと",
    "うーん",
    "ん",
    "んー",
    "まあ",
    "その",
    "なんか",
]

DEFAULT_EN_FILLERS = [
    "uh",
    "um",
    "umm",
    "erm",
    "er",
    "ah",
    "uhh",
    "hmm",
    "hmmm",
    "well",
    "like",
    "you know",
    "i mean",
]


class FillerOnlyTurnEndGate(TurnEndGate):
    """Wait longer when the recognized text is only a filler phrase."""

    def __init__(
        self,
        *,
        fillers: Optional[Iterable[str]] = None,
        timeout: Optional[float] = 5.0,
        no_text_should_end: bool = True,
        debug: bool = False,
    ):
        self.fillers = list(fillers) if fillers is not None else list(DEFAULT_JA_FILLERS + DEFAULT_EN_FILLERS)
        self.normalized_fillers = {self.normalize_text(filler) for filler in self.fillers}
        self.timeout = timeout
        self.no_text_should_end = no_text_should_end
        self.debug = debug

    @staticmethod
    def normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).lower()
        chars = []
        for ch in normalized:
            category = unicodedata.category(ch)
            if category[0] in {"P", "S", "Z"}:
                continue
            chars.append(ch)
        return "".join(chars).rstrip("ーｰ〜~")

    async def should_end_turn(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        channels: int,
        recorded_duration: float,
        silence_duration: float,
        session_id: str,
        text: Optional[str] = None,
        session: Any = None,
    ) -> TurnEndDecision:
        normalized_text = self.normalize_text(text or "")
        if not normalized_text:
            should_end = self.no_text_should_end
            reason = "filler_no_text"
        else:
            should_end = normalized_text not in self.normalized_fillers
            reason = "filler_only" if not should_end else "not_filler_only"

        if self.debug:
            logger.info(
                "Filler Turn: %s session=%s, text=%r, normalized=%r, timeout=%s",
                "PASS complete" if should_end else "WAIT filler_only",
                session_id,
                text,
                normalized_text,
                self.timeout,
            )

        return TurnEndDecision(
            should_end=should_end,
            confidence=1.0,
            reason=reason,
            timeout=None if should_end else self.timeout,
        )
