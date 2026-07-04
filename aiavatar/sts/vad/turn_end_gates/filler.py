import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Union

from .base import TurnEndDecision, TurnEndGate, TurnEndGateContext

logger = logging.getLogger(__name__)

FillerMatch = Literal["exact", "suffix"]


@dataclass(frozen=True)
class FillerPhrase:
    text: str
    match: FillerMatch = "exact"
    timeout: Optional[float] = None


DEFAULT_JA_FILLERS = [
    "あ",
    FillerPhrase("あの", match="suffix"),
    FillerPhrase("あのね", match="suffix"),
    FillerPhrase("あのさ", match="suffix"),
    "え",
    FillerPhrase("ええ", match="suffix"),
    FillerPhrase("えー", match="suffix"),
    FillerPhrase("えっと", match="suffix"),
    FillerPhrase("えと", match="suffix"),
    FillerPhrase("ええと", match="suffix"),
    FillerPhrase("えーと", match="suffix"),
    FillerPhrase("うーん", match="suffix"),
    "ん",
    FillerPhrase("んー", match="suffix"),
    FillerPhrase("まあ", match="suffix"),
    FillerPhrase("その", match="suffix"),
    FillerPhrase("なんか", match="suffix"),
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
    """Wait longer when recognized text is only or ends with a filler phrase."""

    def __init__(
        self,
        *,
        name: str = "filler",
        fillers: Optional[Iterable[Union[str, FillerPhrase]]] = None,
        timeout: Optional[float] = 5.0,
        no_text_should_end: bool = True,
        debug: bool = False,
    ):
        self.name = name
        self.fillers = list(fillers) if fillers is not None else list(DEFAULT_JA_FILLERS + DEFAULT_EN_FILLERS)
        self.normalized_exact_fillers = {}
        self.normalized_suffix_fillers = {}
        for filler in self.fillers:
            phrase = self._coerce_filler_phrase(filler)
            normalized = self.normalize_text(phrase.text)
            if not normalized:
                continue
            if phrase.match == "exact":
                self.normalized_exact_fillers[normalized] = phrase.timeout
            elif phrase.match == "suffix":
                self.normalized_exact_fillers[normalized] = phrase.timeout
                if len(normalized) >= 2:
                    self.normalized_suffix_fillers[normalized] = phrase.timeout
            else:
                raise ValueError("FillerPhrase.match must be 'exact' or 'suffix'")
        self.normalized_suffix_fillers_by_length = sorted(
            self.normalized_suffix_fillers,
            key=len,
            reverse=True,
        )
        self.timeout = timeout
        self.no_text_should_end = no_text_should_end
        self.debug = debug

    def _coerce_filler_phrase(self, filler: Union[str, FillerPhrase]) -> FillerPhrase:
        if isinstance(filler, FillerPhrase):
            return filler
        return FillerPhrase(str(filler), match="exact")

    def _resolve_timeout(self, phrase_timeout: Optional[float]) -> Optional[float]:
        return self.timeout if phrase_timeout is None else phrase_timeout

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
        context: Optional[TurnEndGateContext] = None,
    ) -> TurnEndDecision:
        normalized_text = self.normalize_text(text or "")
        phrase_timeout = None
        if not normalized_text:
            should_end = self.no_text_should_end
            reason = "filler_no_text"
        elif normalized_text in self.normalized_exact_fillers:
            should_end = False
            reason = "filler_only"
            phrase_timeout = self.normalized_exact_fillers[normalized_text]
        else:
            should_end = True
            reason = "not_filler"
            for filler in self.normalized_suffix_fillers_by_length:
                if normalized_text.endswith(filler):
                    should_end = False
                    reason = "trailing_filler"
                    phrase_timeout = self.normalized_suffix_fillers[filler]
                    break

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
            timeout=None if should_end else self._resolve_timeout(phrase_timeout),
        )
