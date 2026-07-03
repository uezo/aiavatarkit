from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TurnEndDecision:
    should_end: bool
    confidence: Optional[float] = None
    reason: Optional[str] = None


class TurnEndGate(ABC):
    """Optional confirmation gate for VAD turn-end candidates.

    VAD implementations call this only after their normal turn-end candidate
    condition is met, such as silence_duration_threshold for Silero.
    """

    @abstractmethod
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
        pass
