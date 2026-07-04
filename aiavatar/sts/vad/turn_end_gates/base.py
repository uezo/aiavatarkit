from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TurnEndDecision:
    should_end: bool
    confidence: Optional[float] = None
    reason: Optional[str] = None
    timeout: Optional[float] = None
    pending: bool = False


@dataclass
class TurnEndGateContext:
    decisions: Dict[str, TurnEndDecision] = field(default_factory=dict)

    def add_decision(self, gate_name: str, decision: TurnEndDecision):
        self.decisions[gate_name] = decision

    def get_decision(self, gate_name: str) -> Optional[TurnEndDecision]:
        return self.decisions.get(gate_name)

    def is_waiting(self, gate_name: str) -> bool:
        decision = self.get_decision(gate_name)
        return decision is not None and not decision.should_end


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
        context: Optional[TurnEndGateContext] = None,
    ) -> TurnEndDecision:
        pass
