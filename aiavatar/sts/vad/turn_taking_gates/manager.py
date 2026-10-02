"""Compose turn-taking gates with ordered, short-circuit acceptance."""

import logging
from typing import Callable, Optional

from .base import TurnTakingDecision, TurnTakingGate

logger = logging.getLogger(__name__)


class TurnTakingGateManager(TurnTakingGate):
    """A gate that accepts the first positive decision from its child gates.

    Children run in registration order. False continues to the next child;
    only when every child declines is the input rejected. An empty manager
    allows the turn.

    Use this manager wherever a single ``TurnTakingGate`` is accepted. Its
    session owns playback estimation and cancellation for the entire sequence.
    Children receive the same estimated spoken text and known full text through their raw
    ``should_take_turn()`` methods; their sessions and ``evaluate()`` are unused.

    Configure ``skip_condition`` and ``response_end_grace_seconds`` on the
    manager. Child values are ignored, even when the manager's condition is
    unset; in that case the session's VAD-supplied default applies. Provider
    settings such as Jev's threshold, request timeout, and debug remain active.

    Exceptions propagate to the inherited ``evaluate()`` failure fallback,
    which allows the input. Cancellation propagates. Provider timeouts remain
    per child; there is no additional timeout for the whole sequence.
    """

    def __init__(
        self, gates: Optional[list[TurnTakingGate]] = None, *,
        response_end_grace_seconds: float = 0.0,
        skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        debug: bool = False,
    ):
        super().__init__(
            response_end_grace_seconds=response_end_grace_seconds,
            skip_condition=skip_condition,
            debug=debug,
        )
        self.gates = list(gates) if gates is not None else []

    async def should_take_turn(
        self, user_text: Optional[str], assistant_spoken_text: Optional[str], *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
    ) -> TurnTakingDecision:
        if not self.gates:
            return TurnTakingDecision(True, None, "no_gates")

        for index, gate in enumerate(self.gates, start=1):
            try:
                decision = await gate.should_take_turn(
                    user_text, assistant_spoken_text, session_id=session_id,
                    assistant_full_text=assistant_full_text,
                )
                if self.debug:
                    logger.info(
                        "Turn Taking Gate Manager: session=%s, gate_index=%s, gate=%s, "
                        "should_take_turn=%s, probability=%s, reason=%s",
                        session_id, index, type(gate).__name__,
                        decision.should_take_turn, decision.probability, decision.reason,
                    )
                if decision.should_take_turn:
                    return decision
            except Exception as exc:
                if self.debug:
                    logger.warning(
                        "Turn Taking Gate Manager: session=%s, gate_index=%s, gate=%s, error_type=%s",
                        session_id, index, type(gate).__name__, type(exc).__name__,
                    )
                raise

        return TurnTakingDecision(False, None, "all_gates_declined")
