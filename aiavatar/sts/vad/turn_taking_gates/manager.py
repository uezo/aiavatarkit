"""Compose turn-taking gates with ordered allow/continue/block decisions."""

import logging
from typing import Callable, Optional, TYPE_CHECKING

from .base import TurnTakingDecision, TurnTakingGate, _check_evaluation_current
from .bypass import TurnTakingBypassGate

if TYPE_CHECKING:
    from .session import PlaybackEstimate

logger = logging.getLogger(__name__)


class TurnTakingGateManager(TurnTakingGate):
    """Run gates until the first True (allow) or False (block) decision.

    None advances to the next child. An empty manager or a chain of only None
    results defers to its caller; the outer session allows that input.

    The manager's session owns playback estimation and cancellation for the
    sequence. Children receive the same input and context; their sessions and
    ``evaluate()`` are unused. With ``bypass_enabled=None``, automatic bypasses
    are disabled if the initial, direct children include a ``TurnTakingBypassGate``;
    otherwise they are enabled. An explicit True or False overrides this choice.
    Put a bypass gate after required checks to control where playback and duration
    policies allow input, or set ``bypass_enabled=False`` to omit automatic bypasses.

    Exceptions propagate to the inherited ``evaluate()`` failure fallback,
    which allows the input. Cancellation propagates. Provider timeouts remain
    per child; there is no additional timeout for the whole sequence.
    """

    def __init__(
        self, gates: Optional[list[TurnTakingGate]] = None, *,
        response_end_grace_seconds: float = 0.0,
        skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        bypass_enabled: Optional[bool] = None,
        debug: bool = False,
    ):
        self.gates = list(gates) if gates is not None else []
        if bypass_enabled is None:
            bypass_enabled = not any(isinstance(gate, TurnTakingBypassGate) for gate in self.gates)
        super().__init__(
            response_end_grace_seconds=response_end_grace_seconds,
            skip_condition=skip_condition, bypass_enabled=bypass_enabled, debug=debug,
        )

    async def should_take_turn(
        self, user_text: Optional[str], assistant_spoken_text: Optional[str], *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
        playback: Optional["PlaybackEstimate"] = None,
        recorded_duration: float = 0.0,
        default_skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        is_current: Optional[Callable[[], bool]] = None,
    ) -> TurnTakingDecision:
        _check_evaluation_current(is_current)
        for index, gate in enumerate(self.gates, start=1):
            try:
                decision = await gate.should_take_turn(
                    user_text, assistant_spoken_text, session_id=session_id,
                    assistant_full_text=assistant_full_text,
                    playback=playback, recorded_duration=recorded_duration,
                    default_skip_condition=default_skip_condition,
                    is_current=is_current,
                )
                # A provider may swallow cancellation or even call uncancel().
                # Reject stale input before a later gate can consume permission.
                _check_evaluation_current(is_current)
                if self.debug:
                    logger.info(
                        "Turn Taking Gate Manager: session=%s, gate_index=%s, gate=%s, "
                        "should_take_turn=%s, probability=%s, reason=%s",
                        session_id, index, type(gate).__name__,
                        decision.should_take_turn, decision.probability, decision.reason,
                    )
                if decision.should_take_turn is not None:
                    return decision
            except Exception as exc:
                _check_evaluation_current(is_current)
                if self.debug:
                    logger.warning(
                        "Turn Taking Gate Manager: session=%s, gate_index=%s, gate=%s, error_type=%s",
                        session_id, index, type(gate).__name__, type(exc).__name__,
                    )
                raise

        return TurnTakingDecision(None, None, "all_gates_continued" if self.gates else "no_gates")
