"""Explicit policies for allowing input without later turn-taking classifiers."""

from typing import Callable, Optional

from .base import TurnTakingDecision, TurnTakingGate
from .session import PlaybackEstimate


class TurnTakingBypassGate(TurnTakingGate):
    """Allow idle playback, missing text, or inputs selected by a skip policy.

    Put required checks, such as wakeword detection, before this gate. A match
    returns True and ends the chain; otherwise None lets later classifiers run.
    An explicit synchronous ``skip_condition(text, recorded_duration)`` replaces
    the session's VAD-supplied default. Returning False disables duration skipping.
    ``response_end_grace_seconds`` also allows input near the confirmed final
    chunk's natural end; zero disables this policy.
    """

    def __init__(
        self, *, response_end_grace_seconds: float = 0.0,
        skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        debug: bool = False,
    ):
        super().__init__(
            response_end_grace_seconds=response_end_grace_seconds,
            skip_condition=skip_condition, bypass_enabled=False, debug=debug,
        )

    async def should_take_turn(
        self, user_text: Optional[str], assistant_spoken_text: Optional[str], *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
        playback: Optional[PlaybackEstimate] = None,
        recorded_duration: float = 0.0,
        default_skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        **kwargs,
    ) -> TurnTakingDecision:
        return self._get_bypass_decision(
            user_text, playback=playback, recorded_duration=recorded_duration,
            default_skip_condition=default_skip_condition,
        ) or TurnTakingDecision(None, None, "turn_taking_no_bypass")
