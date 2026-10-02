"""Provider-independent contract for taking the conversational floor."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .session import TurnTakingSession

logger = logging.getLogger(__name__)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class TurnTakingDecision:
    """Decision with an optional probability of taking the turn, not of its label."""

    should_take_turn: bool
    probability: Optional[float]
    reason: str


class TurnTakingGate(ABC):
    """Base for interchangeable turn-taking classifiers.

    Implement ``should_take_turn()`` to classify user text against the spoken assistant
    prefix. Return ``probability=None`` when the implementation has no score.
    Expected provider failures should allow the turn with a diagnostic reason;
    cancellation must propagate instead of becoming an allowed turn.

    The gate may be shared between sessions. Keep mutable conversation and
    playback state in the session returned by ``create_session()``. Each owner
    must await its session's ``aclose()``; that does not close provider resources.

    ``skip_condition(text, recorded_duration)`` is synchronous and skips
    classification only when it returns True. An explicit condition overrides
    each session's default; use a condition returning False to disable skipping.

    ``get_session(create=True)`` registers managed state by session ID, which
    must uniquely identify each connection when sharing this gate.
    ``close_session()`` removes and closes that state; await the returned
    session's ``aclose()`` when cancellation cleanup must finish before continuing.
    """

    def __init__(
        self, *, response_end_grace_seconds: float = 0.0,
        skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        debug: bool = False,
    ):
        if not _finite_number(response_end_grace_seconds) or response_end_grace_seconds < 0:
            raise ValueError("response_end_grace_seconds must be a finite number greater than or equal to 0")
        self.response_end_grace_seconds = response_end_grace_seconds
        if skip_condition is not None and not callable(skip_condition):
            raise ValueError("skip_condition must be callable or None")
        self.skip_condition = skip_condition
        self.debug = debug
        self._sessions: dict[str, "TurnTakingSession"] = {}

    def create_session(
        self, session_id: str, *,
        default_skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
    ) -> "TurnTakingSession":
        """Create unregistered, caller-owned playback state with an optional default."""
        from .session import TurnTakingSession

        return TurnTakingSession(self, session_id, default_skip_condition=default_skip_condition)

    def get_session(
        self, session_id: str, *, create: bool = False,
        default_skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
    ) -> Optional["TurnTakingSession"]:
        """Get registered state; create missing or closed state only when requested."""
        session = self._sessions.get(session_id)
        if create and (session is None or session.is_closed):
            session = self.create_session(session_id, default_skip_condition=default_skip_condition)
            self._sessions[session_id] = session
        return session

    def is_current_session(
        self, session_id: str, session: Optional["TurnTakingSession"],
    ) -> bool:
        """Whether this is the registered, open helper for the connection ID."""
        return (
            session is not None and self._sessions.get(session_id) is session
            and not session.is_closed
        )

    def close_session(
        self, session_id: str,
    ) -> Optional["TurnTakingSession"]:
        """Remove and synchronously close state; return it for optional async cleanup."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
        return session

    def handle_playback_event(
        self, session_id: str, *, event: Optional[str] = None,
        playback_id: Optional[str] = None, text: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        transaction_id: Optional[str] = None, completed: bool = False,
        **_metadata: Any,
    ) -> bool:
        """Apply a playback event to an existing, open session.

        Accept explicit fields or ``handle_playback_event(session_id, **metadata)``;
        unrelated metadata is ignored. Never create or reopen a session here.
        Return the playback helper's result, or False for missing/closed sessions
        and unknown events. Only literal True marks an end as completed.
        """
        session = self.get_session(session_id, create=False)
        if session is None or session.is_closed:
            reason = "session_missing_or_closed"
        elif event == "start":
            return session.start_playback(
                playback_id, text, duration_seconds, transaction_id=transaction_id,
            )
        elif event == "final":
            return session.mark_response_final(playback_id, transaction_id=transaction_id)
        elif event == "end":
            return session.end_playback(playback_id, completed=completed is True)
        else:
            reason = "unknown_playback_event"
        if self.debug:
            logger.info(
                "Turn Taking Playback: session=%s, event=%r, action=ignore, reason=%s",
                session_id, event, reason,
            )
        return False

    async def evaluate(
        self, session_id: str, user_text: Optional[str], *,
        recorded_duration: float = 0.0, recording_id: Optional[str] = None,
        speech_end_at: Optional[datetime] = None,
    ) -> TurnTakingDecision:
        """Evaluate existing state, rejecting missing or obsolete session results."""
        session = self.get_session(session_id)
        if not self.is_current_session(session_id, session):
            decision = TurnTakingDecision(False, None, "session_missing_or_closed")
            self._log_evaluation(session_id, recording_id, "discard", decision.reason)
            return decision
        task = asyncio.current_task()
        try:
            try:
                decision = await session.should_take_turn(
                    user_text, recorded_duration=recorded_duration,
                    recording_id=recording_id, speech_end_at=speech_end_at,
                )
                # Read these inside the failure boundary, as malformed provider
                # results follow the same fail-open policy as provider errors.
                should_take_turn, reason = decision.should_take_turn, decision.reason
            except Exception as exc:
                reason = f"turn_taking_error:{type(exc).__name__}"
                decision = TurnTakingDecision(True, None, reason)
                should_take_turn = True
                logger.warning(
                    "Turn-taking gate failed: session=%s, error_type=%s",
                    session_id, type(exc).__name__,
                )
            if task is not None and task.cancelling():
                raise asyncio.CancelledError
            if not self.is_current_session(session_id, session):
                decision = TurnTakingDecision(False, None, "session_replaced_or_closed")
                self._log_evaluation(session_id, recording_id, "discard", decision.reason)
                return decision
            self._log_evaluation(session_id, recording_id, "pass" if should_take_turn else "discard", reason)
            return decision
        except asyncio.CancelledError:
            self._log_evaluation(session_id, recording_id, "discard", "cancelled")
            raise

    def _log_evaluation(
        self, session_id: str, recording_id: Optional[str], action: str, reason: str,
    ) -> None:
        if self.debug:
            logger.info(
                "Turn Taking Gate: session=%s, recording_id=%s, action=%s, reason=%s",
                session_id, recording_id, action, reason,
            )

    @abstractmethod
    async def should_take_turn(
        self,
        user_text: Optional[str],
        assistant_spoken_text: Optional[str],
        *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
    ) -> TurnTakingDecision:
        """Classify using the spoken prefix and known text, including unread parts.

        ``assistant_full_text`` covers started chunks of the current response;
        it does not imply that the user heard them or that the response is complete.
        """
        pass
