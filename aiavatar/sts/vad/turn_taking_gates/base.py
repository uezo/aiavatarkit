"""Provider-independent contract for taking the conversational floor."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .session import PlaybackEstimate, TurnTakingSession

logger = logging.getLogger(__name__)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_evaluation_current(is_current: Optional[Callable[[], bool]]) -> None:
    """Reject cancelled or superseded work, including providers that uncancel."""
    task = asyncio.current_task()
    if (task is not None and task.cancelling()) or (is_current is not None and not is_current()):
        raise asyncio.CancelledError


@dataclass(frozen=True)
class TurnTakingDecision:
    """True allows, False blocks, and None continues to the next gate."""

    should_take_turn: Optional[bool]
    probability: Optional[float]
    reason: str


class TurnTakingGate(ABC):
    """Base for interchangeable turn-taking classifiers.

    Implement ``should_take_turn()`` to classify user text against the spoken assistant
    prefix. Return ``probability=None`` when the implementation has no score.
    Return None to defer to the next gate; True and False stop the chain.
    Ordinary provider failures allow the turn with a diagnostic reason.
    Gates that require a positive match must handle expected failures with False;
    cancellation must propagate instead of becoming an allowed turn.

    The gate may be shared between sessions. Keep mutable conversation and
    playback state in the session returned by ``create_session()``. Each owner
    must await its session's ``aclose()``; that does not close provider resources.

    Standalone managed evaluation applies playback and duration bypasses by
    default. An explicit ``skip_condition`` overrides the VAD-supplied duration
    condition; ``bypass_enabled=False`` disables all automatic bypasses.
    Wakeword gates disable automatic bypasses. Managers infer this setting from
    the presence of an explicit ``TurnTakingBypassGate`` unless overridden;
    children's standalone bypass settings are not applied.

    ``get_session(create=True)`` registers managed state by session ID, which
    must uniquely identify each connection when sharing this gate.
    ``close_session()`` removes and closes that state; await the returned
    session's ``aclose()`` when cancellation cleanup must finish before continuing.
    """

    def __init__(
        self, *, response_end_grace_seconds: float = 0.0,
        skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        bypass_enabled: bool = True,
        debug: bool = False,
    ):
        if not _finite_number(response_end_grace_seconds) or response_end_grace_seconds < 0:
            raise ValueError("response_end_grace_seconds must be a finite number greater than or equal to 0")
        if skip_condition is not None and not callable(skip_condition):
            raise ValueError("skip_condition must be callable or None")
        self.response_end_grace_seconds = response_end_grace_seconds
        self.skip_condition = skip_condition
        self.bypass_enabled = bypass_enabled
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

    def _get_bypass_decision(
        self, user_text: Optional[str], *,
        playback: Optional["PlaybackEstimate"] = None,
        recorded_duration: float = 0.0,
        default_skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
    ) -> Optional[TurnTakingDecision]:
        """Shared bypass policy for standalone gates and explicit bypass gates."""
        if not user_text or not user_text.strip():
            return TurnTakingDecision(True, None, "turn_taking_no_text")

        condition = self.skip_condition if self.skip_condition is not None else default_skip_condition
        if condition is not None:
            try:
                if condition(user_text, recorded_duration) is True:
                    return TurnTakingDecision(True, None, "turn_take_skipped")
            except Exception as exc:
                logger.warning(
                    "Turn-taking skip condition failed (%s); skipping playback classification",
                    type(exc).__name__,
                )
                return TurnTakingDecision(True, None, "turn_taking_skip_condition_error")

        if playback is not None:
            if not playback.is_playing:
                return TurnTakingDecision(True, None, "playback_" + playback.reason)
            if (
                playback.reason == "playing" and playback.is_final_chunk
                and self.response_end_grace_seconds > 0
                and (
                    playback.remaining_seconds <= self.response_end_grace_seconds
                    or math.isclose(
                        playback.remaining_seconds, self.response_end_grace_seconds,
                        rel_tol=0.0, abs_tol=1e-9,
                    )
                )
            ):
                # Preserve the tolerance for rounding at the grace threshold.
                return TurnTakingDecision(True, None, "playback_response_end_grace")
            if not playback.assistant_spoken_text.strip():
                return TurnTakingDecision(True, None, "playback_empty_estimate")

        return None

    @abstractmethod
    async def should_take_turn(
        self,
        user_text: Optional[str],
        assistant_spoken_text: Optional[str],
        *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
        playback: Optional["PlaybackEstimate"] = None,
        recorded_duration: float = 0.0,
        default_skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        is_current: Optional[Callable[[], bool]] = None,
    ) -> TurnTakingDecision:
        """Classify using the spoken prefix and known text, including unread parts.

        ``assistant_full_text`` covers started chunks of the current response;
        it does not imply that the user heard them or that the response is complete.
        ``playback`` is a snapshot at speech end. ``recorded_duration`` and the
        session's ``default_skip_condition`` let a bypass gate apply VAD policy.
        ``is_current`` lets composite gates reject superseded input between children.
        Implementations may accept unused context through ``**kwargs``.
        """
        pass
