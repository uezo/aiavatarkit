"""One-shot turn-taking permission for application-selected sessions."""

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Callable, Optional

from .base import TurnTakingDecision, TurnTakingGate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionTurnAllowance:
    expires_at: float
    reason: str = "session_allow"


class SessionAllowTurnTakingGate(TurnTakingGate):
    """Allow the next evaluation of this gate for explicitly selected sessions.

    ``allow()`` arms one permission, consumed by ``should_take_turn()``. Without
    an active permission this gate returns None, letting a manager try its
    next child. Put this gate first to accept an expected answer before Jev.

    This is the next evaluation of this gate, not necessarily the next user
    utterance. Standalone automatic bypasses and earlier accepting gates,
    including TurnTakingBypassGate, do not call this classifier, so they do not
    consume its permission. Use an expiry
    appropriate for the expected answer and call ``release()`` when the answer
    is handled or the connection closes. Manager session cleanup does not clear
    child permissions. Provider resources and background tasks are not created.
    """

    def __init__(
        self, *, default_expires_in: float = 300.0,
        response_end_grace_seconds: float = 0.0,
        skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
        bypass_enabled: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            response_end_grace_seconds=response_end_grace_seconds,
            skip_condition=skip_condition, bypass_enabled=bypass_enabled, debug=debug,
        )
        default_expires_in = float(default_expires_in)
        if not math.isfinite(default_expires_in) or default_expires_in <= 0:
            raise ValueError("default_expires_in must be a finite number greater than 0")
        self.default_expires_in = default_expires_in
        self._allowances: dict[str, SessionTurnAllowance] = {}
        self._lock = threading.Lock()

    def allow(
        self, session_id: str, *, expires_in: Optional[float] = None,
        reason: str = "session_allow",
    ) -> None:
        """Allow one future gate evaluation, replacing any pending permission.

        ``expires_in`` is measured in seconds from this call and defaults to
        ``default_expires_in``. Repeated calls replace, rather than queue, it.
        """
        if not session_id:
            raise ValueError("session_id must not be empty")
        expires_in = self.default_expires_in if expires_in is None else float(expires_in)
        if not math.isfinite(expires_in) or expires_in <= 0:
            raise ValueError("expires_in must be a finite number greater than 0")
        now = time.monotonic()
        allowance = SessionTurnAllowance(expires_at=now + expires_in, reason=reason)
        with self._lock:
            expired = [
                key for key, value in self._allowances.items()
                if value.expires_at <= now
            ]
            for key in expired:
                self._allowances.pop(key, None)
            self._allowances[session_id] = allowance

    def release(self, session_id: str) -> None:
        """Clear any pending permission, including on answer handling/disconnect."""
        with self._lock:
            self._allowances.pop(session_id, None)

    def reset_session(self, session_id: str) -> None:
        self.release(session_id)

    def get_allowance(self, session_id: str) -> Optional[SessionTurnAllowance]:
        """Inspect an immutable, unexpired permission without consuming it."""
        now = time.monotonic()
        with self._lock:
            allowance = self._allowances.get(session_id)
            if allowance is None or allowance.expires_at <= now:
                self._allowances.pop(session_id, None)
                return None
            return allowance

    async def should_take_turn(
        self, user_text: Optional[str], assistant_spoken_text: Optional[str], *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
        **kwargs,
    ) -> TurnTakingDecision:
        now = time.monotonic()
        with self._lock:
            allowance = self._allowances.pop(session_id, None)
        if allowance is None:
            decision = TurnTakingDecision(None, None, "session_allow_inactive")
        elif allowance.expires_at <= now:
            decision = TurnTakingDecision(None, None, "session_allow_expired")
        else:
            decision = TurnTakingDecision(True, None, allowance.reason)
        if self.debug:
            logger.info(
                "Session Allow Turn: session=%s, should_take_turn=%s, reason=%s",
                session_id, decision.should_take_turn, decision.reason,
            )
        return decision
