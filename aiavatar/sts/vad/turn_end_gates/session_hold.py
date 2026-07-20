import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import TurnEndDecision, TurnEndGate, TurnEndGateContext

logger = logging.getLogger(__name__)


@dataclass
class SessionTurnHold:
    timeout: float
    expires_at: float
    reason: str = "session_hold"


class SessionHoldTurnEndGate(TurnEndGate):
    """Hold the next turn-end candidate for explicitly requested sessions."""

    def __init__(
        self,
        *,
        name: str = "session_hold",
        default_expires_in: float = 300.0,
        debug: bool = False,
    ):
        default_expires_in = float(default_expires_in)
        if not math.isfinite(default_expires_in) or default_expires_in <= 0:
            raise ValueError("default_expires_in must be a finite number greater than 0")

        self.name = name
        self.default_expires_in = default_expires_in
        self.debug = debug
        self._holds: Dict[str, SessionTurnHold] = {}
        self._lock = threading.Lock()

    def hold(
        self,
        session_id: str,
        *,
        timeout: float,
        expires_in: Optional[float] = None,
        reason: str = "session_hold",
        replace: bool = True,
    ):
        """Hold the next turn-end candidate for a session.

        timeout is the number of seconds the manager may keep the turn open
        after the normal silence threshold. The hold is consumed when the next
        turn-end candidate is evaluated. expires_in limits how long the hold
        may wait for that candidate and defaults to default_expires_in.
        """
        if not session_id:
            raise ValueError("session_id must not be empty")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be a finite number greater than or equal to 0")

        expires_in = self.default_expires_in if expires_in is None else float(expires_in)
        if not math.isfinite(expires_in) or expires_in <= 0:
            raise ValueError("expires_in must be a finite number greater than 0")

        now = time.monotonic()

        new_hold = SessionTurnHold(
            timeout=timeout,
            expires_at=now + expires_in,
            reason=reason,
        )
        with self._lock:
            self._purge_expired_locked(now)
            if replace or session_id not in self._holds:
                self._holds[session_id] = new_hold
                return

            current = self._holds[session_id]
            current.timeout = max(current.timeout, new_hold.timeout)
            current.expires_at = max(current.expires_at, new_hold.expires_at)
            current.reason = new_hold.reason

    def release(self, session_id: str):
        """Clear any pending hold for a session."""
        with self._lock:
            self._holds.pop(session_id, None)

    def reset_session(self, session_id: str):
        self.release(session_id)

    def get_hold(self, session_id: str) -> Optional[SessionTurnHold]:
        now = time.monotonic()
        with self._lock:
            hold = self._holds.get(session_id)
            if hold is None or hold.expires_at <= now:
                self._holds.pop(session_id, None)
                return None
            return SessionTurnHold(
                timeout=hold.timeout,
                expires_at=hold.expires_at,
                reason=hold.reason,
            )

    def _purge_expired_locked(self, now: float):
        expired_session_ids = [
            session_id
            for session_id, hold in self._holds.items()
            if hold.expires_at <= now
        ]
        for session_id in expired_session_ids:
            self._holds.pop(session_id, None)

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
        now = time.monotonic()
        with self._lock:
            hold = self._holds.pop(session_id, None)
            if hold is None:
                if self.debug:
                    logger.info("Session Hold Turn: PASS session=%s, no active hold", session_id)
                return TurnEndDecision(
                    should_end=True,
                    confidence=1.0,
                    reason="session_hold_inactive",
                )

        if hold.expires_at <= now:
            if self.debug:
                logger.info("Session Hold Turn: PASS session=%s, hold expired", session_id)
            return TurnEndDecision(
                should_end=True,
                confidence=1.0,
                reason="session_hold_expired",
            )

        if self.debug:
            logger.info(
                "Session Hold Turn: WAIT session=%s, timeout=%.3f, reason=%s",
                session_id,
                hold.timeout,
                hold.reason,
            )

        return TurnEndDecision(
            should_end=False,
            confidence=1.0,
            reason=hold.reason,
            timeout=hold.timeout,
        )
