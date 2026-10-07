"""Wakeword admission using caller-supplied conversation activity."""

from datetime import datetime, timezone
import logging
from typing import Awaitable, Callable, Optional

from .base import TurnTakingDecision, TurnTakingGate, _finite_number

logger = logging.getLogger(__name__)

ConversationActivityCallback = Callable[[str], Awaitable[Optional[datetime]]]


class WakewordGate(TurnTakingGate):
    """Require a wakeword unless the caller reports recent conversation activity.

    ``get_last_conversation_at(session_id)`` returns a timezone-aware timestamp
    or None. This gate only reads it: accepting an input never renews activity.
    Without a callback, every input requires a wakeword. An empty or omitted
    ``wakewords`` list disables this literal gate.

    A match is a case-sensitive substring of the unchanged input, just as in
    STSPipeline. Admission returns None so later gates can still classify
    the turn; a sleeping, unmatched input returns False. Place this gate before
    a TurnTakingBypassGate to check wakewords before playback and duration
    bypasses. Callback failures block the input, while cancellation propagates
    to the session owner.

    Subclasses may replace ``_detect_wakeword()`` and ``_wake_detection_enabled()``
    to reuse continuation checks without relying on literal wakewords.
    """

    def __init__(
        self, wakewords: Optional[list[str]] = None, *,
        wakeword_timeout: float = 60.0,
        get_last_conversation_at: Optional[ConversationActivityCallback] = None,
        debug: bool = False,
    ):
        if wakewords is not None and (
            not isinstance(wakewords, list)
            or any(not isinstance(word, str) or not word for word in wakewords)
        ):
            raise ValueError("wakewords must be a list of non-empty strings or None")
        if not _finite_number(wakeword_timeout) or wakeword_timeout < 0:
            raise ValueError("wakeword_timeout must be a finite number greater than or equal to 0")
        if get_last_conversation_at is not None and not callable(get_last_conversation_at):
            raise ValueError("get_last_conversation_at must be callable or None")
        super().__init__(bypass_enabled=False, debug=debug)
        self.wakewords = list(wakewords) if wakewords is not None else []
        self.wakeword_timeout = wakeword_timeout
        self.get_last_conversation_at = get_last_conversation_at

    def _wake_detection_enabled(self) -> bool:
        return bool(self.wakewords)

    async def _detect_wakeword(
        self, user_text: str, *, session_id: Optional[str] = None,
    ) -> TurnTakingDecision:
        matched = any(word in user_text for word in self.wakewords)
        return TurnTakingDecision(
            should_take_turn=None if matched else False,
            probability=None,
            reason="wakeword_match" if matched else "wakeword_missing",
        )

    async def should_take_turn(
        self, user_text: Optional[str], assistant_spoken_text: Optional[str], *,
        session_id: Optional[str] = None,
        assistant_full_text: Optional[str] = None,
        **kwargs,
    ) -> TurnTakingDecision:
        try:
            decision = await self._evaluate_wakeword(user_text, session_id)
        except Exception as exc:
            # Do not log callback exception bodies: they may contain user data.
            # CancelledError is deliberately not caught.
            logger.warning("Wakeword gate failed (%s); blocking input", type(exc).__name__)
            decision = TurnTakingDecision(
                should_take_turn=False, probability=None, reason="wakeword_error",
            )
        if self.debug:
            logger.info(
                "Wakeword Gate: session=%s, should_take_turn=%s, probability=%s, reason=%s",
                session_id, decision.should_take_turn, decision.probability, decision.reason,
            )
        return decision

    async def _evaluate_wakeword(
        self, user_text: Optional[str], session_id: Optional[str],
    ) -> TurnTakingDecision:
        if not self._wake_detection_enabled():
            return TurnTakingDecision(
                should_take_turn=None, probability=None, reason="wakeword_disabled",
            )
        if self.get_last_conversation_at is not None and session_id and self.wakeword_timeout > 0:
            last_conversation_at = await self.get_last_conversation_at(session_id)
            if last_conversation_at is not None:
                if not isinstance(last_conversation_at, datetime) or last_conversation_at.utcoffset() is None:
                    raise ValueError("get_last_conversation_at must return an aware datetime or None")
                elapsed = (datetime.now(timezone.utc) - last_conversation_at).total_seconds()
                if elapsed < self.wakeword_timeout:
                    return TurnTakingDecision(
                        should_take_turn=None, probability=None,
                        reason="wakeword_conversation_active",
                    )
        if not user_text or not user_text.strip():
            return TurnTakingDecision(
                should_take_turn=False, probability=None, reason="wakeword_no_text",
            )
        return await self._detect_wakeword(user_text, session_id=session_id)
