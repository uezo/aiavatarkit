"""Playback context and per-session coordination shared by turn-taking gates."""

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import logging
import math
import time
from typing import Callable, Optional

from .base import TurnTakingDecision, TurnTakingGate, _finite_number

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybackEstimate:
    """Estimated response prefix; timing fields describe its last included chunk.

    ``is_playing`` includes known gaps between chunks of the same transaction.
    ``continuation_hint`` supplements only ``evaluation_text``, not progress.
    ``assistant_full_text`` includes unread text from all known started chunks
    of this response, which may still be incomplete.
    """

    playback_id: Optional[str]
    is_playing: bool
    assistant_spoken_text: str
    elapsed_seconds: float
    duration_seconds: float
    reason: str
    transaction_id: Optional[str] = None
    continuation_hint: str = ""
    is_final_chunk: bool = False
    assistant_full_text: str = ""

    @property
    def evaluation_text(self) -> str:
        return self.assistant_spoken_text + self.continuation_hint

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.duration_seconds - self.elapsed_seconds)


@dataclass(frozen=True)
class _Playback:
    playback_id: str
    text: str
    duration_seconds: float
    started_at: float
    ended_at: Optional[float] = None
    completed: bool = False
    is_final_chunk: bool = False


class TurnTakingSession:
    """Own one connection's playback and latest valid turn-taking decision.

    Start notifications timestamp receipt on the server's monotonic clock.
    ``should_take_turn()`` estimates the spoken prefix once, when the finalized user input
    arrives. An optional timezone-aware ``speech_end_at`` removes the delay since
    the user's estimated speech end from active playback progress. The estimate
    is proportional to elapsed time and Unicode code-point length within each
    chunk. A valid ``transaction_id`` groups started chunks from the same
    response, allowing reconstruction of its spoken prefix at the target time.
    Only the current transaction is retained; chunks without a valid ID remain
    independent. Known gaps between same-response chunks use the preceding
    spoken prefix. For classification only, a completed chunk followed by a
    known chunk with zero estimated characters gains that next chunk's first
    two characters plus an ellipsis as a continuation hint. After the last
    known chunk ends, the turn is allowed unless
    ``speech_end_at`` places the input within earlier retained playback.

    Each finalized input cancels the previous pending decision without waiting
    for it to finish, even when the new input bypasses classification. Superseded decisions
    propagate cancellation instead of returning a result to the caller.
    A synchronous skip condition receives the text and recorded duration for
    each input. The gate's explicit condition takes priority over this session's
    default condition. Returning True allows the input without classification.
    Missing user text also allows the input, for recognition later in the pipeline.

    ``close()`` invalidates the session and cancels pending work synchronously;
    await ``aclose()`` to join that work. Closing cancels pending
    decisions without closing caller-owned provider resources. Closed sessions
    ignore controls and allow turns without a classifier call. Callers still check
    connection ownership before applying a decision after an await.

    Debug events ``session_started`` and ``session_closed`` describe this
    helper's creation and teardown, rather than network connection events.
    """

    def __init__(
        self, gate: TurnTakingGate, session_id: str, *,
        default_skip_condition: Optional[Callable[[Optional[str], float], bool]] = None,
    ):
        if default_skip_condition is not None and not callable(default_skip_condition):
            raise ValueError("default_skip_condition must be callable or None")
        self._gate = gate
        self._default_skip_condition = default_skip_condition
        self.session_id = session_id
        self._playback: Optional[_Playback] = None
        self._transaction_id: Optional[str] = None
        self._playbacks: list[_Playback] = []
        self._current_task: Optional[asyncio.Task] = None
        # Include retiring tasks until they exit so disconnect can reclaim them.
        self._pending: set[asyncio.Task] = set()
        self._closed = False
        if self._gate.debug:
            logger.info(
                "Turn Take Session: session=%s, event=session_started, reason=initialized",
                self.session_id,
            )

    @property
    def is_closed(self) -> bool:
        return self._closed

    def start_playback(
        self, playback_id: str, text: str, duration_seconds: float, *,
        transaction_id: Optional[str] = None,
    ) -> bool:
        """Register a started chunk, accumulating only within a valid transaction.

        Missing/invalid transaction IDs keep the independent-chunk behavior.
        Invalid playback data clears all retained context.
        """
        if self._closed:
            self._log_control("invalid_start", playback_id, "closed")
            return False
        if (
            not isinstance(playback_id, str) or not playback_id.strip()
            or not isinstance(text, str) or not text.strip()
            or not _finite_number(duration_seconds) or duration_seconds <= 0
        ):
            self._playback = None
            self._transaction_id = None
            self._playbacks.clear()
            self._log_control("invalid_start", playback_id, "invalid_payload")
            return False
        transaction_id = (
            transaction_id if isinstance(transaction_id, str) and transaction_id.strip() else None
        )
        if transaction_id is None or transaction_id != self._transaction_id:
            self._playbacks.clear()
        elif any(playback.playback_id == playback_id for playback in self._playbacks):
            self._log_control("duplicate_start", playback_id, "ignored")
            return True
        self._transaction_id = transaction_id
        started_at = time.monotonic()
        if self._playbacks and self._playbacks[-1].is_final_chunk:
            # A later chunk proves the earlier final designation is obsolete.
            self._playbacks[-1] = replace(self._playbacks[-1], is_final_chunk=False)
        if self._playbacks and self._playbacks[-1].ended_at is None:
            previous = self._playbacks[-1]
            # Playback is serial. A new start closes a missing end notification
            # without assuming that a prematurely interrupted chunk was heard.
            natural_end = previous.started_at + previous.duration_seconds
            self._playbacks[-1] = replace(
                previous,
                ended_at=min(started_at, natural_end),
                completed=started_at >= natural_end,
            )
        self._playback = _Playback(
            playback_id=playback_id,
            text=text,
            duration_seconds=float(duration_seconds),
            started_at=started_at,
        )
        if transaction_id is not None:
            self._playbacks.append(self._playback)
        self._log_control("start", playback_id, "playing")
        return True

    def end_playback(self, playback_id: str, completed: bool = False) -> bool:
        """End only the matching current chunk; late or invalid ends do nothing."""
        playback = self._playback
        if (
            self._closed or playback is None or playback.ended_at is not None
            or not isinstance(playback_id, str) or playback_id != playback.playback_id
            or not isinstance(completed, bool)
        ):
            self._log_control("stale_end", playback_id, "ignored")
            return False
        self._playback = replace(
            playback, ended_at=time.monotonic(), completed=completed,
            is_final_chunk=playback.is_final_chunk and completed,
        )
        if self._playbacks:
            self._playbacks[-1] = self._playback
        self._log_control("end", playback_id, "completed" if completed else "stopped")
        return True

    def mark_response_final(self, playback_id: str, *, transaction_id: str) -> bool:
        """Confirm the current transaction's last chunk, including after completion.

        Ignore unknown/stale IDs and interrupted playback. A subsequent chunk
        revokes this designation; only the last known chunk can receive grace.
        """
        playback = self._playback
        if (
            self._closed or playback is None or self._transaction_id is None
            or not isinstance(transaction_id, str) or not transaction_id.strip()
            or transaction_id != self._transaction_id
            or not isinstance(playback_id, str) or playback_id != playback.playback_id
            or (playback.ended_at is not None and not playback.completed)
        ):
            self._log_control("stale_final", playback_id, "ignored")
            return False
        self._playback = replace(playback, is_final_chunk=True)
        self._playbacks[-1] = self._playback
        self._log_control("final", playback_id, "response_final")
        return True

    def estimate_playback(self, *, speech_end_at: Optional[datetime] = None) -> PlaybackEstimate:
        playback = self._playback
        if self._closed or playback is None:
            return PlaybackEstimate(None, False, "", 0.0, 0.0, "closed" if self._closed else "idle")
        now = time.monotonic()
        target_at = now
        if isinstance(speech_end_at, datetime) and speech_end_at.utcoffset() is not None:
            # Convert the wall-clock timestamp to a delay only once, then apply
            # that delay to the monotonic timeline shared by retained chunks.
            delay = max(0.0, (datetime.now(timezone.utc) - speech_end_at).total_seconds())
            target_at -= delay
        if self._transaction_id is not None:
            return replace(
                self._estimate_transaction(target_at),
                assistant_full_text="".join(chunk.text for chunk in self._playbacks),
            )

        # Preserve independent-chunk behavior for clients without a transaction
        # ID: finished chunks bypass even if speech_end_at predates their end.
        elapsed = max(0.0, (now if playback.ended_at is None else playback.ended_at) - playback.started_at)
        if playback.ended_at is not None:
            reason = "completed" if playback.completed else "stopped"
        elif elapsed >= playback.duration_seconds:
            reason = "expired"
        else:
            reason = "playing"
        if reason == "playing":
            elapsed = max(0.0, target_at - playback.started_at)
        progress = 1.0 if playback.completed else min(1.0, elapsed / playback.duration_seconds)
        offset = math.floor(len(playback.text) * progress)
        return PlaybackEstimate(
            playback_id=playback.playback_id,
            is_playing=reason == "playing",
            assistant_spoken_text=playback.text[:offset],
            elapsed_seconds=elapsed,
            duration_seconds=playback.duration_seconds,
            reason=reason,
            assistant_full_text=playback.text,
        )

    def _estimate_transaction(self, target_at: float) -> PlaybackEstimate:
        spoken_text = ""
        previous = None
        previous_elapsed = 0.0
        previous_completed = False
        for index, playback in enumerate(self._playbacks):
            if target_at < playback.started_at:
                if previous is None:
                    return PlaybackEstimate(
                        playback.playback_id, False, "", 0.0, playback.duration_seconds,
                        "before_start", self._transaction_id,
                    )
                # A later chunk is known to belong to this same response, so
                # the gap did not mean the assistant had yielded its turn.
                return PlaybackEstimate(
                    previous.playback_id, True, spoken_text, previous_elapsed,
                    previous.duration_seconds, "gap", self._transaction_id,
                    continuation_hint=playback.text[:2] + "…" if previous_completed else "",
                )

            natural_end = playback.started_at + playback.duration_seconds
            ended_at = min(natural_end, playback.ended_at) if playback.ended_at is not None else natural_end
            elapsed = max(0.0, min(target_at, ended_at) - playback.started_at)
            # Subtracting a fractional duration from a large monotonic value
            # can round elapsed just below the duration. Once the natural end
            # is reached, include all characters unless an earlier stop cut it.
            completed = target_at >= ended_at and (playback.completed or ended_at == natural_end)
            progress = 1.0 if completed else min(1.0, elapsed / playback.duration_seconds)
            offset = math.floor(len(playback.text) * progress)
            spoken_text += playback.text[:offset]
            if target_at < ended_at:
                return PlaybackEstimate(
                    playback.playback_id, True, spoken_text, elapsed, playback.duration_seconds,
                    "playing", self._transaction_id,
                    continuation_hint=playback.text[:2] + "…" if previous_completed and offset == 0 else "",
                    is_final_chunk=playback.is_final_chunk,
                )
            if index == len(self._playbacks) - 1:
                reason = (
                    "completed" if playback.completed else "stopped"
                ) if playback.ended_at is not None else "expired"
                return PlaybackEstimate(
                    playback.playback_id, False, spoken_text, elapsed, playback.duration_seconds,
                    reason, self._transaction_id,
                    is_final_chunk=playback.is_final_chunk,
                )
            previous = playback
            previous_elapsed = elapsed
            previous_completed = completed

        # A valid transaction always starts with a chunk; this is defensive if
        # the session has already been cleared by its owner.
        return PlaybackEstimate(None, False, "", 0.0, 0.0, "idle", self._transaction_id)

    def _log_control(self, event: str, playback_id: Optional[str], reason: str) -> None:
        if self._gate.debug:
            playback = self._playback
            logger.info(
                "Turn Take Playback: session=%s, event=%s, playback_id=%r, reason=%s, "
                "current_playback_id=%r, transaction_id=%r, text=%r, duration_seconds=%s, is_final_chunk=%s",
                self.session_id, event, playback_id, reason,
                playback.playback_id if playback else None,
                self._transaction_id,
                playback.text if playback else "",
                playback.duration_seconds if playback else None,
                playback.is_final_chunk if playback else False,
            )

    def _log_input(
        self, event: str, user_text: Optional[str], recording_id: Optional[str],
        estimate: PlaybackEstimate, *, decision: Optional[TurnTakingDecision] = None,
    ) -> None:
        if self._gate.debug:
            logger.info(
                "Turn Take Session: session=%s, event=%s, recording_id=%s, playback_id=%r, transaction_id=%r, "
                "user_text=%r, assistant_spoken_text=%r, assistant_full_text=%r, evaluation_text=%r, continuation_hint=%r, is_playing=%s, "
                "elapsed_seconds=%.3f, duration_seconds=%.3f, remaining_seconds=%.3f, is_final_chunk=%s, estimate_reason=%s, "
                "should_take_turn=%s, probability=%s, reason=%s",
                self.session_id, event, recording_id, estimate.playback_id, estimate.transaction_id,
                user_text, estimate.assistant_spoken_text, estimate.assistant_full_text, estimate.evaluation_text,
                estimate.continuation_hint, estimate.is_playing,
                estimate.elapsed_seconds, estimate.duration_seconds, estimate.remaining_seconds,
                estimate.is_final_chunk, estimate.reason,
                decision.should_take_turn if decision else None,
                decision.probability if decision else None,
                decision.reason if decision else event,
            )

    async def should_take_turn(
        self, user_text: Optional[str], *, recorded_duration: float = 0.0,
        recording_id: Optional[str] = None,
        speech_end_at: Optional[datetime] = None,
    ) -> TurnTakingDecision:
        # Freeze the corrected estimate before any network await. A later
        # start/end notification must not change the context of this decision.
        estimate = self.estimate_playback(speech_end_at=speech_end_at)
        task = asyncio.current_task()
        previous = self._current_task
        self._current_task = task
        self._pending.add(task)
        if previous is not None and previous is not task:
            previous.cancel()
        # Do not await the previous task: cancellation cleanup must not hold up
        # the new input. Only the current task may return a decision.

        try:
            self._log_input("final_input", user_text, recording_id, estimate)
            bypass_reason = None
            if self._closed:
                bypass_reason = "turn_take_session_closed"
            elif not user_text or not user_text.strip():
                bypass_reason = "turn_taking_no_text"
            else:
                condition = (
                    self._gate.skip_condition if self._gate.skip_condition is not None
                    else self._default_skip_condition
                )
                if condition is not None:
                    try:
                        if condition(user_text, recorded_duration) is True:
                            bypass_reason = "turn_take_skipped"
                    except Exception as exc:
                        logger.warning(
                            "Turn-taking skip condition failed (%s); allowing user's turn",
                            type(exc).__name__,
                        )
                        bypass_reason = "turn_taking_skip_condition_error"
            if self._current_task is not task or task.cancelling():
                raise asyncio.CancelledError
            if bypass_reason is None:
                if not estimate.is_playing:
                    bypass_reason = "playback_" + estimate.reason
                elif (
                    estimate.reason == "playing" and estimate.is_final_chunk
                    and self._gate.response_end_grace_seconds > 0
                    and (
                        estimate.remaining_seconds <= self._gate.response_end_grace_seconds
                        or math.isclose(
                            estimate.remaining_seconds, self._gate.response_end_grace_seconds,
                            rel_tol=0.0, abs_tol=1e-9,
                        )
                    )
                ):
                    # Allow only the confirmed response ending. The nanosecond
                    # tolerance absorbs subtraction rounding at the threshold.
                    bypass_reason = "playback_response_end_grace"
                elif not estimate.assistant_spoken_text.strip():
                    bypass_reason = "playback_empty_estimate"
            if bypass_reason:
                decision = TurnTakingDecision(True, None, bypass_reason)
                self._log_input("bypass", user_text, recording_id, estimate, decision=decision)
                return decision

            try:
                decision = await self._gate.should_take_turn(
                    user_text, estimate.evaluation_text, session_id=self.session_id,
                    assistant_full_text=estimate.assistant_full_text,
                )
            except Exception:
                # A gate might suppress cancellation and then fail. Do not let
                # the application's error fallback pass that stale input.
                if self._current_task is not task or task.cancelling():
                    raise asyncio.CancelledError from None
                raise
            if self._current_task is not task or task.cancelling():
                raise asyncio.CancelledError
            self._log_input("decision", user_text, recording_id, estimate, decision=decision)
            return decision
        except asyncio.CancelledError:
            event = "superseded" if self._current_task is not task and not self._closed else "cancelled"
            self._log_input(event, user_text, recording_id, estimate)
            raise
        finally:
            self._pending.discard(task)
            if self._current_task is task:
                self._current_task = None

    def close(self) -> None:
        """Immediately invalidate playback and cancel pending decisions."""
        already_closed = self._closed
        self._closed = True
        self._playback = None
        self._transaction_id = None
        self._playbacks.clear()
        self._current_task = None
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        tasks = [task for task in self._pending if task is not current_task]
        if not already_closed:
            if self._gate.debug:
                logger.info(
                    "Turn Take Session: session=%s, event=session_closed, reason=teardown",
                    self.session_id,
                )
            for task in tasks:
                task.cancel()

    async def aclose(self) -> None:
        """Close the session and await cancellation cleanup, excluding this task."""
        self.close()
        current_task = asyncio.current_task()
        tasks = [task for task in self._pending if task is not current_task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
