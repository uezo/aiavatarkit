"""Track response playback for interruption context on an audio connection."""

from dataclasses import dataclass
from sys import float_info
from time import monotonic
from typing import Optional

from ..sts.models import Interruption, STSResponse


@dataclass
class _Response:
    context_id: Optional[str]
    voice_text: str = ""
    answered: bool = False
    finished: bool = False


@dataclass
class _Playback:
    playback_id: str
    transaction_id: str
    text: str
    duration_seconds: float
    started_at: float
    preceding_text: str
    ended_at: Optional[float] = None
    completed: bool = False
    is_final_chunk: bool = False

    @property
    def natural_end(self) -> float:
        return self.started_at + self.duration_seconds

    def spoken_text(self, now: float) -> str:
        at = self.ended_at if self.ended_at is not None else now
        if self.completed or at >= self.natural_end:
            return self.preceding_text + self.text
        progress = max(0.0, at - self.started_at) / self.duration_seconds
        return self.preceding_text + self.text[:int(len(self.text) * progress)]


class InterruptionState:
    """Track delivery and playback of recent responses on one connection."""

    def __init__(self):
        self._responses: dict[str, _Response] = {}
        self._last_answered: Optional[tuple[Optional[str], str]] = None
        self._playback: Optional[_Playback] = None
        self._playback_ids: set[str] = set()

    def mark_answered(self, transaction_id: Optional[str]):
        if transaction_id and (response := self._responses.get(transaction_id)) and not response.answered:
            response.answered = True
            self._last_answered = (response.context_id, transaction_id)

    def observe(self, response: STSResponse):
        transaction_id = response.transaction_id
        if not transaction_id:
            return
        if response.type == "start":
            self._responses[transaction_id] = _Response(context_id=response.context_id)
            if len(self._responses) > 8:
                self._responses.pop(next(iter(self._responses)))
        elif previous := self._responses.get(transaction_id):
            if response.type == "chunk" and response.audio_data:
                previous.voice_text += response.voice_text or ""
            elif response.type in ("final", "error", "canceled", "cancelled"):
                previous.finished = True

    def handle_playback_event(
        self, /, *, event=None, playback_id=None, transaction_id=None,
        text=None, duration_seconds=None, completed=False, **_metadata,
    ):
        if event == "start":
            self._start_playback(playback_id, transaction_id, text, duration_seconds)
            return

        playback = self._playback
        if playback is None or playback.playback_id != playback_id:
            return
        if event == "end" and playback.ended_at is None:
            playback.ended_at = monotonic()
            playback.completed = completed is True
        elif event == "final" and playback.transaction_id == transaction_id:
            playback.is_final_chunk = True

    def _start_playback(self, playback_id, transaction_id, text, duration_seconds):
        if (
            not isinstance(playback_id, str) or not playback_id.strip()
            or not isinstance(transaction_id, str) or not transaction_id.strip()
            or not isinstance(text, str) or not text.strip()
            or type(duration_seconds) not in (int, float) or not 0 < duration_seconds <= float_info.max
        ):
            self._playback = None
            self._playback_ids.clear()
            return

        now = monotonic()
        previous = self._playback
        preceding_text = ""
        if previous is not None and previous.transaction_id == transaction_id:
            if playback_id in self._playback_ids:
                return
            # A new start closes the previous chunk even if its end report was missed.
            preceding_text = previous.spoken_text(now)
        else:
            self._playback_ids.clear()
        self._playback_ids.add(playback_id)
        self._playback = _Playback(playback_id, transaction_id, text, duration_seconds, now, preceding_text)

    def snapshot(
        self, active_transaction_id: Optional[str], *, muted_transaction_id: Optional[str] = None,
    ) -> Optional[Interruption]:
        now = monotonic()
        playback = self._playback
        if playback is not None and playback.ended_at is None and now < playback.natural_end:
            return self._spoken_interruption(playback, now)

        if active_transaction_id and (previous := self._responses.get(active_transaction_id)):
            # Keep the spoken prefix during the silence before the next chunk.
            if (
                playback is not None and playback.completed  # Chunk played in full
                and not playback.is_final_chunk and not previous.finished  # Last chunk unconfirmed, response open
                and playback.transaction_id == active_transaction_id  # Same active response
                and active_transaction_id != muted_transaction_id  # Response unmuted
            ):
                return self._spoken_interruption(playback, now)
            if previous.answered or previous.finished:
                return None
            answered_through = self._last_answered
            return Interruption(
                interrupted_response_context_id=previous.context_id,
                interrupted_response_transaction_id=active_transaction_id,
                answered_through_transaction_id=(
                    answered_through[1] if answered_through and answered_through[0] == previous.context_id else None
                ),
            )
        return None

    def _spoken_interruption(self, playback: _Playback, now: float) -> Optional[Interruption]:
        transaction_id = playback.transaction_id
        previous = self._responses.get(transaction_id)
        spoken_text = playback.spoken_text(now)
        # Verify the playback text against the response produced by the server.
        if previous and spoken_text and previous.voice_text.startswith(spoken_text):
            return Interruption(
                interrupted_response_context_id=previous.context_id,
                interrupted_response_transaction_id=transaction_id,
                spoken_text=spoken_text,
            )
        return None
