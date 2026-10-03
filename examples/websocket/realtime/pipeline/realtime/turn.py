"""Turn identity and history state, independent of execution and persistence."""

import asyncio
from dataclasses import dataclass, field

from aiavatar.sts.models import STSRequest


@dataclass
class TurnHistory:
    """History retained after a turn stops generating or playing its response."""

    input_item_id: str | None = None
    user_text: str | None = None
    assistant_text: str | None = None
    response_completed_at: float = 0.0
    input_ready: asyncio.Event = field(default_factory=asyncio.Event)
    response_ready: asyncio.Event = field(default_factory=asyncio.Event)

    def complete_response(self, now: float, assistant_text: str | None = None):
        if self.response_ready.is_set():
            return
        self.assistant_text = assistant_text
        self.response_completed_at = now
        self.response_ready.set()

    def accept_transcript(self, text: str | None = None, *, failed: bool = False):
        if self.input_ready.is_set():
            return
        if not failed:
            self.user_text = text
        self.input_ready.set()

    def remaining(self, now: float, timeout: float) -> float:
        return max(0.0, self.response_completed_at + timeout - now)

    def messages(self) -> list[dict]:
        messages = []
        if self.user_text and self.user_text.strip():
            messages.append({"role": "user", "content": self.user_text})
        if self.assistant_text and self.assistant_text.strip():
            messages.append({"role": "assistant", "content": self.assistant_text})
        return messages

    def close(self):
        """Release pending history using only information already received."""
        self.input_ready.set()
        self.response_ready.set()


@dataclass(kw_only=True)
class Turn:
    request: STSRequest
    transaction_id: str
    input_item_id: str | None = None
    response_id: str | None = None
    response_requested: bool = False
    completed: bool = False
    finished: bool = False
    terminal: bool = False
    context: TurnHistory | None = None

    def matches_response(self, response_id: str | None) -> bool:
        return self.response_id == response_id and not self.completed

    def blocks_input(self, barge_in_enabled: bool) -> bool:
        return not self.finished and (self.request.block_barge_in or not barge_in_enabled)


class TurnHistoryTracker:
    """Correlate pending histories with late transcripts within one session."""

    def __init__(self):
        self.pending: dict[int, TurnHistory] = {}
        self.inputs: dict[str, TurnHistory] = {}

    @staticmethod
    def begin(
        *,
        text_input: bool,
        request_text: str | None,
        input_item_id: str | None,
        transcription_enabled: bool,
        current_input_item_id: str | None,
        transcription_completed: bool,
        input_caption: str | None,
    ) -> TurnHistory:
        """Create history state; retain it only after the writer accepts it."""
        history = TurnHistory(input_item_id=None if text_input else input_item_id)
        if text_input:
            history.accept_transcript(request_text)
        elif not transcription_enabled or history.input_item_id is None:
            history.accept_transcript()
        elif history.input_item_id == current_input_item_id and transcription_completed:
            history.accept_transcript(input_caption)
        return history

    def retain(self, history: TurnHistory):
        self.pending[id(history)] = history
        if history.input_item_id is not None and not history.input_ready.is_set():
            self.inputs[history.input_item_id] = history

    def discard(self, history: TurnHistory):
        self.pending.pop(id(history), None)
        if self.inputs.get(history.input_item_id) is history:
            del self.inputs[history.input_item_id]

    def capture_transcript(self, item_id: str | None, text: str | None = None, *, failed: bool = False):
        history = self.inputs.get(item_id)
        if history is not None:
            history.accept_transcript(text, failed=failed)

    def close(self):
        for history in self.pending.values():
            history.close()

    def clear(self):
        self.pending.clear()
        self.inputs.clear()
