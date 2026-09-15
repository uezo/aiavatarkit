"""One conversation's bounded, in-memory nod history and pause decisions."""

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
import json
import logging
import math
from typing import Awaitable, Callable

from .engine import NodDecision, NodEngine
from .input import plain_input

logger = logging.getLogger(__name__)


@dataclass
class _Message:
    role: str
    id: str
    text: str = ""
    offset: int = 0
    nods: deque = field(default_factory=lambda: deque(maxlen=3))

    def snapshot(self):
        return {"role": self.role, "content": self.text, "text_offset": self.offset,
                "nods": [dict(nod) for nod in self.nods]}


class NodSession:
    """Create one per conversation; use it from a single asyncio event loop.

    The engine may be shared. This object owns neither the engine nor transport.
    Start an input before forwarding cumulative ASR updates. Ending eligibility
    is separate from final ASR: late updates can still correct retained history.
    ``emit`` returns True only after successful delivery; it must propagate
    cancellation and check transport ownership immediately before sending.
    """

    def __init__(self, engine: NodEngine, emit: Callable[[NodDecision], Awaitable[bool]], *,
                 timeout: float = 1.5, min_interval: float = 2.0,
                 history_limit: int = 10, max_text_chars: int = 2400,
                 log_inputs: bool = False):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("min_interval must be nonnegative and finite")
        if not isinstance(history_limit, int) or not 1 <= history_limit <= 10:
            raise ValueError("history_limit must be 1..10 messages")
        if not isinstance(max_text_chars, int) or max_text_chars < 1800:
            raise ValueError("max_text_chars must be at least 1800")
        self.engine, self.emit = engine, emit
        self.timeout, self.min_interval = timeout, min_interval
        self.history_limit, self.max_text_chars = history_limit, max_text_chars
        self.log_inputs = log_inputs
        self._messages = OrderedDict()
        self._active = None
        self._generation = 0
        self._task = None
        self._delivery = None
        self._updated = asyncio.Event()
        self._closed = False

    @staticmethod
    def _validate_id(identifier):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Message IDs must be nonempty strings")

    def _check_open(self):
        if self._closed:
            raise RuntimeError("NodSession is closed")

    def _trim(self):
        # Retain ten prior messages plus the current input, if any.
        limit = self.history_limit + (self._active is not None)
        while len(self._messages) > limit:
            key = next(key for key in self._messages if key != self._active)
            del self._messages[key]

    def start_input(self, utterance_id: str):
        """Begin a new user input in conversation order, before ASR arrives."""
        self._check_open()
        self._validate_id(utterance_id)
        key = ("user", utterance_id)
        if key == self._active:
            return
        if key in self._messages:
            raise ValueError("A finished input cannot be reopened; use a new ID")
        if self._active is not None:
            self.end_input(self._active[1])
        self._generation += 1
        self._active = key
        self._messages[key] = _Message("user", utterance_id)
        self._trim()

    def _update(self, message, text):
        if not isinstance(text, str):
            raise TypeError("Text must be a cumulative string")
        message.offset = max(0, len(text) - self.max_text_chars)
        message.text = text[-self.max_text_chars:]

    def update_user(self, utterance_id: str, text: str, *, create: bool = False) -> bool:
        """Replace cumulative ASR for a retained input; never reactivate old IDs.

        False means the input was never started or has left the bounded history.
        With create=True, a missing final input is retained as history before
        the active input, without starting it or cancelling current work.
        """
        self._check_open()
        message = self._messages.get(("user", utterance_id))
        if message is None and create:
            self._validate_id(utterance_id)
            if not isinstance(text, str):
                raise TypeError("Text must be a cumulative string")
            message = _Message("user", utterance_id)
            self._messages[("user", utterance_id)] = message
            if self._active is not None:
                self._messages.move_to_end(self._active)
            self._trim()
        if message is None:
            return False
        self._update(message, text)
        self._updated.set()
        return True

    def update_assistant(self, response_id: str, spoken_text: str):
        """Upsert cumulative main-response speech, without app-specific tags.

        Call end_input when the main response is accepted, before emitting it.
        Updates to an existing response preserve its place in conversation order.
        """
        self._check_open()
        self._validate_id(response_id)
        if not isinstance(spoken_text, str):
            raise TypeError("Text must be a cumulative string")
        key = ("assistant", response_id)
        if key not in self._messages:
            self._messages[key] = _Message("assistant", response_id)
        self._update(self._messages[key], spoken_text)
        self._trim()

    def end_input(self, utterance_id: str):
        """Stop new nods on commit/main-response acceptance; keep the history."""
        if self._active != ("user", utterance_id):
            return
        self._active = None
        self._generation += 1
        self._updated.set()
        if self._task is not None and self._task is not asyncio.current_task():
            self._task.cancel()
        self._trim()

    def build_input(self, utterance_id: str) -> str:
        """Inspect the exact plain-text user message, without querying the LLM."""
        key = ("user", utterance_id)
        current = self._messages[key]
        history = [m.snapshot() for k, m in self._messages.items() if k != key]
        return plain_input(history, current.text, current.nods,
                           history_limit=self.history_limit, text_offset=current.offset)

    def _current(self, key, generation):
        return not self._closed and self._active == key and self._generation == generation

    def _cooling_down(self, message, now):
        return bool(message.nods and now - message.nods[-1]["sent_at"] < self.min_interval)

    @staticmethod
    def _unchanged(message, text, end):
        start = end - len(text) - message.offset
        return start >= 0 and message.text[start:start + len(text)] == text

    def is_current(self, decision: NodDecision) -> bool:
        """Recheck immediately before a transport write after waiting for a lock.

        Valid only while this exact decision is being passed to ``emit``.
        A successful earlier send cannot be retracted by this check.
        """
        if self._delivery is None or self._delivery[0] is not decision:
            return False
        _, key, generation, text, end = self._delivery
        return (self._current(key, generation)
                and self._unchanged(self._messages[key], text, end))

    async def on_pause(self, utterance_id: str, *, detected_at: float | None = None) -> NodDecision:
        """Wait for nonempty ASR, decide once, and deliver within one deadline.

        ``detected_at`` is an optional asyncio loop.time() timestamp. Repeated
        pauses while busy are dropped. Resumed speech alone does not cancel this
        call. Cancellation from its caller, end_input, or close is propagated.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        if detected_at is not None and not math.isfinite(detected_at):
            raise ValueError("detected_at must be a finite monotonic timestamp")
        deadline = (started if detected_at is None else detected_at) + self.timeout
        key, generation = ("user", utterance_id), self._generation
        result = NodDecision("none", utterance_id=utterance_id, outcome="inactive")
        if not self._current(key, generation):
            return result
        if self._task is not None:
            return replace(result, outcome="busy")
        task = asyncio.current_task()
        self._task = task
        message = self._messages[key]
        user_input, phase = None, "transcript"
        try:
            if self._cooling_down(message, started):
                result = replace(result, outcome="cooldown")
                return result
            async with asyncio.timeout_at(deadline):
                if loop.time() >= deadline:
                    result = replace(result, outcome="timeout")
                    return result
                while not message.text.strip():
                    self._updated.clear()
                    await self._updated.wait()
                    if not self._current(key, generation):
                        return result
                text = message.text[-1200:]
                end = message.offset + len(message.text)
                user_input = self.build_input(utterance_id)
                phase = "api"
                result = replace(await self.engine.decide(user_input), utterance_id=utterance_id,
                                 acknowledged_text=text[-600:], text_length=end)
                # A custom engine may swallow cancellation; do not deliver its
                # late result after the caller has cancelled this pause task.
                if task.cancelling():
                    raise asyncio.CancelledError
                if not self._current(key, generation):
                    result = replace(result, outcome="inactive")
                elif loop.time() >= deadline:
                    result = replace(result, outcome="timeout")
                elif not self._unchanged(message, text, end):
                    result = replace(result, outcome="transcript_changed")
                elif result.phrase is not None:
                    phase = "emit"
                    self._delivery = (result, key, generation, text, end)
                    sent = await self.emit(result)
                    result = replace(result, outcome="sent" if sent is True else "not_sent")
                    if sent is True:
                        message.nods.append({"phrase": result.phrase, "id": result.id,
                                             "acknowledged_text": result.acknowledged_text,
                                             "text_length": end, "sent_at": loop.time()})
        except TimeoutError:
            result = replace(result, outcome="timeout")
        except asyncio.CancelledError:
            result = replace(result, outcome="cancelled")
            raise
        except Exception as error:
            result = replace(result, outcome="error")
            logger.warning("Nod failed: error_type=%s", type(error).__name__)
        finally:
            self._delivery = None
            if self._task is task:
                self._task = None
            logger.info("Nod decision: %s", json.dumps({
                "utterance_id": utterance_id, "outcome": result.outcome, "id": result.id,
                "phrase": result.phrase, "phase": phase, "finish_reason": result.finish_reason,
                "elapsed_ms": round((loop.time() - started) * 1000, 2),
                "api_ms": round(result.elapsed_ms, 2),
                **({"input": user_input} if self.log_inputs else {}),
            }, ensure_ascii=False))
        return result

    async def close(self):
        """Cancel/join pending work and release only this conversation's memory."""
        self._closed = True
        task = self._task
        if self._active is not None:
            self.end_input(self._active[1])
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._messages.clear()

    async def __aenter__(self):
        self._check_open()
        return self

    async def __aexit__(self, *exc):
        await self.close()
