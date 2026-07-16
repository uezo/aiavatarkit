import asyncio
import copy
from dataclasses import dataclass, field
import inspect
import logging
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

from .models import STSRequest, STSResponse


logger = logging.getLogger(__name__)

_UNSET = object()


def _freeze(value: Any) -> Any:
    """Create a recursively read-only snapshot without copying immutable bytes."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, bytes):
        return value
    return copy.deepcopy(value)


@dataclass(frozen=True)
class DialogRequest:
    """Read-only request data exposed to dialog observer functions."""

    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    audio_duration: float = 0
    files: Any = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    channel: Optional[str] = None

    @classmethod
    def from_sts_request(
        cls,
        request: STSRequest,
        *,
        text: Any = _UNSET,
    ) -> "DialogRequest":
        return cls(
            text=request.text if text is _UNSET else text,
            audio_data=request.audio_data,
            audio_duration=request.audio_duration,
            files=_freeze(request.files),
            metadata=_freeze(request.metadata or {}),
            channel=request.channel,
        )


@dataclass(frozen=True)
class DialogResponse:
    """Read-only final response data exposed to dialog observer functions."""

    text: Optional[str] = None
    voice_text: Optional[str] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    structured_content: Any = None

    @classmethod
    def from_sts_response(cls, response: STSResponse) -> "DialogResponse":
        return cls(
            text=response.text,
            voice_text=response.voice_text,
            metadata=_freeze(response.metadata or {}),
            structured_content=_freeze(response.structured_content),
        )


@dataclass(frozen=True)
class DialogObservation:
    """A request or response observation with stable pipeline identifiers."""

    session_id: str
    transaction_id: str
    context_id: Optional[str]
    user_id: Optional[str]
    request: DialogRequest
    response: Optional[DialogResponse] = None


DialogObserverFunction = Callable[
    [DialogObservation, dict[Any, Any]],
    Awaitable[Optional[dict[Any, Any]]],
]


@dataclass
class _ObserverSlot:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sequence: int = 0
    task: Optional[asyncio.Task] = None


class DialogObserver:
    """Run latest-wins asynchronous observers for requests and responses.

    A running observer is isolated by session, phase, and registered function.
    A newer observation cancels and replaces the older invocation in the same
    slot. State patches are committed only when the invocation still has the
    latest sequence, so a slow or cancellation-resistant observer cannot
    overwrite newer state.
    """

    REQUEST_PHASE = "request"
    RESPONSE_PHASE = "response"

    def __init__(self, *, handler_timeout: Optional[float] = None):
        if handler_timeout is not None and handler_timeout <= 0:
            raise ValueError("handler_timeout must be greater than zero")

        self.handler_timeout = handler_timeout
        self._request_handlers: list[DialogObserverFunction] = []
        self._response_handlers: list[DialogObserverFunction] = []

        self._states: Dict[str, dict[Any, Any]] = {}
        self._state_locks: Dict[str, asyncio.Lock] = {}
        self._slots: Dict[Tuple[str, str, int], _ObserverSlot] = {}

        self._replacement_task_sessions: Dict[asyncio.Task, str] = {}
        self._closed = False

    def on_request(self, function: DialogObserverFunction) -> DialogObserverFunction:
        """Register an async function invoked for request observations."""
        self._register(self._request_handlers, function)
        return function

    def on_response(self, function: DialogObserverFunction) -> DialogObserverFunction:
        """Register an async function invoked for final response observations."""
        self._register(self._response_handlers, function)
        return function

    def _register(
        self,
        handlers: list[DialogObserverFunction],
        function: DialogObserverFunction,
    ) -> None:
        if not inspect.iscoroutinefunction(function):
            raise TypeError("Dialog observer functions must be async")
        if function not in handlers:
            handlers.append(function)

    def observe_request(self, observation: DialogObservation) -> None:
        """Schedule request observers without waiting in the pipeline."""
        if observation.response is not None:
            raise ValueError("Request observation must not contain a response")
        self._observe(self.REQUEST_PHASE, self._request_handlers, observation)

    def observe_response(self, observation: DialogObservation) -> None:
        """Schedule response observers without waiting in the pipeline."""
        if observation.response is None:
            raise ValueError("Response observation requires a response")
        self._observe(self.RESPONSE_PHASE, self._response_handlers, observation)

    def _observe(
        self,
        phase: str,
        handlers: list[DialogObserverFunction],
        observation: DialogObservation,
    ) -> None:
        if self._closed or not handlers:
            return

        for function in handlers:
            slot_key = (
                observation.session_id,
                phase,
                id(function),
            )
            slot = self._slots.get(slot_key)
            if slot is None:
                slot = _ObserverSlot()
                self._slots[slot_key] = slot
            slot.sequence += 1

            # Request cancellation immediately. _replace_observer waits for the
            # observer to stop before it starts the replacement.
            if slot.task is not None and not slot.task.done():
                slot.task.cancel()

            replacement_task = asyncio.create_task(
                self._replace_observer(
                    slot=slot,
                    function=function,
                    observation=observation,
                    sequence=slot.sequence,
                )
            )
            self._track_replacement_task(
                replacement_task,
                observation.session_id,
            )

    async def _replace_observer(
        self,
        *,
        slot: _ObserverSlot,
        function: DialogObserverFunction,
        observation: DialogObservation,
        sequence: int,
    ) -> None:
        try:
            async with slot.lock:
                if sequence != slot.sequence:
                    return

                previous_task = slot.task
                if previous_task is not None and not previous_task.done():
                    previous_task.cancel()
                    await asyncio.gather(previous_task, return_exceptions=True)

                if sequence != slot.sequence or self._closed:
                    return

                slot.task = asyncio.create_task(
                    self._run_observer(
                        slot=slot,
                        function=function,
                        observation=observation,
                        sequence=sequence,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to replace dialog observer task")

    async def _run_observer(
        self,
        *,
        slot: _ObserverSlot,
        function: DialogObserverFunction,
        observation: DialogObservation,
        sequence: int,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            state = self.get_state(observation.session_id)
            awaitable = function(observation, state)
            if self.handler_timeout is None:
                states = await awaitable
            else:
                states = await asyncio.wait_for(
                    awaitable,
                    timeout=self.handler_timeout,
                )

            if states is None:
                return
            if not isinstance(states, dict):
                raise TypeError(
                    "Dialog observer functions must return a dict or None"
                )

            # Check both before and after taking the state lock. A replacement
            # may advance the sequence while this observer is completing.
            if sequence != slot.sequence:
                return
            state_lock = self._state_locks.setdefault(
                observation.session_id,
                asyncio.Lock(),
            )
            async with state_lock:
                if sequence != slot.sequence:
                    return
                self._states.setdefault(observation.session_id, {}).update(
                    copy.deepcopy(states)
                )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "Dialog observer timed out: session=%s function=%s",
                observation.session_id,
                getattr(
                    function,
                    "__name__",
                    repr(function),
                ),
            )
        except Exception:
            logger.exception(
                "Dialog observer failed: session=%s function=%s",
                observation.session_id,
                getattr(
                    function,
                    "__name__",
                    repr(function),
                ),
            )
        finally:
            if slot.task is current_task:
                slot.task = None

    def get_state(self, session_id: str) -> dict[Any, Any]:
        """Return a detached copy of the latest committed state."""
        return copy.deepcopy(self._states.get(session_id, {}))

    async def update_state(
        self,
        session_id: str,
        states: dict[Any, Any],
    ) -> None:
        """Atomically update state outside an observer function."""
        state_lock = self._state_locks.setdefault(session_id, asyncio.Lock())
        async with state_lock:
            self._states.setdefault(session_id, {}).update(
                copy.deepcopy(states)
            )

    async def wait_for_idle(self, session_id: Optional[str] = None) -> None:
        """Wait until current replacement and observer tasks have completed."""
        while True:
            tasks = {
                task
                for task, task_session_id in self._replacement_task_sessions.items()
                if not task.done()
                and (session_id is None or task_session_id == session_id)
            }
            tasks.update(
                slot.task
                for (slot_session_id, _, _), slot in self._slots.items()
                if slot.task is not None
                and not slot.task.done()
                and (session_id is None or slot_session_id == session_id)
            )
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0)

    async def clear_session(self, session_id: str) -> None:
        """Cancel observer tasks and delete state for one session."""
        slot_items = [
            (key, slot)
            for key, slot in self._slots.items()
            if key[0] == session_id
        ]
        tasks = []
        for _, slot in slot_items:
            slot.sequence += 1
            if slot.task is not None and not slot.task.done():
                slot.task.cancel()
                tasks.append(slot.task)

        replacement_tasks = [
            task
            for task, task_session_id in self._replacement_task_sessions.items()
            if task_session_id == session_id and not task.done()
        ]
        for task in replacement_tasks:
            task.cancel()
        tasks.extend(replacement_tasks)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for key, _ in slot_items:
            self._slots.pop(key, None)
        self._states.pop(session_id, None)
        self._state_locks.pop(session_id, None)

    async def shutdown(self) -> None:
        """Cancel all observers and release all in-memory state."""
        self._closed = True
        session_ids = {
            *self._states.keys(),
            *(key[0] for key in self._slots),
            *self._replacement_task_sessions.values(),
        }
        for session_id in session_ids:
            await self.clear_session(session_id)

    def _track_replacement_task(
        self,
        task: asyncio.Task,
        session_id: str,
    ) -> None:
        self._replacement_task_sessions[task] = session_id

        def on_done(completed_task: asyncio.Task) -> None:
            self._replacement_task_sessions.pop(completed_task, None)
            if completed_task.cancelled():
                return
            try:
                completed_task.result()
            except Exception:
                logger.exception("Dialog observer replacement task failed")

        task.add_done_callback(on_done)
