import asyncio
from dataclasses import dataclass
import logging
from typing import Any, Awaitable, Callable, Optional

from .state import (
    AsteriskCallEvent,
    AsteriskCallState,
    CALL_STATE_TRANSITIONS,
)


logger = logging.getLogger(__name__)


class AsteriskCallActorClosed(RuntimeError):
    pass


class AsteriskCallActorQueueFull(RuntimeError):
    pass


class AsteriskCallTransitionError(RuntimeError):
    pass


@dataclass(slots=True)
class _Envelope:
    event: AsteriskCallEvent
    result: Optional[asyncio.Future]


class AsteriskCallActor:
    """Serialize lifecycle events for one call without carrying media frames."""

    def __init__(
        self,
        *,
        session_id: str,
        handler: Callable[["AsteriskCallActor", AsteriskCallEvent], Awaitable[Any]],
        initial_state: AsteriskCallState = AsteriskCallState.ACTIVE,
        queue_size: int = 64,
        enqueue_timeout: float = 1.0,
        on_transition: Optional[
            Callable[[
                "AsteriskCallActor",
                AsteriskCallState,
                AsteriskCallState,
                AsteriskCallEvent,
            ], None]
        ] = None,
        on_stopped: Optional[Callable[["AsteriskCallActor"], None]] = None,
    ):
        if not session_id:
            raise ValueError("session_id is required")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if enqueue_timeout <= 0:
            raise ValueError("enqueue_timeout must be positive")

        self.session_id = session_id
        self.state = initial_state
        self._handler = handler
        self._queue: asyncio.Queue[_Envelope] = asyncio.Queue(maxsize=queue_size)
        self._enqueue_timeout = enqueue_timeout
        self._on_transition = on_transition
        self._on_stopped = on_stopped
        self._timeout_task: Optional[asyncio.Task] = None
        self._dispatch_lock = asyncio.Lock()
        self._accepting = True
        self._task = asyncio.create_task(
            self._run(),
            name=f"aiavatar-asterisk-call-{session_id}",
        )

    @property
    def closed(self) -> bool:
        return not self._accepting or self._task.done()

    def transition(
        self,
        state: AsteriskCallState,
        event: AsteriskCallEvent,
    ) -> None:
        if state == self.state:
            return
        previous = self.state
        if state not in CALL_STATE_TRANSITIONS[previous]:
            raise AsteriskCallTransitionError(
                "Invalid Asterisk call state transition: "
                f"{previous.value} -> {state.value} "
                f"({type(event).__name__})"
            )
        self.cancel_timeout()
        self.state = state
        if state == AsteriskCallState.CLOSED:
            self._accepting = False
        if self._on_transition:
            self._on_transition(self, previous, state, event)

    def arm_timeout(
        self,
        event: AsteriskCallEvent,
        *,
        delay: float,
    ) -> None:
        """Dispatch an event later if the actor is still in its current state."""

        if delay <= 0:
            raise ValueError("delay must be positive")
        if self.closed:
            raise AsteriskCallActorClosed(
                f"Asterisk call actor is closed: {self.session_id}"
            )
        self.cancel_timeout()
        expected_state = self.state

        async def dispatch_later() -> None:
            try:
                await asyncio.sleep(delay)
                if self.closed or self.state != expected_state:
                    return
                await self.dispatch(event, wait=False)
            except asyncio.CancelledError:
                return
            except AsteriskCallActorClosed:
                return
            except AsteriskCallActorQueueFull:
                logger.exception(
                    "Failed to enqueue Asterisk call timeout: "
                    "session=%s state=%s event=%s",
                    self.session_id,
                    self.state.value,
                    type(event).__name__,
                )
            finally:
                if self._timeout_task is asyncio.current_task():
                    self._timeout_task = None

        self._timeout_task = asyncio.create_task(
            dispatch_later(),
            name=f"aiavatar-asterisk-timeout-{self.session_id}",
        )

    def cancel_timeout(self) -> None:
        timeout_task = self._timeout_task
        self._timeout_task = None
        if (
            timeout_task is not None
            and not timeout_task.done()
            and timeout_task is not asyncio.current_task()
        ):
            timeout_task.cancel()

    async def dispatch(
        self,
        event: AsteriskCallEvent,
        *,
        wait: bool = True,
    ) -> Any:
        result = asyncio.get_running_loop().create_future() if wait else None
        envelope = _Envelope(event=event, result=result)
        async with self._dispatch_lock:
            if self.closed:
                raise AsteriskCallActorClosed(
                    f"Asterisk call actor is closed: {self.session_id}"
                )
            try:
                await asyncio.wait_for(
                    self._queue.put(envelope),
                    timeout=self._enqueue_timeout,
                )
            except asyncio.TimeoutError as ex:
                raise AsteriskCallActorQueueFull(
                    f"Asterisk call event queue is full: {self.session_id}"
                ) from ex
        if result is not None:
            return await result
        return None

    async def wait_closed(self) -> None:
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            if not self._task.cancelled():
                raise

    async def cancel(self) -> None:
        self.cancel_timeout()
        if not self._task.done():
            self._task.cancel()
        await self.wait_closed()

    async def _run(self) -> None:
        try:
            while True:
                envelope = await self._queue.get()
                try:
                    result = await self._handler(self, envelope.event)
                except asyncio.CancelledError:
                    if envelope.result and not envelope.result.done():
                        envelope.result.cancel()
                    raise
                except Exception as ex:
                    logger.exception(
                        "Asterisk call event failed: session=%s state=%s event=%s",
                        self.session_id,
                        self.state.value,
                        type(envelope.event).__name__,
                    )
                    if envelope.result and not envelope.result.done():
                        envelope.result.set_exception(ex)
                else:
                    if envelope.result and not envelope.result.done():
                        envelope.result.set_result(result)
                finally:
                    self._queue.task_done()

                if self.state == AsteriskCallState.CLOSED:
                    break
        finally:
            self.cancel_timeout()
            async with self._dispatch_lock:
                self._accepting = False
                self._reject_pending()
            if self._on_stopped:
                self._on_stopped(self)

    def _reject_pending(self) -> None:
        while True:
            try:
                envelope = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if envelope.result and not envelope.result.done():
                envelope.result.set_exception(AsteriskCallActorClosed(
                    f"Asterisk call actor is closed: {self.session_id}"
                ))
            self._queue.task_done()
