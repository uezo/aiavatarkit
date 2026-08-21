import asyncio

import pytest

from aiavatar.adapter.asterisk.actor import (
    AsteriskCallActor,
    AsteriskCallActorClosed,
    AsteriskCallActorQueueFull,
    AsteriskCallTransitionError,
)
from aiavatar.adapter.asterisk.state import (
    AsteriskCallState,
    HangupRequested,
    ReferTimedOut,
    TransferRequested,
)


@pytest.mark.asyncio
async def test_events_for_one_call_are_processed_in_order():
    transfer_started = asyncio.Event()
    release_transfer = asyncio.Event()
    handled = []

    async def handler(actor, event):
        handled.append(("start", type(event).__name__))
        if isinstance(event, TransferRequested):
            transfer_started.set()
            await release_transfer.wait()
        if isinstance(event, HangupRequested):
            actor.transition(AsteriskCallState.CLOSED, event)
        handled.append(("end", type(event).__name__))

    actor = AsteriskCallActor(session_id="call-1", handler=handler)
    transfer = asyncio.create_task(actor.dispatch(TransferRequested("operator")))
    await transfer_started.wait()
    hangup = asyncio.create_task(actor.dispatch(HangupRequested()))
    await asyncio.sleep(0)

    assert handled == [("start", "TransferRequested")]

    release_transfer.set()
    await asyncio.gather(transfer, hangup)
    await actor.wait_closed()

    assert handled == [
        ("start", "TransferRequested"),
        ("end", "TransferRequested"),
        ("start", "HangupRequested"),
        ("end", "HangupRequested"),
    ]


@pytest.mark.asyncio
async def test_closing_call_rejects_already_queued_events():
    hangup_started = asyncio.Event()
    release_hangup = asyncio.Event()

    async def handler(actor, event):
        if isinstance(event, HangupRequested):
            hangup_started.set()
            await release_hangup.wait()
            actor.transition(AsteriskCallState.CLOSED, event)

    actor = AsteriskCallActor(session_id="call-1", handler=handler)
    hangup = asyncio.create_task(actor.dispatch(HangupRequested()))
    await hangup_started.wait()
    stale_transfer = asyncio.create_task(
        actor.dispatch(TransferRequested("operator"))
    )
    await asyncio.sleep(0)

    release_hangup.set()
    await hangup
    with pytest.raises(AsteriskCallActorClosed):
        await stale_transfer
    await actor.wait_closed()


@pytest.mark.asyncio
async def test_slow_call_does_not_block_another_call_actor():
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    fast_completed = asyncio.Event()

    async def slow_handler(actor, event):
        slow_started.set()
        await release_slow.wait()
        actor.transition(AsteriskCallState.CLOSED, event)

    async def fast_handler(actor, event):
        fast_completed.set()
        actor.transition(AsteriskCallState.CLOSED, event)

    slow_actor = AsteriskCallActor(session_id="slow", handler=slow_handler)
    fast_actor = AsteriskCallActor(session_id="fast", handler=fast_handler)
    slow = asyncio.create_task(slow_actor.dispatch(HangupRequested()))
    await slow_started.wait()

    await asyncio.wait_for(
        fast_actor.dispatch(HangupRequested()),
        timeout=0.1,
    )
    assert fast_completed.is_set()

    release_slow.set()
    await slow
    await asyncio.gather(slow_actor.wait_closed(), fast_actor.wait_closed())


@pytest.mark.asyncio
async def test_invalid_state_transition_is_rejected():
    async def handler(actor, event):
        actor.transition(AsteriskCallState.BRIDGE_COMPLETED, event)

    actor = AsteriskCallActor(session_id="call-1", handler=handler)
    with pytest.raises(AsteriskCallTransitionError):
        await actor.dispatch(HangupRequested())
    await actor.cancel()


@pytest.mark.asyncio
async def test_event_queue_is_bounded():
    transfer_started = asyncio.Event()
    release_transfer = asyncio.Event()

    async def handler(actor, event):
        if isinstance(event, TransferRequested):
            transfer_started.set()
            await release_transfer.wait()
        if isinstance(event, HangupRequested):
            actor.transition(AsteriskCallState.CLOSED, event)

    actor = AsteriskCallActor(
        session_id="call-1",
        handler=handler,
        queue_size=1,
        enqueue_timeout=0.01,
    )
    transfer = asyncio.create_task(actor.dispatch(TransferRequested("operator")))
    await transfer_started.wait()
    hangup = asyncio.create_task(actor.dispatch(HangupRequested()))
    await asyncio.sleep(0)

    with pytest.raises(AsteriskCallActorQueueFull):
        await actor.dispatch(TransferRequested("sales"))

    release_transfer.set()
    await asyncio.gather(transfer, hangup)
    await actor.wait_closed()


@pytest.mark.asyncio
async def test_state_timeout_is_dispatched_through_actor_queue():
    timeout_seen = asyncio.Event()

    async def handler(actor, event):
        if isinstance(event, ReferTimedOut):
            timeout_seen.set()
            actor.transition(AsteriskCallState.CLEANING_UP, event)
            actor.transition(AsteriskCallState.CLOSED, event)

    actor = AsteriskCallActor(session_id="call-1", handler=handler)
    actor.arm_timeout(ReferTimedOut(), delay=0.01)

    await asyncio.wait_for(timeout_seen.wait(), timeout=0.1)
    await actor.wait_closed()


@pytest.mark.asyncio
async def test_state_transition_cancels_armed_timeout():
    timeout_seen = asyncio.Event()

    async def handler(actor, event):
        if isinstance(event, ReferTimedOut):
            timeout_seen.set()
        if isinstance(event, HangupRequested):
            actor.transition(AsteriskCallState.CLEANING_UP, event)
            actor.transition(AsteriskCallState.CLOSED, event)

    actor = AsteriskCallActor(session_id="call-1", handler=handler)
    actor.arm_timeout(ReferTimedOut(), delay=0.01)
    await actor.dispatch(HangupRequested())
    await actor.wait_closed()
    await asyncio.sleep(0.02)

    assert timeout_seen.is_set() is False


@pytest.mark.asyncio
async def test_enqueue_racing_with_actor_close_never_waits_forever():
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(actor, event):
        handler_started.set()
        await release_handler.wait()
        actor.transition(AsteriskCallState.CLOSED, event)

    actor = AsteriskCallActor(
        session_id="call-1",
        handler=handler,
        queue_size=1,
        enqueue_timeout=0.02,
    )
    first = asyncio.create_task(actor.dispatch(HangupRequested()))
    await handler_started.wait()
    await actor.dispatch(TransferRequested("operator"), wait=False)
    racing = asyncio.create_task(actor.dispatch(TransferRequested("sales")))

    release_handler.set()
    await first
    with pytest.raises((AsteriskCallActorClosed, AsteriskCallActorQueueFull)):
        await asyncio.wait_for(racing, timeout=0.1)
    await actor.wait_closed()


@pytest.mark.asyncio
async def test_canceling_waiter_does_not_cancel_actor_cleanup():
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(actor, event):
        handler_started.set()
        await release_handler.wait()
        actor.transition(AsteriskCallState.CLOSED, event)

    actor = AsteriskCallActor(session_id="call-1", handler=handler)
    dispatch = asyncio.create_task(actor.dispatch(HangupRequested()))
    await handler_started.wait()
    waiter = asyncio.create_task(actor.wait_closed())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert actor._task.done() is False
    release_handler.set()
    await dispatch
    await actor.wait_closed()
