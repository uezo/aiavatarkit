import asyncio
from dataclasses import FrozenInstanceError

import pytest

from aiavatar.sts import STSPipeline
from aiavatar.sts.dialog_observer import (
    DialogObservation,
    DialogObserver,
    DialogRequest,
    DialogResponse,
)
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.models import STSRequest
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy


def make_observation(
    transaction_id: str,
    text: str,
    *,
    session_id: str = "session",
    response_text: str = None,
) -> DialogObservation:
    return DialogObservation(
        session_id=session_id,
        transaction_id=transaction_id,
        context_id="context",
        user_id="user",
        request=DialogRequest(text=text),
        response=(
            DialogResponse(text=response_text)
            if response_text is not None
            else None
        ),
    )


def test_dialog_snapshots_are_immutable_and_detached():
    files = [{"url": "data:image/png;base64,abc", "tags": ["one"]}]
    metadata = {"nested": {"value": 1}}
    audio_data = b"audio"
    request = STSRequest(
        text="original",
        audio_data=audio_data,
        audio_duration=1.25,
        files=files,
        metadata=metadata,
        channel="websocket",
    )
    snapshot = DialogRequest.from_sts_request(request, text="recognized")

    files[0]["url"] = "changed"
    metadata["nested"]["value"] = 2

    assert snapshot.text == "recognized"
    assert snapshot.audio_data is audio_data
    assert snapshot.files[0]["url"] == "data:image/png;base64,abc"
    assert snapshot.metadata["nested"]["value"] == 1
    with pytest.raises(TypeError):
        snapshot.files[0]["url"] = "cannot-mutate"
    with pytest.raises(TypeError):
        snapshot.metadata["new"] = "cannot-mutate"
    with pytest.raises(FrozenInstanceError):
        snapshot.text = "cannot-mutate"


def test_dialog_observer_rejects_sync_functions():
    observer = DialogObserver()

    with pytest.raises(TypeError, match="must be async"):
        observer.on_request(lambda observation, state: None)


@pytest.mark.asyncio
async def test_state_is_a_detached_dict_and_supports_non_string_keys():
    observer = DialogObserver()
    await observer.update_state("session", {1: {"value": "original"}})

    state = observer.get_state("session")
    state[1]["value"] = "changed"
    state[2] = "local-only"

    assert isinstance(state, dict)
    assert observer.get_state("session") == {1: {"value": "original"}}
    await observer.shutdown()


@pytest.mark.asyncio
async def test_request_observer_is_latest_wins_and_discards_canceled_patch():
    observer = DialogObserver()
    first_started = asyncio.Event()
    first_canceled = asyncio.Event()

    @observer.on_request
    async def observe_request(observation, state):
        if observation.request.text == "first":
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_canceled.set()
                # Even a cancellation-resistant observer cannot commit this.
                return {"value": "stale"}
        return {"value": observation.request.text}

    observer.observe_request(make_observation("transaction-1", "first"))
    await asyncio.wait_for(first_started.wait(), timeout=1)

    observer.observe_request(make_observation("transaction-2", "second"))
    await asyncio.wait_for(observer.wait_for_idle("session"), timeout=1)

    assert first_canceled.is_set()
    assert observer.get_state("session") == {"value": "second"}
    await observer.shutdown()


@pytest.mark.asyncio
async def test_request_and_response_slots_are_independent():
    observer = DialogObserver()
    response_started = asyncio.Event()
    release_response = asyncio.Event()
    response_canceled = False

    async def observe(observation, state):
        nonlocal response_canceled
        if observation.response is not None:
            response_started.set()
            try:
                await release_response.wait()
            except asyncio.CancelledError:
                response_canceled = True
                raise
            return {"response": observation.response.text}
        return {"request": observation.request.text}

    observer.on_request(observe)
    observer.on_response(observe)
    observer.observe_response(
        make_observation(
            "transaction-1",
            "first",
            response_text="first-response",
        )
    )
    await asyncio.wait_for(response_started.wait(), timeout=1)

    observer.observe_request(make_observation("transaction-2", "second"))
    await asyncio.sleep(0)
    assert not response_canceled

    release_response.set()
    await asyncio.wait_for(observer.wait_for_idle("session"), timeout=1)
    assert observer.get_state("session") == {
        "request": "second",
        "response": "first-response",
    }
    await observer.shutdown()


@pytest.mark.asyncio
async def test_clear_session_cancels_tasks_and_deletes_state():
    observer = DialogObserver()
    started = asyncio.Event()
    canceled = asyncio.Event()

    await observer.update_state("session", {"existing": True})

    @observer.on_request
    async def observe_request(observation, state):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            canceled.set()
            raise

    observer.observe_request(make_observation("transaction", "request"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(observer.clear_session("session"), timeout=1)

    assert canceled.is_set()
    assert observer.get_state("session") == {}
    await observer.shutdown()


@pytest.mark.asyncio
async def test_pipeline_observes_processed_request_and_final_response(tmp_path):
    observer = DialogObserver()
    request_observations = []
    response_observations = []

    @observer.on_request
    async def record_request_observation(observation, state):
        request_observations.append(observation)
        return {"request_seen": observation.request.text}

    @observer.on_response
    async def record_response_observation(observation, state):
        response_observations.append(observation)
        return {"response_seen": observation.response.text}

    db_path = str(tmp_path / "dialog-observer.db")
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=LLMServiceDummy(
            response_text="Final answer.",
            db_connection_str=db_path,
        ),
        tts=SpeechSynthesizerDummy(),
        db_connection_str=db_path,
        voice_recorder_enabled=False,
        insert_channel_tag=True,
    )

    @pipeline.on_before_llm
    async def observe_dialog_request(request):
        observer.observe_request(DialogObservation(
            session_id=request.session_id,
            transaction_id=request.transaction_id,
            context_id=request.context_id,
            user_id=request.user_id,
            request=DialogRequest.from_sts_request(request),
        ))

    @pipeline.on_finish
    async def observe_dialog_response(request, response):
        observer.observe_response(DialogObservation(
            session_id=request.session_id,
            transaction_id=request.transaction_id,
            context_id=response.context_id,
            user_id=response.user_id,
            request=DialogRequest.from_sts_request(request),
            response=DialogResponse.from_sts_response(response),
        ))

    request = STSRequest(
        session_id="session",
        context_id=None,
        user_id="user",
        text="Original request",
        audio_data=b"immutable-audio",
        audio_duration=2.5,
        files=[{"url": "data:image/png;base64,abc"}],
        metadata={"source": "test"},
        channel="websocket",
    )

    responses = [response async for response in pipeline.invoke(request)]
    final_response = next(response for response in responses if response.type == "final")
    await asyncio.wait_for(observer.wait_for_idle("session"), timeout=1)

    assert len(request_observations) == 1
    request_observation = request_observations[0]
    assert request_observation.session_id == "session"
    assert request_observation.transaction_id == request.transaction_id
    assert request_observation.context_id == final_response.context_id
    assert request_observation.user_id == "user"
    assert request_observation.request.text.startswith(
        "<channel name='websocket' />Original request"
    )
    assert request_observation.request.audio_data == b"immutable-audio"
    assert request_observation.request.audio_duration == 2.5
    assert request_observation.request.files[0]["url"].startswith("data:image")
    assert request_observation.request.metadata == {"source": "test"}

    assert len(response_observations) == 1
    response_observation = response_observations[0]
    assert response_observation.transaction_id == request_observation.transaction_id
    assert response_observation.request.text == request_observation.request.text
    assert response_observation.response.text == "Final answer."
    assert observer.get_state("session") == {
        "request_seen": "<channel name='websocket' />Original request",
        "response_seen": "Final answer.",
    }

    await pipeline.shutdown()
    await observer.shutdown()
    await pipeline.stt.close()
    await pipeline.tts.close()
