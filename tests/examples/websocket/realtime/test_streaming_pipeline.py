"""Startup ordering and cleanup at the example-local streaming boundary."""

import asyncio

import pytest

from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.pipeline import ResponseHandler
from examples.websocket.realtime.pipeline.streaming import StreamingSTSPipeline


class FakeStreamingPipeline(StreamingSTSPipeline):
    def __init__(self):
        super().__init__()
        self.responses = []
        self.audio = []
        self.tasks = {}
        self.fail_start = False
        self.add_response_handler(ResponseHandler(
            can_handle=lambda session_id: True,
            handle_response=self.capture,
            stop_response=self.stop,
        ))

    async def capture(self, response):
        self.responses.append(response)

    async def stop(self, session_id, context_id):
        pass

    async def start_session(self, request):
        if self.fail_start:
            raise RuntimeError("Startup failed")
        self._remember_session(request)
        ready = asyncio.get_running_loop().create_future()

        async def receive():
            await self._session_connected(STSResponse(
                type="connected", session_id=request.session_id,
                user_id=request.user_id, context_id=request.context_id,
            ), ready)
            await self.handle_response(STSResponse(type="chunk", session_id=request.session_id, text="Ready"))

        self.tasks[request.session_id] = asyncio.create_task(receive())
        try:
            await ready
        except BaseException:
            await self.finalize(request.session_id)
            raise

    async def process_audio_samples(self, samples, context_id):
        self.audio.append((context_id, samples))

    async def finalize(self, session_id):
        self._release_session(session_id)
        if task := self.tasks.pop(session_id, None):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def shutdown(self):
        for session_id in list(self.tasks):
            await self.finalize(session_id)


@pytest.mark.asyncio
async def test_prepared_start_holds_output_until_adapter_activation():
    pipeline = FakeStreamingPipeline()
    request = STSRequest(
        session_id="session", user_id="user", context_id="context", channel="websocket",
        system_prompt_params={"name": "Guest"}, metadata={"barge_in_enabled": False},
    )
    try:
        await pipeline.prepare_session(request)
        assert pipeline.responses == []
        assert not pipeline.tasks["session"].done()
        for key in ("user_id", "context_id", "channel", "system_prompt_params"):
            assert pipeline.vad.get_session_data("session", key) == getattr(request, key)
        assert pipeline.vad.get_session_data("session", "barge_in_enabled") is False

        pipeline.activate_session("session")
        pipeline.activate_session("session")
        await pipeline.tasks["session"]
        assert [response.type for response in pipeline.responses] == ["chunk"]
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
async def test_failed_prepare_releases_gate_and_allows_retry():
    pipeline = FakeStreamingPipeline()
    pipeline.fail_start = True
    with pytest.raises(RuntimeError, match="Startup failed"):
        await pipeline.prepare_session(STSRequest(session_id="session"))
    assert pipeline._prepared_sessions == {}
    assert pipeline.vad._session_data == {}

    pipeline.fail_start = False
    try:
        await pipeline.prepare_session(STSRequest(session_id="session"))
        assert "session" in pipeline._prepared_sessions
    finally:
        await pipeline.shutdown()


@pytest.mark.asyncio
async def test_cancelled_prepare_releases_gate_and_metadata():
    pipeline = FakeStreamingPipeline()
    starting = asyncio.Event()

    async def blocked_start(request):
        pipeline._remember_session(request)
        starting.set()
        await asyncio.Future()

    pipeline.start_session = blocked_start
    task = asyncio.create_task(pipeline.prepare_session(STSRequest(session_id="session")))
    try:
        await starting.wait()
        gate = pipeline._prepared_sessions["session"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.is_set()
        assert pipeline._prepared_sessions == {}
        assert pipeline.vad._session_data == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [False, True])
async def test_duplicate_prepare_preserves_active_session(prepared):
    pipeline = FakeStreamingPipeline()
    request = STSRequest(session_id="session", user_id="original")
    try:
        if prepared:
            await pipeline.prepare_session(request)
        else:
            await pipeline.start_session(request)
        task = pipeline.tasks["session"]
        gate = pipeline._prepared_sessions.get("session")
        with pytest.raises(ValueError, match="already active"):
            await pipeline.prepare_session(STSRequest(session_id="session", user_id="replacement"))
        assert pipeline.tasks["session"] is task
        assert pipeline._prepared_sessions.get("session") is gate
        assert pipeline.vad.get_session_data("session", "user_id") == "original"
    finally:
        await pipeline.shutdown()
