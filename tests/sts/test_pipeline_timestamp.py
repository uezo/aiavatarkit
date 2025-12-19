from datetime import datetime, timezone, timedelta
import pytest
from aiavatar.sts import STSPipeline
from aiavatar.sts.models import STSRequest
from aiavatar.sts.llm import LLMResponse
from aiavatar.sts.session_state_manager import SessionStateManager, SessionState
from aiavatar.sts.vad import SpeechDetectorDummy


class DummyContextManager:
    async def get_last_created_at(self, context_id: str):
        return datetime.min.replace(tzinfo=timezone.utc)


class DummyLLM:
    def __init__(self):
        self.context_manager = DummyContextManager()

    async def chat_stream(self, context_id, user_id, text, files, system_prompt_params):
        # Minimal single-chunk stream
        yield LLMResponse(context_id=context_id, text="ok", voice_text="ok")


class DummyTTS:
    async def synthesize(self, text: str, style_info=None, language=None) -> bytes:
        return b""


class DummyPerformanceRecorder:
    def record(self, performance):
        pass

    def close(self):
        pass


class InMemorySessionStateManager(SessionStateManager):
    def __init__(self):
        self.store = {}

    async def get_session_state(self, session_id: str) -> SessionState:
        if session_id not in self.store:
            now = datetime.now(timezone.utc)
            self.store[session_id] = SessionState(session_id, updated_at=now, created_at=now)
        return self.store[session_id]

    async def update_transaction(self, session_id: str, transaction_id: str, timestamp_inserted_at=None) -> None:
        state = await self.get_session_state(session_id)
        state.active_transaction_id = transaction_id
        if timestamp_inserted_at:
            state.timestamp_inserted_at = timestamp_inserted_at
        state.updated_at = datetime.now(timezone.utc)

    async def update_previous_request(self, session_id: str, timestamp: datetime, text, files) -> None:
        state = await self.get_session_state(session_id)
        state.previous_request_timestamp = timestamp
        state.previous_request_text = text
        state.previous_request_files = files
        state.updated_at = datetime.now(timezone.utc)

    async def clear_session(self, session_id: str) -> None:
        self.store.pop(session_id, None)

    async def cleanup_old_sessions(self, timeout_seconds: int = 3600) -> None:
        pass


async def _collect_start_text(pipeline: STSPipeline, request: STSRequest):
    start_text = None
    async for resp in pipeline.invoke(request):
        if resp.type == "start":
            start_text = resp.metadata["request_text"]
    return start_text


@pytest.mark.asyncio
async def test_no_timestamp_when_interval_zero():
    session_manager = InMemorySessionStateManager()
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=None,
        llm=DummyLLM(),
        tts=DummyTTS(),
        performance_recorder=DummyPerformanceRecorder(),
        session_state_manager=session_manager,
        voice_recorder_enabled=False,
        timestamp_interval_seconds=0,
    )

    start_text = await _collect_start_text(
        pipeline,
        STSRequest(session_id="s1", user_id="u", text="hello"),
    )

    assert start_text == "hello"
    state = await session_manager.get_session_state("s1")
    assert state.timestamp_inserted_at == datetime.min.replace(tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_timestamp_inserted_and_respects_interval():
    session_manager = InMemorySessionStateManager()
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=None,
        llm=DummyLLM(),
        tts=DummyTTS(),
        performance_recorder=DummyPerformanceRecorder(),
        session_state_manager=session_manager,
        voice_recorder_enabled=False,
        timestamp_interval_seconds=5,
    )

    # First call: should insert timestamp
    first_text = await _collect_start_text(
        pipeline,
        STSRequest(session_id="s2", user_id="u", text="first"),
    )
    assert first_text.startswith(pipeline.timestamp_prefix)
    state = await session_manager.get_session_state("s2")
    first_inserted = state.timestamp_inserted_at
    assert first_inserted != datetime.min

    # Second call immediately: should not insert again
    second_text = await _collect_start_text(
        pipeline,
        STSRequest(session_id="s2", user_id="u", text="second"),
    )
    assert second_text == "second"
    state = await session_manager.get_session_state("s2")
    assert state.timestamp_inserted_at == first_inserted

    # Simulate time passing beyond interval
    state.timestamp_inserted_at = datetime.now(timezone.utc) - timedelta(seconds=6)

    third_text = await _collect_start_text(
        pipeline,
        STSRequest(session_id="s2", user_id="u", text="third"),
    )
    assert third_text.startswith(pipeline.timestamp_prefix)
    state = await session_manager.get_session_state("s2")
    assert state.timestamp_inserted_at > first_inserted
