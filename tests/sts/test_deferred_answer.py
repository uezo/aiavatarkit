import asyncio
from io import BytesIO
from types import SimpleNamespace
import wave

import pytest

from aiavatar.sts.deferred_answer import DEFAULT_RESUME_TEXT, DeferredAnswerScheduler
from aiavatar.sts.models import STSRequest, STSResponse


RESUME_TEXT = "$The user has paused. Answer the pending question now."
EMPTY_ANSWER = "<ack>はい。</ack><think>続きを待つ。</think><answer> \n </answer>"


class FakeVAD:
    def __init__(self):
        self.recording_sessions = {}
        self.contexts = {}

    def on_voiced(self, handler):
        self.voiced = handler
        return handler

    def get_session_data(self, session_id, key):
        if key == "context_id":
            return self.contexts.get(session_id)


class FakePipeline:
    def __init__(self):
        self.vad = FakeVAD()
        self.invocations = asyncio.Queue()
        self.responses = asyncio.Queue()

    def can_accept(self, handler):
        self.before_accept = handler
        return handler

    def on_before_llm(self, handler):
        self.before_llm = handler
        return handler

    def on_response(self, handler):
        self.observe_response = handler
        return handler

    def on_finalize(self, handler):
        self.finalize = handler
        return handler

    def on_shutdown(self, handler):
        self.shutdown = handler
        return handler

    async def invoke(self, request):
        if not self.before_accept(request):
            return
        await self.before_llm(request)
        self.invocations.put_nowait(request)
        response = response_for(request)
        yield response
        await self.observe_response(request, response)

    async def handle_response(self, response):
        self.responses.put_nowait(response)


def response_for(request, *, type="final", text=EMPTY_ANSWER, **fields):
    return STSResponse(
        type=type, session_id=request.session_id, context_id=request.context_id,
        user_id=request.user_id, transaction_id=request.transaction_id, text=text, **fields,
    )


def accept_speech(sts, session_id="session", context_id="context"):
    request = STSRequest(
        session_id=session_id, context_id=context_id, user_id="user", channel="websocket",
        audio_data=b"recording", audio_duration=1.0,
        text="紅茶を選ぶならどれがいい？", system_prompt_params={"name": "Aki"},
    )
    sts.vad.contexts[session_id] = context_id
    assert sts.before_accept(request)
    return request


@pytest.mark.asyncio
async def test_empty_answer_resumes_once_in_the_same_conversation():
    sts = FakePipeline()
    scheduler = DeferredAnswerScheduler(silence_seconds=0)
    scheduler.bind(sts)
    try:
        source = accept_speech(sts)
        assert not scheduler.is_resume(source)
        await sts.observe_response(source, response_for(source))
        resumed = await asyncio.wait_for(sts.invocations.get(), timeout=1)
        assert scheduler.is_resume(resumed)
        assert resumed.text == DEFAULT_RESUME_TEXT
        assert resumed.transaction_id != source.transaction_id
        assert (resumed.session_id, resumed.context_id, resumed.user_id, resumed.channel) == (
            source.session_id, source.context_id, source.user_id, source.channel,
        )
        assert resumed.system_prompt_params == source.system_prompt_params
        assert not resumed.audio_data
        final = await asyncio.wait_for(sts.responses.get(), timeout=1)
        assert final.transaction_id == resumed.transaction_id

        await sts.observe_response(source, response_for(source))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert sts.invocations.empty()
    finally:
        await sts.shutdown()


@pytest.mark.asyncio
async def test_custom_empty_answer_pattern_resumes_an_alternate_format():
    sts = FakePipeline()
    scheduler = DeferredAnswerScheduler(
        resume_text=RESUME_TEXT, silence_seconds=0,
        empty_answer_pattern=r"<reply>\s*</reply>",
    )
    scheduler.bind(sts)
    try:
        source = accept_speech(sts)
        await sts.observe_response(source, response_for(source, text="<reply>\n</reply>"))
        resumed = await asyncio.wait_for(sts.invocations.get(), timeout=1)
        assert scheduler.is_resume(resumed)
        assert resumed.text == RESUME_TEXT
    finally:
        await sts.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "before_final", "enabled"), [
    ("<ack>はい。</ack><think></think><answer>ケニアがおすすめです。</answer>", None, True),
    (EMPTY_ANSWER, {"type": "tool_call"}, True),
    (EMPTY_ANSWER, {"type": "chunk", "metadata": {"is_guardrail_triggered": True}}, True),
    ("はい。", None, True),
    (EMPTY_ANSWER, None, False),
], ids=["answer", "tool", "guardrail", "untagged", "disabled"])
async def test_only_enabled_deferred_answers_schedule_a_resume(text, before_final, enabled):
    sts = FakePipeline()
    DeferredAnswerScheduler(
        resume_text=RESUME_TEXT, silence_seconds=0, is_enabled=lambda _session_id: enabled,
    ).bind(sts)
    try:
        source = accept_speech(sts)
        if before_final:
            await sts.observe_response(source, response_for(source, **before_final))
        await sts.observe_response(source, response_for(source, text=text))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert sts.invocations.empty()
    finally:
        await sts.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["speech", "accepted", "disconnect", "shutdown"])
async def test_pending_resume_yields_to_new_input_or_disconnect(replacement):
    sts = FakePipeline()
    DeferredAnswerScheduler(resume_text=RESUME_TEXT, silence_seconds=0).bind(sts)
    try:
        source = accept_speech(sts)
        await sts.observe_response(source, response_for(source))

        if replacement == "speech":
            accept_speech(sts)
        elif replacement == "accepted":
            assert sts.before_accept(STSRequest(session_id=source.session_id, context_id=source.context_id))
        elif replacement == "disconnect":
            await sts.finalize(source.session_id)
        else:
            await sts.shutdown()
            await asyncio.sleep(0)
            assert sts.invocations.empty()
            return

        # Work for an independent connection can proceed after the first one stops waiting.
        other = accept_speech(sts, session_id="other-session", context_id="other-context")
        await sts.observe_response(other, response_for(other))
        resumed = await asyncio.wait_for(sts.invocations.get(), timeout=1)
        assert resumed.session_id == other.session_id
        assert sts.invocations.empty()
    finally:
        await sts.shutdown()


@pytest.mark.asyncio
async def test_recording_delays_resume_until_recording_ends():
    sts = FakePipeline()
    DeferredAnswerScheduler(resume_text=RESUME_TEXT, silence_seconds=0).bind(sts)
    try:
        source = accept_speech(sts)
        recording = SimpleNamespace(is_recording=True)
        sts.vad.recording_sessions[source.session_id] = recording
        await sts.vad.voiced(source.session_id)
        await sts.observe_response(source, response_for(source))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert sts.invocations.empty()

        recording.is_recording = False
        resumed = await asyncio.wait_for(sts.invocations.get(), timeout=1)
        assert resumed.session_id == source.session_id
        assert resumed.text == RESUME_TEXT
    finally:
        await sts.shutdown()


@pytest.mark.asyncio
async def test_silence_delay_applies_to_each_turn_without_voiced_notices():
    sts = FakePipeline()
    DeferredAnswerScheduler(resume_text=RESUME_TEXT, silence_seconds=0.02).bind(sts)
    loop = asyncio.get_running_loop()
    try:
        for _ in range(2):
            started_at = loop.time()
            source = accept_speech(sts)
            await sts.observe_response(source, response_for(source))
            await asyncio.wait_for(sts.invocations.get(), timeout=1)
            assert loop.time() - started_at >= 0.02
    finally:
        await sts.shutdown()


@pytest.mark.asyncio
async def test_new_speech_invalidates_a_resume_before_queue_acceptance():
    queued = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class QueuedPipeline(FakePipeline):
        async def invoke(self, request):
            queued.set()
            await release.wait()
            async for response in super().invoke(request):
                yield response
            finished.set()

    sts = QueuedPipeline()
    DeferredAnswerScheduler(resume_text=RESUME_TEXT, silence_seconds=0).bind(sts)
    try:
        source = accept_speech(sts)
        await sts.observe_response(source, response_for(source))
        await asyncio.wait_for(queued.wait(), timeout=1)
        accept_speech(sts)
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        assert sts.invocations.empty()
    finally:
        await sts.shutdown()


@pytest.mark.asyncio
async def test_ack_delay_uses_available_wav_frames_even_with_placeholder_size():
    sts = FakePipeline()
    DeferredAnswerScheduler(resume_text=RESUME_TEXT, silence_seconds=0).bind(sts)
    try:
        source = accept_speech(sts)
        wav_file = BytesIO()
        with wave.open(wav_file, "wb") as output:
            output.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
            output.writeframes(bytes(4_000))  # 0.125 seconds of audio.
        audio = bytearray(wav_file.getvalue())
        size_field = audio.index(b"data") + len(b"data")
        audio[size_field:size_field + 4] = b"\xff" * 4

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        await sts.observe_response(source, response_for(
            source, type="chunk", voice_text_tag="ack", audio_data=bytes(audio),
        ))
        await sts.observe_response(source, response_for(source))
        await asyncio.wait_for(sts.invocations.get(), timeout=1)
        assert loop.time() - started_at >= 0.125
    finally:
        await sts.shutdown()


def test_scheduler_registers_hooks_only_when_bound_once():
    sts = FakePipeline()
    scheduler = DeferredAnswerScheduler()
    assert not hasattr(sts, "before_accept")

    scheduler.bind(sts)
    assert scheduler.sts is sts
    assert hasattr(sts, "before_accept")
    with pytest.raises(RuntimeError, match="already bound"):
        scheduler.bind(FakePipeline())
