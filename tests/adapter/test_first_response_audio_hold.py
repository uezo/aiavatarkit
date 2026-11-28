import asyncio
from types import SimpleNamespace

import pytest

from aiavatar.adapter.barge_in import FirstResponseAudioHold
from aiavatar.adapter.models import AIAvatarResponse
from aiavatar.sts.models import STSRequest, STSResponse


CHANNEL = "websocket"
SESSION = "session"


class FakePipeline:
    def __init__(self):
        self.vad = SimpleNamespace(
            recording_sessions={}, on_recording_started_min_duration=1.5,
            get_session_data=lambda _session_id, key: True if key == "barge_in_enabled" else None,
        )

    def on_accepted(self, callback=None, *, channels=None):
        def register(handler):
            self.accepted = handler
            self.channels = channels
            return handler

        return register if callback is None else register(callback)

    def on_finalize(self, handler):
        self.finalize = handler
        return handler

    def on_shutdown(self, handler):
        self.shutdown = handler
        return handler

    def start_recording(self, session_id=SESSION, *, duration=0.3, silence=0.0):
        recording = SimpleNamespace(
            recording_id=f"recording-{session_id}", record_duration=duration, silence_duration=silence,
            is_recording=True, on_recording_started_triggered=False,
        )
        self.vad.recording_sessions[session_id] = recording
        return recording


class FakeAdapter:
    def __init__(self, sts):
        self.sts = sts
        self.channel = CHANNEL

    def on_response(self, handler):
        self.prepare_response = handler
        return handler

    async def prepare_audio(self, request, *, tag="answer", audio=b"wave", guardrail=False):
        fields = dict(
            type="chunk", session_id=request.session_id, transaction_id=request.transaction_id,
            voice_text="Hello.", voice_text_tag=tag, audio_data=audio,
            metadata={"is_guardrail_triggered": guardrail},
        )
        await self.prepare_response(AIAvatarResponse(**fields), STSResponse(**fields))


@pytest.fixture
def hold():
    sts = FakePipeline()
    adapter = FakeAdapter(sts)
    FirstResponseAudioHold(applies_to=lambda request: bool(request.audio_data)).bind(adapter)
    return sts, adapter


async def accept_voice(sts, session_id=SESSION):
    request = STSRequest(session_id=session_id, channel=CHANNEL, audio_data=b"voice")
    await sts.accepted(request)
    return request


@pytest.mark.asyncio
async def test_ack_and_later_audio_pass_through_first_answer_hold(hold):
    sts, adapter = hold
    request = await accept_voice(sts)
    recording = sts.start_recording()

    async with asyncio.timeout(0.5):
        assert sts.channels == CHANNEL
        await adapter.prepare_audio(request, tag="ack")
        await adapter.prepare_audio(request, audio=None)

        first_answer = asyncio.create_task(adapter.prepare_audio(request))
        try:
            await asyncio.sleep(0)
            assert not first_answer.done()
            recording.is_recording = False
            await first_answer
        finally:
            if not first_answer.done():
                first_answer.cancel()
                await asyncio.gather(first_answer, return_exceptions=True)

        sts.start_recording()
        await adapter.prepare_audio(request)
        await sts.shutdown()


@pytest.mark.asyncio
async def test_hold_waits_only_for_remaining_voiced_duration(hold):
    sts, adapter = hold
    request = await accept_voice(sts)
    recording = sts.start_recording(duration=1.6, silence=0.18)
    loop = asyncio.get_running_loop()
    started_at = loop.time()

    async with asyncio.timeout(0.5):
        await adapter.prepare_audio(request)

    assert loop.time() - started_at >= 0.075
    assert recording.is_recording
    assert not recording.on_recording_started_triggered
    await sts.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("release", ["confirmed-voice", "new-request", "finalize", "shutdown"])
async def test_an_early_boundary_releases_held_audio(hold, release):
    sts, adapter = hold
    request = await accept_voice(sts)
    recording = sts.start_recording()
    first_answer = asyncio.create_task(adapter.prepare_audio(request))
    try:
        async with asyncio.timeout(0.5):
            await asyncio.sleep(0)
            assert not first_answer.done()
            if release == "confirmed-voice":
                recording.on_recording_started_triggered = True
            elif release == "new-request":
                await sts.accepted(STSRequest(session_id=SESSION, channel=CHANNEL, text="Follow-up"))
            elif release == "finalize":
                await sts.finalize(SESSION)
            else:
                await sts.shutdown()
            await first_answer
    finally:
        if not first_answer.done():
            first_answer.cancel()
            await asyncio.gather(first_answer, return_exceptions=True)
        await sts.shutdown()


def test_hold_registers_hooks_only_when_bound_once():
    sts = FakePipeline()
    audio_hold = FirstResponseAudioHold(applies_to=lambda request: True)
    assert not hasattr(sts, "accepted")

    audio_hold.bind(FakeAdapter(sts))
    assert sts.channels == CHANNEL
    with pytest.raises(RuntimeError, match="already bound"):
        audio_hold.bind(FakeAdapter(FakePipeline()))
