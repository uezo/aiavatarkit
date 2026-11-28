"""Exercise InterruptionContext with a fake pipeline. No providers, files, or network."""

import pytest

from aiavatar.sts.interruption import InterruptionContext
from aiavatar.sts.models import STSRequest, STSResponse


SESSION = "session"
CONTEXT = "conversation"


class FakeVAD:
    def on_speech_started(self, handler):
        self.speech_started = handler
        return handler


class FakePipeline:
    def __init__(self):
        self.vad = FakeVAD()
        self.channels = {}

    def on_before_start(self, handler=None, *, channels=None):
        self.before_start = handler
        self.channels["before_start"] = channels
        return handler

    def on_before_llm(self, handler=None, *, channels=None):
        self.before_llm = handler
        self.channels["before_llm"] = channels
        return handler

    async def run_hooks(self, request):
        await self.before_start(request)
        await self.before_llm(request)

    def on_finalize(self, handler):
        self.finalize = handler
        return handler

    def on_shutdown(self, handler):
        self.shutdown = handler
        return handler


def capture(sts):
    metadata = {}
    sts.vad.speech_started(SESSION, metadata)
    return metadata


@pytest.mark.asyncio
async def test_notes_wrap_the_request_in_order_with_custom_templates():
    sts = FakePipeline()
    context = InterruptionContext(
        get_active_transaction_id={SESSION: "answer"}.get,
        spoken_template="Heard [{spoken_text}] {request_text}",
        discarded_template="Dropped [{discarded_text}] {request_text}",
    )
    context.bind(sts, channels="websocket")
    context.open_session(SESSION)
    context.observe(STSResponse(type="start", session_id=SESSION, context_id=CONTEXT, transaction_id="answer"))
    for text in ("ABCD", "EFGH"):
        context.observe(STSResponse(
            type="chunk", session_id=SESSION, transaction_id="answer", voice_text=text, audio_data=b"voice",
        ))
    context.handle_playback_event(
        SESSION, event="start", playback_id="first", transaction_id="answer", text="ABCD", duration_seconds=4,
    )
    context.handle_playback_event(SESSION, event="end", playback_id="first", completed=True)
    context.handle_playback_event(SESSION, event="discarded", transaction_id="answer", text="EFGH")

    request = STSRequest(
        session_id=SESSION, context_id=CONTEXT, text="<channel name='ws' />Wait", recognized_text="Wait",
        metadata=capture(sts),
    )
    assert InterruptionContext.get_interruption(request).spoken_text == "ABCD"

    # The interruption note comes before start; discard reports are read later.
    await sts.before_start(request)
    assert request.text == "Heard [ABCD] <channel name='ws' />Wait"
    await sts.before_llm(request)

    assert sts.channels == {"before_start": "websocket", "before_llm": "websocket"}
    assert request.text == "Dropped [EFGH] Heard [ABCD] <channel name='ws' />Wait"


@pytest.mark.asyncio
async def test_before_audio_note_quotes_the_recognized_text_of_unanswered_turns():
    sts = FakePipeline()
    active = {}
    context = InterruptionContext(
        get_active_transaction_id=active.get, before_audio_template="Pending [{previous_request}] {request_text}",
    )
    context.bind(sts)
    context.open_session(SESSION)

    first = STSRequest(
        session_id=SESSION, context_id=CONTEXT, transaction_id="first",
        text="$Current date and time: now\n\nPlan a trip.", recognized_text="Plan a trip.",
    )
    context.track(first)
    await sts.run_hooks(first)
    assert first.text == "$Current date and time: now\n\nPlan a trip."
    context.observe(STSResponse(type="start", session_id=SESSION, context_id=CONTEXT, transaction_id="first"))
    active[SESSION] = "first"

    # The first answer has sent no audio when the user speaks again.
    second = STSRequest(
        session_id=SESSION, context_id=CONTEXT, transaction_id="second",
        text="Add lunch.", recognized_text="Add lunch.", metadata=capture(sts),
    )
    await sts.run_hooks(second)
    assert second.text == "Pending [Plan a trip.] Add lunch."


@pytest.mark.asyncio
async def test_requests_outside_open_sessions_or_user_turns_are_unchanged():
    sts = FakePipeline()
    context = InterruptionContext(get_active_transaction_id=lambda _session_id: None)
    context.bind(sts)
    request = STSRequest(session_id=SESSION, context_id=CONTEXT, text="Hello", recognized_text="Hello")

    await sts.run_hooks(request)
    assert capture(sts) == {}

    context.open_session(SESSION)
    await sts.run_hooks(request)
    assert request.text == "Hello"
    assert context.get_ledger(SESSION) is not None

    await sts.finalize(SESSION)
    assert context.get_ledger(SESSION) is None
    assert capture(sts) == {}


def test_requires_a_vad_that_reports_speech_start():
    sts = FakePipeline()
    sts.vad = object()
    context = InterruptionContext(get_active_transaction_id=lambda _session_id: None)
    with pytest.raises(ValueError, match="on_speech_started"):
        context.bind(sts)


def test_plugin_registers_hooks_only_when_bound_once():
    sts = FakePipeline()
    context = InterruptionContext(get_active_transaction_id=lambda _session_id: None)
    assert not hasattr(sts, "before_start")

    context.bind(sts)
    assert hasattr(sts, "before_start")
    with pytest.raises(RuntimeError, match="already bound"):
        context.bind(FakePipeline())
