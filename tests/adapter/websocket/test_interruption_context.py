"""Exercise interruption context using a real VAD and queue with local fakes."""

import asyncio
import base64
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import pytest

from aiavatar.adapter import interruption as adapter_interruption
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData
from aiavatar.sts.llm import LLMResponse, LLMServiceDummy
from aiavatar.sts.models import STSResponse
from aiavatar.sts.stt.base import SpeechRecognitionResult
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_taking_gates import TurnTakingDecision, TurnTakingGate
from aiavatar.sts.vad.turn_taking_gates import session as playback_session


SESSION_ID = "connection"
USER_ID = "test-user"
FIRST_REQUEST = 'Plan a trip to "Kyoto".'
FOLLOW_UP = 'What about "lunch"?'
ADDITIONAL_REQUEST = "Add a museum visit too."
NORMAL_REQUEST = "Thanks!"
FIRST_RESPONSE = "ABCDEFGH"
ACK_TEXT = "そうなんですね。"
VOICE = b"\x01\x00" * 1600
SILENCE = bytes(3200)
SYSTEM_PROMPT = "Reply to {name}."
BEFORE_AUDIO_TEMPLATE = "Pending:\n{previous_request}\n\nCurrent:\n{request_text}"
SPOKEN_TEMPLATE = "Heard [{spoken_text}]\nCurrent [{request_text}]"
DEFAULT_GATE = object()


class PlaybackGate(TurnTakingGate):
    async def should_take_turn(self, user_text, assistant_spoken_text, **kwargs):
        return TurnTakingDecision(True, None, "fixture")


class CapturingLLM(LLMServiceDummy):
    def __init__(self, database, *, first_ack=False):
        super().__init__(db_connection_str=str(database), system_prompt=SYSTEM_PROMPT)
        self.calls = asyncio.Queue()
        self.release_first_response = asyncio.Event()
        self.first_ack = first_ack

    async def get_llm_stream_response(
        self, context_id, user_id, messages, system_prompt_params=None, **kwargs,
    ):
        self.calls.put_nowait((messages[0]["content"], messages[-1]["content"], system_prompt_params))
        if messages[-1]["content"] == FIRST_REQUEST:
            if self.first_ack:
                yield LLMResponse(context_id=context_id, text=ACK_TEXT, voice_text=ACK_TEXT)
            await self.release_first_response.wait()
            answer = FIRST_RESPONSE
        else:
            answer = "Sure."
        yield LLMResponse(context_id=context_id, text=answer, voice_text=answer)


class Socket:
    def __init__(self, server):
        self.server = server
        self.connection = WebSocketSessionData()
        self.incoming = None
        self.sent = []
        self.changed = asyncio.Event()

    async def receive_text(self):
        return json.dumps(self.incoming)

    async def send_text(self, text):
        self.sent.append(json.loads(text))
        self.changed.set()

    async def send(self, request_type, **fields):
        self.incoming = {"type": request_type, "session_id": SESSION_ID, **fields}
        await self.server.process_websocket(self, self.connection)

    async def feed(self, pcm):
        await self.send("data", audio_data=base64.b64encode(pcm).decode())

    async def wait_for(self, predicate, after=0):
        while True:
            for message in self.sent[after:]:
                if predicate(message):
                    return message
            self.changed.clear()
            await self.changed.wait()


@asynccontextmanager
async def connected_voice_app(
    monkeypatch, tmp_path, *, wakewords=None, turn_taking_gate=DEFAULT_GATE,
    transcripts=(FIRST_REQUEST, FOLLOW_UP, NORMAL_REQUEST), first_ack=False, **server_options,
):
    transcripts = iter(transcripts)
    if turn_taking_gate is DEFAULT_GATE:
        turn_taking_gate = PlaybackGate()

    async def recognize(session_id, data):
        return SpeechRecognitionResult(text=next(transcripts))

    monkeypatch.setattr(SileroStreamSpeechDetector, "_init_silero_model", lambda *args: None)
    monkeypatch.setattr(SileroStreamSpeechDetector, "_create_vad_iterator", lambda *args: None)
    monkeypatch.setattr(SileroStreamSpeechDetector, "_detect_speech_silero", lambda self, pcm, _: any(pcm))
    recognizer = SimpleNamespace(recognize=recognize)
    vad = SileroStreamSpeechDetector(
        speech_recognizer=recognizer,
        turn_taking_gate=turn_taking_gate,
        silence_duration_threshold=0.1,
        segment_silence_threshold=0.2,
        min_duration=0.1,
        on_recording_started_min_duration=0.15,
    )
    dispatch_tasks = set()

    @vad.on_speech_detected
    async def track_dispatch(*args):
        dispatch_tasks.add(asyncio.current_task())

    llm = CapturingLLM(tmp_path / "context.db", first_ack=first_ack)
    tts = SpeechSynthesizerDummy(synthesized_bytes=b"voice")
    server = AIAvatarWebSocketServer(
        vad=vad,
        stt=recognizer,
        llm=llm,
        tts=tts,
        db_connection_str=str(tmp_path / "pipeline.db"),
        voice_recorder=SimpleNamespace(),
        voice_recorder_enabled=False,
        use_invoke_queue=True,
        merge_request_threshold=0,
        mute_on_barge_in=True,
        include_interruption_context=True,
        wakewords=wakewords,
        wakeword_timeout=0,
        **server_options,
    )

    if turn_taking_gate is not None:
        @server.on_request
        async def forward_playback(request):
            if request.type == "playback":
                assert turn_taking_gate.handle_playback_event(request.session_id, **request.metadata)

    socket = Socket(server)
    shared_prompt_params = {"name": "Aki"}
    try:
        await socket.send("start", user_id=USER_ID)
        vad.set_session_data(SESSION_ID, "system_prompt_params", shared_prompt_params)
        yield socket, llm, shared_prompt_params
    finally:
        await server.sts.finalize(SESSION_ID)
        tasks = dispatch_tasks.union(server.sts._invoke_workers.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.sts.shutdown()
        await tts.close()


def fake_playback_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(playback_session, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(adapter_interruption, "monotonic", lambda: clock.now)
    return clock


async def start_first_response(socket, llm):
    await socket.feed(VOICE)
    await socket.feed(SILENCE)
    start = await socket.wait_for(lambda message: message["type"] == "start")
    prompt, text, params = await llm.calls.get()
    assert prompt == "Reply to Aki."
    assert text == FIRST_REQUEST
    assert params == {"name": "Aki"}
    return start


async def assert_new_start(socket, previous_transaction_id, recognized_text, request_text, *, after=0):
    start = await socket.wait_for(lambda message: (
        message["type"] == "start" and message["transaction_id"] != previous_transaction_id
    ), after=after)
    assert start["metadata"]["recognized_text"] == recognized_text
    assert start["metadata"]["request_text"] == request_text
    return start


async def trigger_barge_in(socket):
    # The caller has started the recording; this packet crosses the mute threshold.
    prior_messages = len(socket.sent)
    await socket.feed(VOICE)
    await socket.wait_for(lambda message: message["type"] == "stop", after=prior_messages)


async def record_follow_up(socket):
    await socket.feed(VOICE)
    await trigger_barge_in(socket)
    await socket.feed(SILENCE)


async def report_playback_start(socket, transaction_id, playback_id, duration_seconds, *, text=None):
    audio = await socket.wait_for(lambda message: (
        message["transaction_id"] == transaction_id and bool(message.get("audio_data"))
        and (text is None or message["voice_text"] == text)
    ))
    await socket.send("playback", metadata={
        "event": "start",
        "playback_id": playback_id,
        "transaction_id": transaction_id,
        "text": audio["voice_text"],
        "duration_seconds": duration_seconds,
    })


async def report_completed_playback(socket, transaction_id, *, is_final=False):
    playback_id = f"audio-{transaction_id}"
    await report_playback_start(socket, transaction_id, playback_id, duration_seconds=0.1)
    await socket.send("playback", metadata={
        "event": "end", "playback_id": playback_id, "completed": True,
    })
    if is_final:
        final = await socket.wait_for(lambda message: (
            message["type"] == "final" and message["transaction_id"] == transaction_id
        ))
        assert not (final.get("metadata") or {}).get("interrupted")
        await socket.send("playback", metadata={
            "event": "final", "playback_id": playback_id, "transaction_id": transaction_id,
        })


async def interrupt_during_playback(socket, llm, clock):
    llm.release_first_response.set()
    first = await start_first_response(socket, llm)
    transaction_id = first["transaction_id"]
    await socket.wait_for(lambda message: message["type"] == "final")
    await report_playback_start(socket, transaction_id, "first-audio", duration_seconds=8)

    clock.now = 102.0
    await socket.feed(VOICE)
    clock.now = 106.0
    await trigger_barge_in(socket)
    await socket.send("playback", metadata={
        "event": "end", "playback_id": "first-audio", "completed": False,
    })
    await socket.feed(SILENCE)
    return transaction_id


def test_playback_prefix_accumulates_across_chunks_when_previous_end_arrives_late(monkeypatch):
    clock = fake_playback_clock(monkeypatch)
    state = adapter_interruption.InterruptionState()
    transaction_id = "response"
    state.observe(STSResponse(type="start", context_id="conversation", transaction_id=transaction_id))
    for text in ("ABCD", "EFGH"):
        state.observe(STSResponse(type="chunk", transaction_id=transaction_id, voice_text=text, audio_data=b"voice"))

    state.handle_playback_event(
        event="start", playback_id="first", transaction_id=transaction_id, text="ABCD", duration_seconds=4,
    )
    clock.now = 105.0
    state.handle_playback_event(
        event="start", playback_id="second", transaction_id=transaction_id, text="EFGH", duration_seconds=4,
    )
    state.handle_playback_event(event="end", playback_id="first", completed=True)
    clock.now = 107.0

    interruption = state.snapshot(transaction_id)
    assert interruption is not None
    assert interruption.spoken_text == "ABCDEF"


def test_unplayed_ack_is_excluded_when_answer_starts(monkeypatch):
    clock = fake_playback_clock(monkeypatch)
    state = adapter_interruption.InterruptionState()
    state.observe(STSResponse(type="start", context_id="conversation", transaction_id="response"))
    for text in (ACK_TEXT, FIRST_RESPONSE):
        state.observe(STSResponse(type="chunk", transaction_id="response", voice_text=text, audio_data=b"voice"))

    state.handle_playback_event(
        event="start", playback_id="answer", transaction_id="response", text=FIRST_RESPONSE, duration_seconds=8,
    )
    clock.now = 102.0
    interruption = state.snapshot("response")
    assert interruption is not None and interruption.spoken_text == "AB"


@pytest.mark.parametrize("replace_answer", [False, True], ids=["answer-prefix", "answer-replaced"])
def test_ack_playback_from_previous_transaction_is_attributed_to_its_audio(monkeypatch, replace_answer):
    clock = fake_playback_clock(monkeypatch)
    state = adapter_interruption.InterruptionState()
    context_id = "conversation"
    state.observe(STSResponse(type="start", context_id=context_id, transaction_id="earlier-response"))
    state.observe(STSResponse(
        type="chunk", transaction_id="earlier-response", voice_text=ACK_TEXT, audio_data=b"voice",
    ))
    state.handle_playback_event(
        event="start", playback_id="earlier-ack", transaction_id="earlier-response",
        text=ACK_TEXT, duration_seconds=8,
    )
    state.observe(STSResponse(type="start", context_id=context_id, transaction_id="next-response"))
    clock.now = 104.0

    interruption = state.snapshot("next-response")
    assert interruption is not None
    assert interruption.interrupted_response_transaction_id == "earlier-response"
    assert interruption.spoken_text == "そうなん"

    clock.now = 108.0
    state.handle_playback_event(
        event="end", playback_id="earlier-ack", transaction_id="earlier-response", completed=True,
    )
    state.observe(STSResponse(
        type="chunk", transaction_id="next-response", voice_text=FIRST_RESPONSE, audio_data=b"voice",
    ))
    state.handle_playback_event(
        event="start", playback_id="next-audio", transaction_id="next-response",
        text=FIRST_RESPONSE, duration_seconds=8,
    )
    interruption = state.snapshot("next-response")
    assert interruption is not None
    assert interruption.interrupted_response_transaction_id == "earlier-response"
    assert interruption.spoken_text == ACK_TEXT

    clock.now = 110.0
    if replace_answer:
        state.observe(STSResponse(type="start", context_id=context_id, transaction_id="replacement"))
        state.observe(STSResponse(
            type="chunk", transaction_id="replacement", voice_text=FIRST_RESPONSE, audio_data=b"voice",
        ))
        state.handle_playback_event(
            event="start", playback_id="replacement-audio", transaction_id="replacement",
            text=FIRST_RESPONSE, duration_seconds=8,
        )
    interruption = state.snapshot("replacement" if replace_answer else "next-response")
    assert interruption is not None
    assert interruption.interrupted_response_transaction_id == "next-response"
    assert interruption.spoken_text == "AB"


def test_playback_from_another_context_does_not_hide_pending_request(monkeypatch):
    clock = fake_playback_clock(monkeypatch)
    state = adapter_interruption.InterruptionState()
    state.observe(STSResponse(type="start", context_id="earlier-context", transaction_id="earlier-response"))
    state.observe(STSResponse(
        type="chunk", transaction_id="earlier-response", voice_text=ACK_TEXT, audio_data=b"voice",
    ))
    state.handle_playback_event(
        event="start", playback_id="earlier-ack", transaction_id="earlier-response",
        text=ACK_TEXT, duration_seconds=8,
    )
    clock.now = 108.0
    state.handle_playback_event(
        event="end", playback_id="earlier-ack", transaction_id="earlier-response", completed=True,
    )
    state.observe(STSResponse(type="start", context_id="new-context", transaction_id="new-response"))

    interruption = state.snapshot("new-response")
    assert interruption is not None
    assert interruption.interrupted_response_context_id == "new-context"
    assert interruption.interrupted_response_transaction_id == "new-response"
    assert interruption.spoken_text is None


def test_nonfinal_audio_remains_eligible_after_playback_ends(monkeypatch):
    clock = fake_playback_clock(monkeypatch)
    state = adapter_interruption.InterruptionState()
    state.observe(STSResponse(type="start", context_id="conversation", transaction_id="response"))
    state.observe(STSResponse(
        type="chunk", transaction_id="response", voice_text=FIRST_RESPONSE, audio_data=b"voice",
    ))
    state.handle_playback_event(
        event="start", playback_id="audio", transaction_id="response",
        text=FIRST_RESPONSE, duration_seconds=8,
    )
    clock.now = 102.0
    interruption = state.snapshot(None)
    assert interruption is not None and interruption.spoken_text == "AB"

    clock.now = 108.0
    state.handle_playback_event(
        event="end", playback_id="audio", transaction_id="response", completed=True,
    )
    interruption = state.snapshot(None)
    assert interruption is not None and interruption.spoken_text == FIRST_RESPONSE
    assert state.snapshot(None) == interruption


@pytest.mark.parametrize("final_first", [True, False], ids=["final-before-end", "final-after-end"])
def test_final_playback_remains_eligible_only_until_completed(monkeypatch, final_first):
    clock = fake_playback_clock(monkeypatch)
    state = adapter_interruption.InterruptionState()
    state.observe(STSResponse(type="start", context_id="conversation", transaction_id="response"))
    state.observe(STSResponse(
        type="chunk", transaction_id="response", voice_text=FIRST_RESPONSE, audio_data=b"voice",
    ))
    state.handle_playback_event(
        event="start", playback_id="audio", transaction_id="response", text=FIRST_RESPONSE, duration_seconds=8,
    )
    state.observe(STSResponse(type="final", transaction_id="response"))
    final = {"event": "final", "playback_id": "audio", "transaction_id": "response"}
    if final_first:
        state.handle_playback_event(**final)
    clock.now = 102.0
    interruption = state.snapshot(None)
    assert interruption is not None and interruption.spoken_text == "AB"

    clock.now = 108.0
    state.handle_playback_event(event="end", playback_id="audio", completed=True)
    if not final_first:
        interruption = state.snapshot(None)
        assert interruption is not None and interruption.spoken_text == FIRST_RESPONSE
        state.handle_playback_event(**final)
    assert state.snapshot(None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("with_gate", [True, False], ids=["with-gate", "without-gate"])
async def test_spoken_prefix_is_captured_at_recording_onset_after_sts_final(monkeypatch, tmp_path, with_gate):
    clock = fake_playback_clock(monkeypatch)
    gate = PlaybackGate() if with_gate else None

    async with connected_voice_app(
        monkeypatch, tmp_path, turn_taking_gate=gate,
    ) as (socket, llm, shared_params):
        before_llm = []

        @socket.server.sts.on_before_llm(channels=socket.server.channel)
        async def observe_request(request):
            before_llm.append(request.text)

        async with asyncio.timeout(3):
            transaction_id = await interrupt_during_playback(socket, llm, clock)
            if gate is not None:
                playback = gate.get_session(SESSION_ID).estimate_playback()
                assert playback.assistant_spoken_text == "ABCDEF"

            prompt, text, params = await llm.calls.get()
            assert prompt == "Reply to Aki."
            assert params == {"name": "Aki"}
            assert text.startswith("$The user started speaking during the previous response.")
            assert text.endswith(
                "\n\nAssistant audio played so far (estimated):\nAB\n\n"
                f"Current user message:\n{FOLLOW_UP}"
            )
            assert "CDEFGH" not in text
            assert "Treat the following quotation" not in text
            assert shared_params == {"name": "Aki"}
            assert before_llm == [FIRST_REQUEST, text]
            follow_up = await assert_new_start(socket, transaction_id, FOLLOW_UP, text)
            follow_up_id = follow_up["transaction_id"]

            await report_completed_playback(socket, follow_up_id, is_final=True)
            prior_messages = len(socket.sent)
            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            assert await llm.calls.get() == ("Reply to Aki.", NORMAL_REQUEST, {"name": "Aki"})
            await assert_new_start(socket, follow_up_id, NORMAL_REQUEST, NORMAL_REQUEST, after=prior_messages)
            assert before_llm == [FIRST_REQUEST, text, NORMAL_REQUEST]


@pytest.mark.asyncio
@pytest.mark.parametrize(("phase", "heard", "heard_next"), [
    ("during-ack", "そう", ACK_TEXT),
    ("after-ack", ACK_TEXT, ACK_TEXT),
    ("answer-just-started", ACK_TEXT, ACK_TEXT + "ABCD"),
    ("during-answer", ACK_TEXT + "AB", ACK_TEXT + "ABCDEF"),
], ids=["during-ack", "after-ack", "answer-just-started", "during-answer"])
async def test_ack_and_answer_use_same_spoken_template_at_onset(monkeypatch, tmp_path, phase, heard, heard_next):
    clock = fake_playback_clock(monkeypatch)

    async with connected_voice_app(
        monkeypatch, tmp_path, turn_taking_gate=None, first_ack=True, interruption_spoken_template=SPOKEN_TEMPLATE,
    ) as (socket, llm, _):
        async with asyncio.timeout(3):
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            await report_playback_start(socket, first_id, "first-ack", duration_seconds=8, text=ACK_TEXT)

            if phase != "during-ack":
                clock.now = 108.0
                await socket.send("playback", metadata={
                    "event": "end", "transaction_id": first_id,
                    "playback_id": "first-ack", "completed": True,
                })
            answer_started = phase in ("answer-just-started", "during-answer")
            if answer_started:
                llm.release_first_response.set()
                await socket.wait_for(lambda message: message["type"] == "final")
                await report_playback_start(socket, first_id, "first-audio", duration_seconds=8, text=FIRST_RESPONSE)
                await socket.send("playback", metadata={
                    "event": "final", "playback_id": "first-audio", "transaction_id": first_id,
                })

            if phase != "answer-just-started":
                clock.now += 2.0
            await socket.feed(VOICE)
            clock.now += 4.0
            await trigger_barge_in(socket)
            if answer_started:
                await socket.send("playback", metadata={
                    "event": "end", "playback_id": "first-audio", "completed": False,
                })
            elif phase == "during-ack":
                await socket.send("playback", metadata={
                    "event": "end", "transaction_id": first_id,
                    "playback_id": "first-ack", "completed": True,
                })
            await socket.feed(SILENCE)
            llm.release_first_response.set()

            expected = SPOKEN_TEMPLATE.format(spoken_text=heard, request_text=FOLLOW_UP)
            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            follow_up = await assert_new_start(socket, first_id, FOLLOW_UP, expected)

            await socket.wait_for(lambda message: (
                message["type"] == "final" and message["transaction_id"] == follow_up["transaction_id"]
            ))
            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            # The follow-up response has not played, so the unfinished audio still supplies context.
            next_request = SPOKEN_TEMPLATE.format(spoken_text=heard_next, request_text=NORMAL_REQUEST)
            assert await llm.calls.get() == ("Reply to Aki.", next_request, {"name": "Aki"})


@pytest.mark.asyncio
async def test_discarded_answer_is_reported_once_after_later_playback(monkeypatch, tmp_path):
    clock = fake_playback_clock(monkeypatch)

    async with connected_voice_app(
        monkeypatch, tmp_path, turn_taking_gate=None, first_ack=True,
        transcripts=(FIRST_REQUEST, FOLLOW_UP, ADDITIONAL_REQUEST, NORMAL_REQUEST),
        interruption_spoken_template=SPOKEN_TEMPLATE,
    ) as (socket, llm, _):
        async with asyncio.timeout(3):
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            await report_playback_start(socket, first_id, "first-ack", duration_seconds=8, text=ACK_TEXT)
            clock.now = 108.0
            await socket.send("playback", metadata={
                "event": "end", "transaction_id": first_id,
                "playback_id": "first-ack", "completed": True,
            })
            llm.release_first_response.set()
            await socket.wait_for(lambda message: (
                message["transaction_id"] == first_id and message["voice_text"] == FIRST_RESPONSE
                and bool(message.get("audio_data"))
            ))
            await socket.wait_for(lambda message: message["type"] == "final")

            for transaction_id, text in (
                ("unknown-transaction", FIRST_RESPONSE), (first_id, "Unknown audio"), (first_id, ACK_TEXT),
            ):
                await socket.send("playback", metadata={
                    "event": "discarded", "transaction_id": transaction_id, "text": text,
                })

            await record_follow_up(socket)
            expected = SPOKEN_TEMPLATE.format(spoken_text=ACK_TEXT, request_text=FOLLOW_UP)
            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            follow_up = await assert_new_start(socket, first_id, FOLLOW_UP, expected)
            follow_up_id = follow_up["transaction_id"]

            # The browser confirms the older answer was discarded only after accepting the follow-up.
            discarded_event = {"event": "discarded", "transaction_id": first_id, "text": FIRST_RESPONSE}
            await socket.send("playback", metadata=discarded_event)
            await socket.send("playback", metadata=discarded_event)

            await report_completed_playback(socket, follow_up_id, is_final=True)
            prior_messages = len(socket.sent)
            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            expected = (
                "$The following assistant speech was canceled before playback started.\n"
                f"{FIRST_RESPONSE}\n\n{ADDITIONAL_REQUEST}"
            )
            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            next_turn = await assert_new_start(
                socket, follow_up_id, ADDITIONAL_REQUEST, ADDITIONAL_REQUEST, after=prior_messages,
            )

            await report_completed_playback(socket, next_turn["transaction_id"], is_final=True)
            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            assert await llm.calls.get() == ("Reply to Aki.", NORMAL_REQUEST, {"name": "Aki"})
            messages = await llm.context_manager.get_histories(first["context_id"])
            assert messages.count({"role": "user", "content": expected}) == 1
            assert {"role": "assistant", "content": ACK_TEXT + FIRST_RESPONSE} in messages


@pytest.mark.asyncio
@pytest.mark.parametrize("same_connection", [True, False], ids=["new-context", "new-connection"])
async def test_discarded_audio_does_not_cross_context_or_connection(monkeypatch, tmp_path, same_connection):
    async with connected_voice_app(monkeypatch, tmp_path, turn_taking_gate=None) as (socket, llm, _):
        async with asyncio.timeout(3):
            llm.release_first_response.set()
            first = await start_first_response(socket, llm)
            await socket.wait_for(lambda message: message["type"] == "final")
            await socket.send("playback", metadata={
                "event": "discarded", "transaction_id": first["transaction_id"], "text": FIRST_RESPONSE,
            })

            session_id = SESSION_ID if same_connection else "other-connection"
            context_id = "other-context" if same_connection else first["context_id"]
            other = socket if same_connection else Socket(socket.server)
            try:
                if same_connection:
                    await llm.context_manager.add_histories(context_id, [
                        {"role": "user", "content": "A separate conversation."},
                    ])
                else:
                    await other.send("start", session_id=session_id, user_id=USER_ID, context_id=context_id)
                # The recognition snapshot and response tracking are both real for this connection.
                vad = socket.server.sts.vad
                vad.set_session_data(session_id, "system_prompt_params", {"name": "Aki"})
                vad.set_session_data(session_id, "context_id", context_id)
                for pcm in (VOICE, SILENCE):
                    await other.send("data", session_id=session_id, audio_data=base64.b64encode(pcm).decode())
                assert await llm.calls.get() == ("Reply to Aki.", FOLLOW_UP, {"name": "Aki"})
                follow_up = await assert_new_start(other, first["transaction_id"], FOLLOW_UP, FOLLOW_UP)
                assert follow_up["context_id"] == context_id
                await other.wait_for(lambda message: (
                    message["type"] == "final" and message["transaction_id"] == follow_up["transaction_id"]
                ))
            finally:
                if not same_connection:
                    await socket.server.sts.finalize(session_id)


@pytest.mark.asyncio
async def test_unverified_playback_preserves_before_audio_context(monkeypatch, tmp_path):
    clock = fake_playback_clock(monkeypatch)

    async with connected_voice_app(
        monkeypatch, tmp_path, turn_taking_gate=None, interruption_before_audio_template=BEFORE_AUDIO_TEMPLATE,
    ) as (socket, llm, _):
        async with asyncio.timeout(3):
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            await socket.send("playback", metadata={
                "event": "start", "playback_id": "unknown", "transaction_id": first_id,
                "text": "Unverified speech", "duration_seconds": 8,
            })

            clock.now = 104.0
            await record_follow_up(socket)
            llm.release_first_response.set()

            expected = BEFORE_AUDIO_TEMPLATE.format(previous_request=FIRST_REQUEST, request_text=FOLLOW_UP)
            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            await assert_new_start(socket, first_id, FOLLOW_UP, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "with_gate"),
    [("completed", True), ("completed", False), ("stopped", True), ("final", True), ("muted", True)],
    ids=["completed", "completed-without-gate", "stopped", "final", "muted"],
)
async def test_gap_after_audio_retains_spoken_prefix(monkeypatch, tmp_path, boundary, with_gate):
    clock = fake_playback_clock(monkeypatch)

    async with connected_voice_app(
        monkeypatch, tmp_path, interruption_spoken_template=SPOKEN_TEMPLATE,
        turn_taking_gate=PlaybackGate() if with_gate else None,
    ) as (socket, llm, _):
        release_first_final = asyncio.Event()

        @socket.server.sts.on_finish(channels=socket.server.channel)
        async def hold_first_final(request, _response):
            if request.text == FIRST_REQUEST:
                await release_first_final.wait()

        async with asyncio.timeout(3):
            llm.release_first_response.set()
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            playback_id = "first-audio"
            await report_playback_start(socket, first_id, playback_id, duration_seconds=8)
            if boundary == "final":
                await socket.send("playback", metadata={
                    "event": "final", "playback_id": playback_id, "transaction_id": first_id,
                })
            if boundary == "muted":
                socket.connection.muted_transaction_id = first_id
            clock.now = 104.0 if boundary == "stopped" else 108.0
            await socket.send("playback", metadata={
                "event": "end", "playback_id": playback_id, "completed": boundary != "stopped",
            })

            # No later chunk starts. Only the client can confirm that playback finished.
            await record_follow_up(socket)
            release_first_final.set()

            if boundary == "final":
                expected = FOLLOW_UP
            else:
                heard = "ABCD" if boundary == "stopped" else FIRST_RESPONSE
                expected = SPOKEN_TEMPLATE.format(spoken_text=heard, request_text=FOLLOW_UP)
            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            await assert_new_start(socket, first_id, FOLLOW_UP, expected)


@pytest.mark.asyncio
async def test_spoken_prefix_cannot_wake_pipeline(monkeypatch, tmp_path):
    clock = fake_playback_clock(monkeypatch)

    # "Kyoto" wakes the first request, and "AB" is in the captured playback only.
    async with connected_voice_app(monkeypatch, tmp_path, wakewords=["Kyoto", "AB"]) as (socket, llm, _):
        async with asyncio.timeout(3):
            first_id = await interrupt_during_playback(socket, llm, clock)
            follow_up = await assert_new_start(socket, first_id, FOLLOW_UP, None)
            await socket.wait_for(lambda message: (
                message["type"] == "final" and message["transaction_id"] == follow_up["transaction_id"]
            ))
            assert llm.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_sent", [False, True], ids=["not-yet-sent", "sent-without-playback"])
async def test_before_audio_context_requires_evidence_audio_was_not_sent(monkeypatch, tmp_path, audio_sent):
    async with connected_voice_app(monkeypatch, tmp_path) as (socket, llm, shared_params):
        async with asyncio.timeout(3):
            if audio_sent:
                llm.release_first_response.set()
            first = await start_first_response(socket, llm)
            if audio_sent:
                await socket.wait_for(lambda message: message["type"] == "final")
                await socket.wait_for(lambda message: bool(message.get("audio_data")))
            else:
                assert not any(message.get("audio_data") for message in socket.sent)

            await record_follow_up(socket)
            llm.release_first_response.set()

            prompt, text, params = await llm.calls.get()
            assert prompt == "Reply to Aki."
            assert params == {"name": "Aki"}
            assert shared_params == {"name": "Aki"}
            if audio_sent:
                assert text == FOLLOW_UP
            else:
                assert text.startswith("$The user spoke again before audio for the previous request began.")
                assert text.endswith(
                    f"\n\nPrevious user request:\n{FIRST_REQUEST}\n\n"
                    f"Current user message:\n{FOLLOW_UP}"
                )
                assert "Treat the following quotation" not in text
            await assert_new_start(socket, first["transaction_id"], FOLLOW_UP, text)


@pytest.mark.asyncio
async def test_rejected_barge_in_does_not_leave_stale_interruption_context(monkeypatch, tmp_path):
    rejected = asyncio.Event()
    gate = PlaybackGate(bypass_enabled=False)

    async def decide(user_text, assistant_spoken_text, **kwargs):
        if user_text == FOLLOW_UP:
            rejected.set()
            return TurnTakingDecision(False, None, "fixture")
        return TurnTakingDecision(True, None, "fixture")

    monkeypatch.setattr(gate, "should_take_turn", decide)
    async with connected_voice_app(monkeypatch, tmp_path, turn_taking_gate=gate) as (socket, llm, _):
        async with asyncio.timeout(3):
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            await record_follow_up(socket)
            await rejected.wait()
            llm.release_first_response.set()

            final = await socket.wait_for(lambda message: (
                message["type"] == "final" and message["transaction_id"] == first_id
            ))
            assert final["metadata"]["interrupted"] is True
            assert not any(message.get("audio_data") for message in socket.sent)
            assert llm.calls.empty()

            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            assert await llm.calls.get() == ("Reply to Aki.", NORMAL_REQUEST, {"name": "Aki"})
            await assert_new_start(socket, first_id, NORMAL_REQUEST, NORMAL_REQUEST)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_heard", [False, True], ids=["first-unheard", "first-heard"])
async def test_interruption_excludes_speech_from_replaced_queue_entry(monkeypatch, tmp_path, first_heard):
    async with connected_voice_app(
        monkeypatch,
        tmp_path,
        transcripts=(FIRST_REQUEST, FOLLOW_UP, ADDITIONAL_REQUEST, NORMAL_REQUEST),
        interruption_before_audio_template=BEFORE_AUDIO_TEMPLATE,
        interruption_spoken_template=SPOKEN_TEMPLATE,
    ) as (socket, llm, _):
        pipeline = socket.server.sts
        release_first_final = asyncio.Event()

        @pipeline.on_finish(channels=socket.server.channel)
        async def hold_first_final(request, _response):
            if request.text == FIRST_REQUEST:
                await release_first_final.wait()

        async with asyncio.timeout(3):
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            request_queue = pipeline._request_queues[SESSION_ID]
            enqueue = request_queue.put
            enqueued = asyncio.Queue()

            async def observe_enqueue(item):
                await enqueue(item)
                enqueued.put_nowait(item[1])

            monkeypatch.setattr(request_queue, "put", observe_enqueue)

            # The gate admits this shorter turn even though it does not mute A.
            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            second = await enqueued.get()
            assert second.text == FOLLOW_UP
            assert request_queue.qsize() == 1

            if first_heard:
                llm.release_first_response.set()
                await report_completed_playback(socket, first_id)
                assert request_queue.qsize() == 1

            await record_follow_up(socket)
            third = await enqueued.get()
            assert third.text == ADDITIONAL_REQUEST
            assert request_queue.qsize() == 1

            llm.release_first_response.set()
            release_first_final.set()

            if first_heard:
                expected = SPOKEN_TEMPLATE.format(spoken_text=FIRST_RESPONSE, request_text=ADDITIONAL_REQUEST)
            else:
                expected = BEFORE_AUDIO_TEMPLATE.format(previous_request=FIRST_REQUEST, request_text=ADDITIONAL_REQUEST)
            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            start = await assert_new_start(socket, first_id, ADDITIONAL_REQUEST, expected)
            assert start["transaction_id"] == third.transaction_id
            assert not any(
                message["type"] == "start" and message["transaction_id"] == second.transaction_id
                for message in socket.sent
            )

            await report_completed_playback(socket, third.transaction_id, is_final=True)
            prior_messages = len(socket.sent)
            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            assert await llm.calls.get() == ("Reply to Aki.", NORMAL_REQUEST, {"name": "Aki"})
            await assert_new_start(socket, third.transaction_id, NORMAL_REQUEST, NORMAL_REQUEST, after=prior_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_heard", [False, True], ids=["first-unheard", "first-heard-after-onset"])
async def test_interruption_combines_processed_requests_before_audio(monkeypatch, tmp_path, first_heard):
    async with connected_voice_app(
        monkeypatch,
        tmp_path,
        transcripts=(FIRST_REQUEST, FOLLOW_UP, ADDITIONAL_REQUEST),
        interruption_before_audio_template=BEFORE_AUDIO_TEMPLATE,
        interruption_spoken_template=SPOKEN_TEMPLATE,
    ) as (socket, llm, _):
        release_second_response = asyncio.Event()

        @socket.server.sts.on_before_llm(channels=socket.server.channel)
        async def pause_second_response(request):
            if request.text.endswith(FOLLOW_UP):
                await release_second_response.wait()

        async with asyncio.timeout(3):
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            await socket.feed(VOICE)
            if first_heard:
                # A sends its audio after B opens, while B is too short to mute it.
                llm.release_first_response.set()
                await report_completed_playback(socket, first_id, is_final=True)
            else:
                await trigger_barge_in(socket)
            await socket.feed(SILENCE)
            llm.release_first_response.set()

            second_text = BEFORE_AUDIO_TEMPLATE.format(previous_request=FIRST_REQUEST, request_text=FOLLOW_UP)
            second = await assert_new_start(socket, first_id, FOLLOW_UP, second_text)
            prior_messages = len(socket.sent)
            await record_follow_up(socket)
            release_second_response.set()
            assert await llm.calls.get() == ("Reply to Aki.", second_text, {"name": "Aki"})

            previous = FOLLOW_UP if first_heard else f"{FIRST_REQUEST}\n{FOLLOW_UP}"
            expected = BEFORE_AUDIO_TEMPLATE.format(previous_request=previous, request_text=ADDITIONAL_REQUEST)
            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            await assert_new_start(socket, second["transaction_id"], ADDITIONAL_REQUEST, expected, after=prior_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("during_playback", [False, True], ids=["before-audio", "during-playback"])
async def test_custom_interruption_templates(monkeypatch, tmp_path, during_playback):
    clock = fake_playback_clock(monkeypatch)

    async with connected_voice_app(
        monkeypatch,
        tmp_path,
        interruption_spoken_template="Heard [{spoken_text}]\nCurrent [{request_text}]",
        interruption_before_audio_template="Pending [{previous_request}]\nCurrent [{request_text}]",
    ) as (socket, llm, _):
        async with asyncio.timeout(3):
            if during_playback:
                first_id = await interrupt_during_playback(socket, llm, clock)
                expected = f"Heard [AB]\nCurrent [{FOLLOW_UP}]"
            else:
                first = await start_first_response(socket, llm)
                first_id = first["transaction_id"]
                await record_follow_up(socket)
                llm.release_first_response.set()
                expected = f"Pending [{FIRST_REQUEST}]\nCurrent [{FOLLOW_UP}]"

            assert await llm.calls.get() == ("Reply to Aki.", expected, {"name": "Aki"})
            await assert_new_start(socket, first_id, FOLLOW_UP, expected)


@pytest.mark.asyncio
async def test_interruption_requires_the_final_request_context(monkeypatch, tmp_path):
    async with connected_voice_app(monkeypatch, tmp_path) as (socket, llm, _):
        new_context_id = "another-context"
        seen_interruptions = []

        @socket.server.sts.on_accepted(channels=socket.server.channel)
        async def change_context(request):
            if request.text == FOLLOW_UP:
                seen_interruptions.append(request.request_context.interruption)
                request.context_id = new_context_id

        async with asyncio.timeout(3):
            await llm.context_manager.add_histories(new_context_id, [
                {"role": "user", "content": "A separate conversation."},
            ])
            first = await start_first_response(socket, llm)
            await record_follow_up(socket)
            llm.release_first_response.set()

            assert await llm.calls.get() == ("Reply to Aki.", FOLLOW_UP, {"name": "Aki"})
            follow_up = await assert_new_start(socket, first["transaction_id"], FOLLOW_UP, FOLLOW_UP)
            assert follow_up["context_id"] == new_context_id
            assert follow_up["user_id"] == USER_ID
            interruption, = seen_interruptions
            assert interruption.interrupted_response_context_id == first["context_id"]
            assert interruption.interrupted_response_transaction_id == first["transaction_id"]


@pytest.mark.asyncio
async def test_queued_spoken_context_is_ignored_after_session_finalization(monkeypatch, tmp_path):
    clock = fake_playback_clock(monkeypatch)

    async with connected_voice_app(monkeypatch, tmp_path) as (socket, llm, _):
        pipeline = socket.server.sts
        release_first_final = asyncio.Event()

        @pipeline.on_finish(channels=socket.server.channel)
        async def hold_first_final(request, _response):
            if request.text == FIRST_REQUEST:
                await release_first_final.wait()

        async with asyncio.timeout(3):
            llm.release_first_response.set()
            first = await start_first_response(socket, llm)
            first_id = first["transaction_id"]
            await report_playback_start(socket, first_id, "first-audio", duration_seconds=8)
            clock.now = 102.0

            request_queue = pipeline._request_queues[SESSION_ID]
            enqueue = request_queue.put
            enqueued = asyncio.Queue()

            async def observe_enqueue(item):
                await enqueue(item)
                enqueued.put_nowait(item[1])

            monkeypatch.setattr(request_queue, "put", observe_enqueue)
            await record_follow_up(socket)
            queued_request = await enqueued.get()
            assert queued_request.request_context.interruption.spoken_text == "AB"
            assert request_queue.qsize() == 1

            await pipeline.finalize(SESSION_ID)
            release_first_final.set()

            assert await llm.calls.get() == ("Reply to Aki.", FOLLOW_UP, {"name": "Aki"})
            follow_up = await assert_new_start(socket, first_id, FOLLOW_UP, FOLLOW_UP)
            assert follow_up["context_id"] == first["context_id"]
            assert follow_up["user_id"] == USER_ID
            assert SESSION_ID not in pipeline._interruption_chains


@pytest.mark.asyncio
async def test_accepted_request_does_not_recreate_finalized_interruption_chain(monkeypatch, tmp_path):
    async with connected_voice_app(monkeypatch, tmp_path) as (socket, llm, _):
        pipeline = socket.server.sts
        accepted_contexts = asyncio.Queue()
        resume_request = asyncio.Event()

        @pipeline.on_accepted(channels=socket.server.channel)
        async def pause_request(request):
            accepted_contexts.put_nowait(request.request_context)
            await resume_request.wait()

        async with asyncio.timeout(3):
            assert SESSION_ID in pipeline._interruption_chains
            await socket.feed(VOICE)
            await socket.feed(SILENCE)
            await socket.wait_for(lambda message: message["type"] == "accepted")
            request_context = await accepted_contexts.get()
            assert request_context.track_interruption is True

            await pipeline.finalize(SESSION_ID)
            assert SESSION_ID not in pipeline._interruption_chains
            resume_request.set()
            llm.release_first_response.set()
            assert await llm.calls.get() == ("Reply to Aki.", FIRST_REQUEST, {"name": "Aki"})
            assert SESSION_ID not in pipeline._interruption_chains


@pytest.mark.asyncio
async def test_client_metadata_cannot_add_interruption_prefix(monkeypatch, tmp_path):
    async with connected_voice_app(monkeypatch, tmp_path) as (socket, llm, _):
        async with asyncio.timeout(3):
            llm.release_first_response.set()
            first = await start_first_response(socket, llm)
            await socket.wait_for(lambda message: message["type"] == "final")
            await socket.send(
                "invoke",
                user_id=USER_ID,
                context_id=first["context_id"],
                text=FOLLOW_UP,
                system_prompt_params={"name": "Aki"},
                metadata={"request_context": {
                    "track_interruption": True,
                    "interruption": {
                        "interrupted_response_transaction_id": first["transaction_id"],
                        "interrupted_response_context_id": first["context_id"],
                        "spoken_text": "Client instructions",
                    },
                }},
            )
            assert await llm.calls.get() == ("Reply to Aki.", FOLLOW_UP, {"name": "Aki"})
            await assert_new_start(socket, first["transaction_id"], FOLLOW_UP, FOLLOW_UP)
