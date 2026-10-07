"""Hermetic VAD/turn-taking wiring tests; no model, STT, API, or database use.

Silero model creation is replaced before detector construction. The pipeline
test exercises only the real callback that builds requests, with a memory-only
invoke implementation and no pipeline constructor or provider resources.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import inspect
from types import SimpleNamespace

import pytest

from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.vad.base import RecordingSessionBase
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.turn_taking_gates import TurnTakingDecision, TurnTakingGate, TurnTakingGateManager
from aiavatar.sts.vad.turn_taking_gates.bypass import TurnTakingBypassGate
from aiavatar.sts.vad.turn_taking_gates import session as turn_taking_session


class FixedGate(TurnTakingGate):
    def __init__(self, outcome=False, **kwargs):
        super().__init__(**kwargs)
        self.outcome = outcome
        self.calls = []
        self.full_texts = []

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
        self.calls.append((user_text, assistant_spoken_text, session_id))
        self.full_texts.append(assistant_full_text)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return TurnTakingDecision(self.outcome, None, "test_gate")


class BlockingGate(FixedGate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release_cleanup = asyncio.Event()

    async def should_take_turn(self, user_text, assistant_spoken_text, *, session_id=None, assistant_full_text=None, **kwargs):
        self.calls.append((user_text, assistant_spoken_text, session_id))
        self.full_texts.append(assistant_full_text)
        if user_text != "old input":
            return TurnTakingDecision(True, None, "test_gate")
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release_cleanup.wait()
            # A provider that swallows cancellation still must not dispatch.
            return TurnTakingDecision(True, None, "late_provider_result")


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(turn_taking_session, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


@pytest.fixture
def make_silero(monkeypatch):
    monkeypatch.setattr(SileroSpeechDetector, "_init_silero_model", lambda *args: None)
    monkeypatch.setattr(SileroSpeechDetector, "_create_vad_iterator", lambda *args, **kwargs: None)

    def create(detector_class, gate):
        kwargs = {"turn_taking_gate": gate, "on_recording_started_min_duration": 1.5}
        if detector_class is SileroStreamSpeechDetector:
            async def unexpected_recognition(*args, **kwargs):
                raise AssertionError("These tests must not invoke speech recognition")

            kwargs["speech_recognizer"] = SimpleNamespace(recognize=unexpected_recognition)
        return detector_class(**kwargs)

    return create


def begin_playback(vad, clock, session_id="session"):
    assert vad._init_turn_taking_session(session_id) is None
    state = vad.turn_taking_gate.get_session(session_id)
    assert state.start_playback("playback", "abcdefghij", 10.0, transaction_id="transaction")
    clock.now += 5.0
    return state


def collect_speech(vad):
    seen = []

    @vad.on_speech_detected
    async def collect(audio, text, metadata, recorded_duration, session_id):
        seen.append((audio, text, metadata, recorded_duration, session_id))

    return seen


async def dispatch(vad, text="yes", duration=0.3, session_id="session", metadata=None):
    await vad._execute_on_speech_detected(b"audio", text, metadata or {}, duration, session_id)


@pytest.mark.asyncio
async def test_unconfigured_gate_preserves_callback_payload():
    vad = SpeechDetectorDummy()
    seen = collect_speech(vad)
    metadata = {"recording_id": "recording"}
    assert vad._init_turn_taking_session("session") is None
    await dispatch(vad, metadata=metadata)
    assert seen == [(b"audio", "yes", metadata, 0.3, "session")]


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [False, True])
async def test_gate_controls_all_speech_detected_handlers(clock, allowed):
    gate = FixedGate(allowed)
    vad = SpeechDetectorDummy(turn_taking_gate=gate)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    later = collect_speech(vad)
    try:
        await dispatch(vad)
        assert gate.calls == [("yes", "abcde", "session")]
        assert gate.full_texts == ["abcdefghij"]
        assert len(seen) == len(later) == int(allowed)
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_accepted_speech_reaches_later_handlers_after_session_finalization(clock):
    gate = FixedGate(True)
    vad = SpeechDetectorDummy(turn_taking_gate=gate)
    state = begin_playback(vad, clock)
    seen = []

    @vad.on_speech_detected
    async def first(*args):
        seen.append(("first", args))
        await vad.finalize_session(args[-1])

    @vad.on_speech_detected
    async def second(*args):
        seen.append(("second", args))

    metadata = {"recording_id": "recording"}
    args = (b"audio", "question", metadata, 0.3, "session")
    try:
        await vad._execute_on_speech_detected(*args)
        assert gate.calls == [("question", "abcde", "session")]
        assert state.is_closed
        assert seen == [("first", args), ("second", args)]
        assert seen[1][1][2] is metadata
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [TimeoutError("timeout"), ValueError("failed gate")])
async def test_gate_failure_allows_the_input(clock, outcome):
    gate = FixedGate(outcome)
    vad = SpeechDetectorDummy(turn_taking_gate=gate)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    try:
        await dispatch(vad)
        assert len(seen) == 1
        assert len(gate.calls) == 1
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_gate_cancellation_does_not_become_an_allowed_turn(clock):
    gate = FixedGate(asyncio.CancelledError())
    vad = SpeechDetectorDummy(turn_taking_gate=gate)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    try:
        with pytest.raises(asyncio.CancelledError):
            await dispatch(vad)
        assert seen == []
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "", "  "])
async def test_missing_recognition_text_passes_to_downstream_stt(clock, text):
    gate = FixedGate(False)
    bypass = TurnTakingBypassGate()
    manager = TurnTakingGateManager([bypass, gate])
    vad = SpeechDetectorDummy(turn_taking_gate=manager)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    try:
        await dispatch(vad, text=text)
        assert len(seen) == 1
        assert seen[0][1] == text
        assert gate.calls == []
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
@pytest.mark.parametrize("configuration", ["standalone", "automatic_manager", "explicit_bypass"])
async def test_default_skip_condition_uses_the_live_detector_duration(clock, configuration):
    gate = FixedGate(False)
    if configuration == "explicit_bypass":
        bypass = TurnTakingBypassGate()
        configured_gate = TurnTakingGateManager([bypass, gate])
    elif configuration == "automatic_manager":
        bypass = configured_gate = TurnTakingGateManager([gate])
    else:
        bypass = configured_gate = gate
    vad = SpeechDetectorDummy(turn_taking_gate=configured_gate, on_recording_started_min_duration=1.5)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    try:
        await dispatch(vad, text="short", duration=1.49)
        await dispatch(vad, text="at boundary", duration=1.5)
        vad.on_recording_started_min_duration = 2.0
        await dispatch(vad, text="now short", duration=1.5)
        await dispatch(vad, text="new boundary", duration=2.0)
        assert [item[1] for item in seen] == ["at boundary", "new boundary"]
        assert [item[0] for item in gate.calls] == ["short", "now short"]
        assert bypass.skip_condition is None
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_recording_started_notification_alone_does_not_force_a_turn(clock):
    gate = FixedGate(False)
    vad = SpeechDetectorDummy(turn_taking_gate=gate, on_recording_started_min_duration=1.5)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    notified = asyncio.Event()

    @vad.on_recording_started
    async def recording_started(session_id):
        notified.set()

    @vad.should_trigger_recording_started
    def custom_recording_started(text, session):
        return text == "short acknowledgment"

    recording = RecordingSessionBase("session")
    try:
        # The explicit callback overrides the duration policy in both directions.
        recording.record_duration = 3.0
        await vad._check_and_trigger_recording_started(recording, "not now")
        assert not recording.on_recording_started_triggered
        assert not notified.is_set()
        recording.record_duration = 0.2
        await vad._check_and_trigger_recording_started(recording, "short acknowledgment")
        await asyncio.wait_for(notified.wait(), 1.0)
        await dispatch(vad, text="short acknowledgment", duration=0.2)
        assert seen == []
        assert len(gate.calls) == 1
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_skip_condition", [False, True], ids=["default_skip", "custom_skip"])
async def test_configured_gate_notifies_recording_start_only_by_voiced_duration(custom_skip_condition):
    gate = FixedGate(False)
    bypass = TurnTakingBypassGate(
        skip_condition=(lambda text, duration: bool(text)) if custom_skip_condition else None,
    )
    manager = TurnTakingGateManager([bypass, gate])
    vad = SpeechDetectorDummy(turn_taking_gate=manager, on_recording_started_min_duration=1.5)
    recording = RecordingSessionBase("session")
    recording.last_recognized_text = "many recognized characters"
    notified = asyncio.Event()
    seen = []

    @vad.on_recording_started
    async def recording_started(session_id):
        seen.append(session_id)
        notified.set()

    try:
        # Long text and trailing silence must not make a short utterance interrupt.
        for duration, silence in [(0.2, 0.0), (1.49, 0.0), (2.0, 0.51)]:
            recording.record_duration = duration
            recording.silence_duration = silence
            await vad._check_and_trigger_recording_started(recording)
            assert not recording.on_recording_started_triggered
        await asyncio.sleep(0)
        assert seen == []

        recording.record_duration = 2.0
        recording.silence_duration = 0.5
        await vad._check_and_trigger_recording_started(recording)
        await asyncio.wait_for(notified.wait(), 1.0)
        assert recording.on_recording_started_triggered
        recording.record_duration = 2.1
        await vad._check_and_trigger_recording_started(recording)
        await asyncio.sleep(0)
        assert seen == ["session"]
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_unconfigured_gate_retains_text_or_duration_recording_start():
    vad = SpeechDetectorDummy(on_recording_started_min_duration=1.5)
    seen = []

    @vad.on_recording_started
    async def recording_started(session_id):
        seen.append(session_id)

    text_recording = RecordingSessionBase("text")
    text_recording.record_duration = 0.2
    await vad._check_and_trigger_recording_started(text_recording, "yes")
    duration_recording = RecordingSessionBase("duration")
    duration_recording.record_duration = 1.5
    await vad._check_and_trigger_recording_started(duration_recording, "")
    await asyncio.sleep(0)
    assert seen == ["text", "duration"]
    assert text_recording.on_recording_started_triggered
    assert duration_recording.on_recording_started_triggered


@pytest.mark.asyncio
async def test_speech_end_metadata_corrects_the_prefix_and_survives_dispatch(clock, monkeypatch):
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(turn_taking_session, "datetime", FixedDateTime)
    speech_end_at = FixedDateTime.fromtimestamp((now - timedelta(seconds=2)).timestamp(), tz=timezone.utc)
    gate = FixedGate(True)
    vad = SpeechDetectorDummy(turn_taking_gate=gate)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    metadata = {"recording_id": "recording", "vad_performance": {"speech_end_at": speech_end_at}}
    try:
        await dispatch(vad, metadata=metadata)
        assert gate.calls == [("yes", "abc", "session")]
        assert gate.full_texts == ["abcdefghij"]
        assert seen[0][2] is metadata
        assert seen[0][2]["vad_performance"]["speech_end_at"] is speech_end_at
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_explicit_text_condition_replaces_default_duration_condition(clock):
    gate = FixedGate(False)
    bypass = TurnTakingBypassGate(skip_condition=lambda text, duration: len(text or "") >= 5)
    manager = TurnTakingGateManager([bypass, gate])
    vad = SpeechDetectorDummy(turn_taking_gate=manager, on_recording_started_min_duration=1.5)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    try:
        await dispatch(vad, text="yes", duration=2.0)
        await dispatch(vad, text="long text", duration=0.2)
        assert [item[1] for item in seen] == ["long text"]
        assert [item[0] for item in gate.calls] == ["yes"]
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_explicit_false_disables_duration_skip(clock):
    gate = FixedGate(False)
    bypass = TurnTakingBypassGate(skip_condition=lambda text, duration: False)
    manager = TurnTakingGateManager([bypass, gate])
    vad = SpeechDetectorDummy(turn_taking_gate=manager)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    try:
        await dispatch(vad, duration=3.0)
        assert seen == []
        assert len(gate.calls) == 1
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_shared_gate_isolates_detector_conditions_and_session_cleanup(clock):
    gate = FixedGate(False)
    bypass = TurnTakingBypassGate()
    manager = TurnTakingGateManager([bypass, gate])
    first = SpeechDetectorDummy(turn_taking_gate=manager, on_recording_started_min_duration=1.0)
    second = SpeechDetectorDummy(turn_taking_gate=manager, on_recording_started_min_duration=2.0)
    first_state = begin_playback(first, clock, "first-session")
    clock.now = 100.0
    second_state = begin_playback(second, clock, "second-session")
    first_seen, second_seen = collect_speech(first), collect_speech(second)
    try:
        await dispatch(first, duration=1.5, session_id="first-session")
        await dispatch(second, duration=1.5, session_id="second-session")
        assert first_state is not second_state
        assert len(first_seen) == 1
        assert second_seen == []
        assert len(gate.calls) == 1
        assert bypass.skip_condition is None

        await first.finalize_session("first-session")
        assert first_state.is_closed
        assert not second_state.is_closed
        assert manager.get_session("second-session") is second_state
        assert second_state.estimate_playback().assistant_spoken_text == "abcde"
        await dispatch(second, text="still judged", duration=1.5, session_id="second-session")
        assert second_seen == []
        assert gate.calls[-1] == ("still judged", "abcde", "second-session")
        await dispatch(second, text="still skipped", duration=2.0, session_id="second-session")
        assert [item[1] for item in second_seen] == ["still skipped"]
        assert len(gate.calls) == 2
    finally:
        await first.finalize_session("first-session")
        await second.finalize_session("second-session")


@pytest.mark.asyncio
@pytest.mark.parametrize("new_text,new_duration", [("new input", 0.3), ("long input", 2.0), (None, 0.3)])
async def test_new_input_supersedes_pending_gate_even_when_bypassed(clock, new_text, new_duration):
    gate = BlockingGate()
    bypass = TurnTakingBypassGate()
    manager = TurnTakingGateManager([bypass, gate])
    vad = SpeechDetectorDummy(turn_taking_gate=manager)
    begin_playback(vad, clock)
    seen = collect_speech(vad)
    pending = asyncio.create_task(dispatch(vad, text="old input"))
    try:
        await asyncio.wait_for(gate.entered.wait(), 1.0)
        await dispatch(vad, text=new_text, duration=new_duration)
        assert [item[1] for item in seen] == [new_text]
        await asyncio.wait_for(gate.cancelled.wait(), 1.0)
        gate.release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert [item[1] for item in seen] == [new_text]
    finally:
        gate.release_cleanup.set()
        await vad.finalize_session("session")
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_closed_session_notification_does_not_use_reconnected_session(clock):
    gate = FixedGate(True)
    vad = SpeechDetectorDummy(turn_taking_gate=gate)
    original = begin_playback(vad, clock)
    seen = collect_speech(vad)
    pending = vad._execute_on_speech_detected(b"audio", "old input", {}, 0.3, "session")
    try:
        await vad.finalize_session("session")
        assert vad._init_turn_taking_session("reconnected") is None
        replacement = gate.get_session("reconnected")
        assert replacement is not original
        assert original.is_closed
        await pending
        assert seen == []
        assert gate.calls == []
        await asyncio.create_task(vad._execute_on_speech_detected(
            b"audio", "new input", {}, 0.3, "reconnected",
        ))
        assert seen == [(b"audio", "new input", {}, 0.3, "reconnected")]
    finally:
        await vad.finalize_session("session")
        await vad.finalize_session("reconnected")


@pytest.mark.asyncio
@pytest.mark.parametrize("detector_class", [SileroSpeechDetector, SileroStreamSpeechDetector], ids=["batch", "stream"])
async def test_silero_recording_reset_keeps_playback_but_delete_discards_it(make_silero, clock, detector_class):
    gate = FixedGate(False)
    vad = make_silero(detector_class, gate)
    recording = vad.get_session("session")
    state = begin_playback(vad, clock)
    try:
        assert vad._init_turn_taking_session("session") is None
        assert gate.get_session("session") is state
        assert state.estimate_playback().assistant_spoken_text == "abcde"
        vad.reset_session("session")
        assert vad.get_session("session") is recording
        assert gate.get_session("session") is state
        assert state.estimate_playback().assistant_spoken_text == "abcde"
        vad.delete_session("session")
        assert state.is_closed
        assert "session" not in vad.recording_sessions
        assert gate.get_session("session") is None
        vad.get_session("session")
        assert gate.get_session("session") is not None
        assert gate.get_session("session") is not state
        vad.delete_session("session")
    finally:
        await vad.finalize_session("session")


@pytest.mark.asyncio
@pytest.mark.parametrize("detector_class", [SileroSpeechDetector, SileroStreamSpeechDetector], ids=["batch", "stream"])
async def test_silero_async_execute_ignores_closed_session_and_dispatches_new_connection(make_silero, clock, detector_class):
    gate = FixedGate(True)
    vad = make_silero(detector_class, gate)
    vad.get_session("session")
    original = begin_playback(vad, clock)
    seen = collect_speech(vad)
    assert inspect.iscoroutinefunction(vad.execute_on_speech_detected)
    if detector_class is SileroStreamSpeechDetector:
        pending = vad.execute_on_speech_detected(b"audio", "old input", {}, 0.3, "session")
    else:
        pending = vad.execute_on_speech_detected(b"audio", 0.3, "session", {})
    try:
        vad.delete_session("session")
        vad.get_session("reconnected")
        assert gate.get_session("reconnected") is not original
        assert original.is_closed
        await pending
        assert seen == []
        assert gate.calls == []
        if detector_class is SileroStreamSpeechDetector:
            notification = vad.execute_on_speech_detected(b"audio", "new input", {}, 0.3, "reconnected")
            expected_text = "new input"
        else:
            notification = vad.execute_on_speech_detected(b"audio", 0.3, "reconnected", {})
            expected_text = None
        await asyncio.create_task(notification)
        assert seen == [(b"audio", expected_text, {}, 0.3, "reconnected")]
    finally:
        await vad.finalize_session("session")
        await vad.finalize_session("reconnected")


@pytest.mark.asyncio
@pytest.mark.parametrize("detector_class", [SileroSpeechDetector, SileroStreamSpeechDetector], ids=["batch", "stream"])
async def test_finalize_joins_cancelled_decisions_without_emitting_speech(make_silero, clock, detector_class):
    gate = BlockingGate()
    vad = make_silero(detector_class, gate)
    vad.get_session("session")
    state = begin_playback(vad, clock)
    seen = collect_speech(vad)
    pending = asyncio.create_task(dispatch(vad, text="old input"))
    finalization = None
    try:
        await asyncio.wait_for(gate.entered.wait(), 1.0)
        finalization = asyncio.create_task(vad.finalize_session("session"))
        await asyncio.wait_for(gate.cancelled.wait(), 1.0)
        assert not finalization.done()
        assert state.is_closed
        gate.release_cleanup.set()
        await asyncio.wait_for(finalization, 1.0)
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert seen == []
        assert "session" not in vad.recording_sessions
        await vad.finalize_session("session")
    finally:
        gate.release_cleanup.set()
        await vad.finalize_session("session")
        await asyncio.gather(*[task for task in (pending, finalization) if task], return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "end_path,recognized_text",
    [("max_duration", "old input"), ("silence", "old input"), ("silence", None)],
    ids=["max_duration", "silence", "silence_without_text"],
)
async def test_reconnect_during_pending_recognition_discards_old_input(
    make_silero, clock, monkeypatch, caplog, end_path, recognized_text,
):
    gate = FixedGate(True, debug=True)
    vad = make_silero(SileroStreamSpeechDetector, gate)
    caplog.set_level("INFO", logger="aiavatar.sts.vad.turn_taking_gates.base")
    monkeypatch.setattr(vad, "_detect_speech_silero", lambda *args: False)
    recording = vad.get_session("session")
    original = begin_playback(vad, clock)
    seen = collect_speech(vad)
    entered, release = asyncio.Event(), asyncio.Event()
    recognition_calls, recognition_errors = [], []

    async def fallback_recognition(*args):
        recognition_calls.append(args)
        return SimpleNamespace(text="late final text")

    monkeypatch.setattr(vad, "_recognize_audio", fallback_recognition)

    @vad.on_speech_recognition_error
    async def recognition_error(error, session_id):
        recognition_errors.append((error, session_id))

    async def pending_recognition():
        entered.set()
        await release.wait()
        recording.last_recognized_text = recognized_text

    recording.is_recording = True
    recording.record_duration = 0.8
    recording.silence_duration = 0.5
    recording.segment_fired = True
    recording.buffer.extend(b"\x00\x00" * 512)
    vad.max_duration = 0.8 if end_path == "max_duration" else 10.0
    recognition = asyncio.create_task(pending_recognition())
    recording.pending_recognition_task = recognition
    processing = asyncio.create_task(vad.process_samples(b"\x00\x00" * 512, "session"))
    try:
        await asyncio.wait_for(entered.wait(), 1.0)
        # Recognition may finish after the connection using this ID is closed.
        assert not processing.done()
        vad.delete_session("session")
        replacement = vad.get_session("reconnected")
        assert replacement is not recording
        assert original.is_closed
        release.set()
        assert await asyncio.wait_for(processing, 1.0) is False
        await asyncio.sleep(0)  # Let the old input reach the gate.
        assert seen == []
        assert gate.calls == []
        assert "Turn Taking Gate: session=session," in caplog.text
        assert "action=discard, reason=session_missing_or_closed" in caplog.text
        assert len(recognition_calls) == int(recognized_text is None)
        assert recognition_errors == []
        assert replacement.last_recognized_text is None
    finally:
        release.set()
        await asyncio.gather(recognition, processing, return_exceptions=True)
        await vad.finalize_session("session")
        await vad.finalize_session("reconnected")


@pytest.mark.asyncio
async def test_reconnect_during_batch_turn_end_gate_discards_old_input(make_silero, clock, monkeypatch, caplog):
    gate = FixedGate(True, debug=True)
    vad = make_silero(SileroSpeechDetector, gate)
    caplog.set_level("INFO", logger="aiavatar.sts.vad.turn_taking_gates.base")
    monkeypatch.setattr(vad, "_detect_speech_silero", lambda *args: False)
    recording = vad.get_session("session")
    original = begin_playback(vad, clock)
    seen = collect_speech(vad)
    entered, release = asyncio.Event(), asyncio.Event()

    async def pending_turn_end(*args):
        entered.set()
        await release.wait()
        return True

    monkeypatch.setattr(vad, "_should_end_turn_with_gate", pending_turn_end)
    recording.is_recording = True
    recording.record_duration = 0.8
    recording.silence_duration = 0.5
    recording.buffer.extend(b"\x00\x00" * 512)
    processing = asyncio.create_task(vad.process_samples(b"\x00\x00" * 512, "session"))
    try:
        await asyncio.wait_for(entered.wait(), 1.0)
        vad.delete_session("session")
        vad.get_session("reconnected")
        assert original.is_closed
        release.set()
        assert await asyncio.wait_for(processing, 1.0) is False
        await asyncio.sleep(0)
        assert seen == []
        assert gate.calls == []
        assert "Turn Taking Gate: session=session," in caplog.text
        assert "action=discard, reason=session_missing_or_closed" in caplog.text
    finally:
        release.set()
        await asyncio.gather(processing, return_exceptions=True)
        await vad.finalize_session("session")
        await vad.finalize_session("reconnected")


@pytest.mark.asyncio
async def test_cancelled_audio_stream_joins_pending_gate_cleanup(make_silero, clock):
    gate = BlockingGate()
    vad = make_silero(SileroStreamSpeechDetector, gate)
    vad.get_session("session")
    state = begin_playback(vad, clock)
    seen = collect_speech(vad)
    stream_entered = asyncio.Event()

    async def input_stream():
        stream_entered.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    pending = asyncio.create_task(dispatch(vad, text="old input"))
    streaming = asyncio.create_task(vad.process_stream(input_stream(), "session"))
    try:
        await asyncio.wait_for(gate.entered.wait(), 1.0)
        await asyncio.wait_for(stream_entered.wait(), 1.0)
        streaming.cancel()
        await asyncio.wait_for(gate.cancelled.wait(), 1.0)
        assert state.is_closed
        assert not streaming.done()
        gate.release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await streaming
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert seen == []
        assert "session" not in vad.recording_sessions
    finally:
        gate.release_cleanup.set()
        if not streaming.done():
            streaming.cancel()
        await asyncio.gather(pending, streaming, return_exceptions=True)
        await vad.finalize_session("session")


@pytest.mark.asyncio
async def test_only_accepted_speech_creates_and_invokes_an_sts_request(clock):
    from aiavatar.sts.models import STSRequest
    from aiavatar.sts.pipeline import STSPipeline

    gate = FixedGate(False)
    vad = SpeechDetectorDummy(turn_taking_gate=gate)
    begin_playback(vad, clock)
    vad.set_session_data("session", "user_id", "user", create_session=True)
    vad.set_session_data("session", "context_id", "context")
    requests = []

    async def memory_invoke(request):
        requests.append(request)
        for response in ():
            yield response

    pipeline = object.__new__(STSPipeline)
    pipeline.vad = vad
    pipeline.response_handlers = [SimpleNamespace(can_handle=lambda session_id: True)]
    pipeline.invoke = memory_invoke
    vad.on_speech_detected(pipeline.on_speech_detected)
    metadata = {"recording_id": "recording"}
    try:
        await dispatch(vad, text="backchannel", metadata=metadata)
        assert requests == []
        gate.outcome = True
        await dispatch(vad, text="question", metadata=metadata)
        assert len(requests) == 1
        request = requests[0]
        assert isinstance(request, STSRequest)
        assert request.session_id == "session"
        assert request.user_id == "user"
        assert request.context_id == "context"
        assert request.text == "question"
        assert request.audio_data == b"audio"
        assert request.audio_duration == 0.3
        assert request.metadata == metadata
    finally:
        await vad.finalize_session("session")
