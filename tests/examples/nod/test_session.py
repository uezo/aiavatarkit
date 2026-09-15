"""Hermetic checks for nod lifetime, delivery, and bounded conversation memory."""

import asyncio
import json
import logging

import pytest

from examples.nod import NodDecision, NodSession


class FakeEngine:
    def __init__(self, *, blocked=False, swallow_cancel=False, decision=None):
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.blocked = blocked
        self.swallow_cancel = swallow_cancel
        self.decision = decision or NodDecision("neutral", "うん", "stop", 12.5)
        self.closed = False
        self.prompt = "SYSTEM_PROMPT_MUST_NOT_BE_LOGGED"

    async def decide(self, user_input):
        self.calls.append(user_input)
        self.entered.set()
        if self.blocked:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                if not self.swallow_cancel:
                    raise
        return self.decision

    async def close(self):
        self.closed = True


class Delivery:
    def __init__(self):
        self.sent = []

    async def __call__(self, decision):
        self.sent.append(decision)
        return True


async def wait_for(event):
    await asyncio.wait_for(event.wait(), 1)


def start(session, text="公園に行ったんだ", identifier="u1"):
    session.start_input(identifier)
    assert session.update_user(identifier, text)


@pytest.mark.asyncio
async def test_pause_waits_for_transcript_and_delivers_exact_judged_snapshot():
    engine, delivery = FakeEngine(), Delivery()
    async with NodSession(engine, delivery) as session:
        session.start_input("u1")
        task = asyncio.create_task(session.on_pause("u1"))
        await asyncio.sleep(0)
        assert not task.done() and not engine.calls
        assert session.update_user("u1", "公園に行ったんだ")
        result = await asyncio.wait_for(task, 1)
        assert result.outcome == "sent"
        assert result.utterance_id == "u1"
        assert result.acknowledged_text == "公園に行ったんだ"
        assert result.text_length == len("公園に行ったんだ")
        assert delivery.sent[0].id == "neutral"
        assert "ユーザー：公園に行ったんだ" in engine.calls[0]
        assert "公園に行ったんだ<nod_assistant>うん</nod_assistant>" in session.build_input("u1")
        assert not session.is_current(delivery.sent[0])
    assert not engine.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["transcript", "api"])
async def test_duplicate_pauses_are_dropped_without_parallel_decisions(phase):
    engine, delivery = FakeEngine(blocked=True), Delivery()
    async with NodSession(engine, delivery) as session:
        start(session, "" if phase == "transcript" else "公園に行ったんだ")
        first = asyncio.create_task(session.on_pause("u1"))
        if phase == "api":
            await wait_for(engine.entered)
        else:
            await asyncio.sleep(0)
        second = await session.on_pause("u1")
        assert second.outcome == "busy"
        session.update_user("u1", "公園に行ったんだ")
        engine.release.set()
        assert (await asyncio.wait_for(first, 1)).outcome == "sent"
        assert len(engine.calls) == len(delivery.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_missing_transcript_expires_without_querying_engine(expired):
    engine, delivery = FakeEngine(), Delivery()
    async with NodSession(engine, delivery, timeout=0.03) as session:
        session.start_input("u1")
        detected_at = asyncio.get_running_loop().time() - 1 if expired else None
        result = await session.on_pause("u1", detected_at=detected_at)
        assert result.outcome == "timeout"
        assert not engine.calls and not delivery.sent


@pytest.mark.asyncio
async def test_api_deadline_cancels_request_and_does_not_remember_candidate():
    engine, delivery = FakeEngine(blocked=True), Delivery()
    async with NodSession(engine, delivery, timeout=0.03) as session:
        start(session)
        result = await session.on_pause("u1")
        assert result.outcome == "timeout" and engine.cancelled.is_set()
        assert not delivery.sent
        assert "<nod_assistant>" not in session.build_input("u1")


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["transcript", "api"])
@pytest.mark.parametrize("action", ["start", "end", "close"])
async def test_input_transition_and_close_cancel_and_join_work(phase, action):
    engine, delivery = FakeEngine(blocked=True), Delivery()
    async with NodSession(engine, delivery) as session:
        start(session, "" if phase == "transcript" else "公園に行ったんだ")
        pending = asyncio.create_task(session.on_pause("u1"))
        if phase == "api":
            await wait_for(engine.entered)
        else:
            await asyncio.sleep(0)
        if action == "start":
            session.start_input("u2")
        elif action == "end":
            session.end_input("u1")
        else:
            await session.close()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert pending.done() and not delivery.sent
        assert (await session.on_pause("u1")).outcome == "inactive"
        if action == "close":
            with pytest.raises(RuntimeError, match="closed"):
                session.start_input("u2")
        else:
            engine.release.set()
            if action == "end":
                session.start_input("u2")
            session.update_user("u2", "桜もきれいだったよ")
            assert (await session.on_pause("u2")).outcome == "sent"
    assert not engine.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["start", "end", "close", "cancel", "timeout"])
async def test_stale_engine_output_is_rejected_even_if_engine_swallows_cancellation(action):
    engine, delivery = FakeEngine(blocked=True, swallow_cancel=True), Delivery()
    async with NodSession(engine, delivery, timeout=0.03 if action == "timeout" else 1) as session:
        start(session)
        pending = asyncio.create_task(session.on_pause("u1"))
        await wait_for(engine.entered)
        if action == "start":
            session.start_input("u2")
        elif action == "end":
            session.end_input("u1")
        elif action == "close":
            await session.close()
        elif action == "cancel":
            pending.cancel()
        if action != "timeout":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 1)
        else:
            result = await asyncio.wait_for(pending, 1)
            assert result.outcome in ("inactive", "timeout", "cancelled")
        assert engine.cancelled.is_set()
        assert not delivery.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("change,expected", [("append", "sent"), ("correct", "transcript_changed")])
async def test_resumed_appended_speech_survives_but_asr_correction_invalidates(change, expected):
    engine, delivery = FakeEngine(blocked=True), Delivery()
    async with NodSession(engine, delivery) as session:
        start(session)
        pending = asyncio.create_task(session.on_pause("u1"))
        await wait_for(engine.entered)
        session.update_user("u1", "公園に行ったんだ、桜がきれいで" if change == "append" else "映画に行ったんだ")
        engine.release.set()
        result = await asyncio.wait_for(pending, 1)
        assert result.outcome == expected
        assert len(delivery.sent) == (change == "append")
        if change == "append":
            assert "公園に行ったんだ<nod_assistant>うん</nod_assistant>、桜がきれいで" in session.build_input("u1")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["end", "correct", "append"])
async def test_emitter_can_recheck_decision_after_waiting_for_transport_lock(change):
    engine, entered, release = FakeEngine(), asyncio.Event(), asyncio.Event()
    delivered = []
    async def emit(decision):
        assert session.is_current(decision)
        entered.set()
        await release.wait()
        if not session.is_current(decision):
            return False
        delivered.append(decision)
        return True
    async with NodSession(engine, emit) as session:
        start(session)
        pending = asyncio.create_task(session.on_pause("u1"))
        await wait_for(entered)
        if change == "end":
            session.end_input("u1")
        else:
            session.update_user("u1", "違う話に訂正" if change == "correct" else "公園に行ったんだ、桜が咲いていて")
        release.set()
        if change == "end":
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert (await pending).outcome == ("sent" if change == "append" else "not_sent")
        assert len(delivered) == (change == "append")


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["false", "none", "error", "cancel"])
async def test_failed_or_cancelled_delivery_never_enters_history_or_cooldown(behavior):
    engine, attempts = FakeEngine(), []
    async def emit(decision):
        attempts.append(decision)
        if behavior == "error":
            raise RuntimeError("fake transport failure")
        if behavior == "cancel":
            raise asyncio.CancelledError
        return False if behavior == "false" else None
    async with NodSession(engine, emit) as session:
        start(session)
        for _ in range(2):
            if behavior == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await session.on_pause("u1")
            else:
                result = await session.on_pause("u1")
                assert result.outcome == ("error" if behavior == "error" else "not_sent")
        assert len(engine.calls) == len(attempts) == 2
        assert "<nod_assistant>" not in session.build_input("u1")


@pytest.mark.asyncio
async def test_silence_does_not_emit_or_start_cooldown():
    engine = FakeEngine(decision=NodDecision("none", outcome="none"))
    delivery = Delivery()
    async with NodSession(engine, delivery) as session:
        start(session)
        assert (await session.on_pause("u1")).outcome == "none"
        assert (await session.on_pause("u1")).outcome == "none"
        assert len(engine.calls) == 2 and not delivery.sent


@pytest.mark.asyncio
async def test_success_starts_input_cooldown_but_new_input_can_receive_nod():
    engine, delivery = FakeEngine(), Delivery()
    async with NodSession(engine, delivery, min_interval=60) as session:
        start(session)
        assert (await session.on_pause("u1")).outcome == "sent"
        session.update_user("u1", "公園に行ったんだ、桜がきれいだったよ")
        assert (await session.on_pause("u1")).outcome == "cooldown"
        assert len(engine.calls) == 1
        session.end_input("u1")
        start(session, "今日は海に行くよ", "u2")
        assert (await session.on_pause("u2")).outcome == "sent"
        assert len(engine.calls) == 2


@pytest.mark.asyncio
async def test_history_keeps_nods_through_late_asr_and_idempotent_assistant_updates():
    engine, delivery = FakeEngine(), Delivery()
    async with NodSession(engine, delivery, min_interval=0) as session:
        start(session, "分かんないけどね、")
        assert (await session.on_pause("u1")).outcome == "sent"
        session.end_input("u1")
        session.update_assistant("a1", "聞いてる")
        session.update_assistant("a1", "聞いてるよ。")
        start(session, "まあね", "u2")
        assert session.update_user("u1", "分かんないけどね、行ってみないと。")
        assert (await session.on_pause("u1")).outcome == "inactive"
        user_input = session.build_input("u2")
        assert "ユーザー：分かんないけどね、<nod_assistant>うん</nod_assistant>行ってみないと。" in user_input
        assert user_input.count("AI：聞いてるよ。") == 1
        assert user_input.index("分かんないけどね") < user_input.index("AI：聞いてるよ。") < user_input.index("ユーザー：まあね")
        session.start_input("u2")  # Duplicate current-input notification is harmless.
        assert session.build_input("u2") == user_input
        with pytest.raises(ValueError, match="cannot be reopened"):
            session.start_input("u1")
        assert not session.update_user("unknown", "知らない入力")


@pytest.mark.asyncio
async def test_same_text_different_ids_retains_nod_on_only_its_original_input():
    engine, delivery = FakeEngine(), Delivery()
    async with NodSession(engine, delivery) as session:
        start(session, "同じ発言", "same")
        await session.on_pause("same")
        session.end_input("same")
        session.update_assistant("same", "同じIDでも別の発話者")
        start(session, "同じ発言", "new")
        user_input = session.build_input("new")
        history, current = user_input.split("【今回のユーザー発言】")
        assert "同じ発言<nod_assistant>うん</nod_assistant>" in history
        assert "同じIDでも別の発話者" in history
        assert "<nod_assistant>" not in current


@pytest.mark.asyncio
async def test_two_sessions_share_engine_but_never_history_cooldown_or_ownership():
    engine, first_delivery, second_delivery = FakeEngine(), Delivery(), Delivery()
    async with NodSession(engine, first_delivery) as first, NodSession(engine, second_delivery) as second:
        start(first, "一人目の秘密", "same")
        start(second, "二人目の話", "same")
        assert (await first.on_pause("same")).outcome == "sent"
        assert (await second.on_pause("same")).outcome == "sent"
        assert "一人目の秘密" not in second.build_input("same")
        assert "二人目の話" not in first.build_input("same")
        await first.close()
        second.end_input("same")
        start(second, "続きの話", "next")
        assert (await second.on_pause("next")).outcome == "sent"
        assert not engine.closed


@pytest.mark.asyncio
async def test_history_and_long_text_are_bounded_without_losing_nod_offsets():
    engine, delivery = FakeEngine(), Delivery()
    async with NodSession(engine, delivery, max_text_chars=1800, min_interval=0) as session:
        text = "長い話" * 1500
        start(session, text)
        for _ in range(5):
            result = await session.on_pause("u1")
            assert result.text_length == len(text)
            assert len(result.acknowledged_text) == 600
            text += "それから"
            session.update_user("u1", text)
        user_input = session.build_input("u1")
        assert user_input.count("<nod_assistant>うん</nod_assistant>") == 3
        assert "<nod_assistant>うん</nod_assistant>それから" in user_input
        assert "位置不明" not in user_input
        session.end_input("u1")
        for number in range(20):
            session.update_assistant(f"a{number}", f"過去{number}：" + "文" * 4000)
        start(session, "今回の発言", "current")
        user_input = session.build_input("current")
        assert user_input.count("AI：") == 10
        history = user_input.split("【今回のユーザー発言】", 1)[0]
        assert history.count("文") == 6000
        assert not session.update_user("u1", "履歴から消えた入力")
        # Verify actual storage is bounded too, rather than only truncating prompts.
        assert len(session._messages) == 11
        assert all(len(message.text) <= 1800 for message in session._messages.values())
        assert all(len(message.nods) <= 3 for message in session._messages.values())


@pytest.mark.asyncio
async def test_correction_after_send_keeps_ack_as_unplaced_history_instead_of_erasing_it():
    engine, delivery = FakeEngine(), Delivery()
    async with NodSession(engine, delivery) as session:
        start(session, "公園に行ったんだ")
        await session.on_pause("u1")
        session.end_input("u1")
        session.update_user("u1", "講演に行ったんだ")
        start(session, "それでね", "u2")
        user_input = session.build_input("u2")
        assert "ユーザー：講演に行ったんだ" in user_input
        assert "AI（送信済み相槌・位置不明）：うん" in user_input
        assert "受け止めた内容：公園に行ったんだ" in user_input


@pytest.mark.asyncio
@pytest.mark.parametrize("log_inputs", [False, True])
async def test_logs_report_outcome_and_latency_with_optional_user_input_only(caplog, log_inputs):
    engine, delivery = FakeEngine(), Delivery()
    caplog.set_level(logging.INFO, logger="examples.nod.session")
    async with NodSession(engine, delivery, log_inputs=log_inputs) as session:
        start(session, "分析用の今回発言")
        expected_input = session.build_input("u1")
        await session.on_pause("u1")
    entries = [json.loads(record.message.removeprefix("Nod decision: ")) for record in caplog.records
               if record.message.startswith("Nod decision: ")]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["utterance_id"] == "u1" and entry["outcome"] == "sent"
    assert entry["id"] == "neutral" and entry["phrase"] == "うん"
    assert entry["elapsed_ms"] >= 0 and entry["api_ms"] == 12.5
    if log_inputs:
        assert entry["input"] == expected_input
    else:
        assert "input" not in entry and "分析用の今回発言" not in caplog.text
    assert engine.prompt not in caplog.text
