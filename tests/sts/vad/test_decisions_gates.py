"""Mocked regression checks and billable live Decisions API tests.

Live cases run during normal pytest execution when OPENAI_API_KEY is set in the
environment or pytest.ini's [pytest] env block; no opt-in flag is required.
They send synthetic text only and close their HTTP clients after each case.
Use -s to display live probabilities and timings.
"""

import asyncio
import configparser
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import httpx
import pytest

from aiavatar.sts.vad.turn_end_gates.decisions import DecisionsTurnEndGate
from aiavatar.sts.vad.turn_end_gates.manager import TurnEndGateManager
from aiavatar.sts.vad.turn_taking_gates.decisions import DecisionsTurnTakingGate
from aiavatar.sts.vad.turn_taking_gates.manager import TurnTakingGateManager
import aiavatar.sts.vad.turn_taking_gates.session as session_module


@pytest.fixture(params=["hold", "take_turn"])
def kind(request):
    return request.param


def gate_type(kind):
    return DecisionsTurnEndGate if kind == "hold" else DecisionsTurnTakingGate


def response_body(kind, probability):
    return {"answers": [{"type": "predicate", "name": kind, "probability": probability}]}


async def decide(gate, text="うん", **kwargs):
    if isinstance(gate, DecisionsTurnEndGate):
        return await gate.should_end_turn(
            audio=b"not-uploaded", sample_rate=16000, channels=1,
            recorded_duration=1.5, silence_duration=0.3, session_id="test-session", text=text,
        )
    return await gate.should_take_turn(
        text, " 説明の途中 ", session_id="test-session", assistant_full_text=" 説明の途中です。 ", **kwargs,
    )


def assert_pass_without_score(decision, reason):
    assert decision.reason == reason
    if hasattr(decision, "should_end"):
        assert decision.should_end is True
        assert decision.confidence is None
        assert decision.timeout is None
    else:
        assert decision.should_take_turn is True
        assert decision.probability is None


@pytest.mark.asyncio
async def test_request_contract_uses_named_predicate_and_caller_owned_client(kind):
    requests = []

    def respond(request):
        requests.append(request)
        # The relevant answer is deliberately not first.
        return httpx.Response(200, json={"answers": [
            {"name": "other", "type": "predicate", "probability": 0},
            response_body(kind, .9)["answers"][0],
        ]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = gate_type(kind)(http_client=client, api_key=" test-key ", model=" test-model ",
                               instructions="Custom predicate criteria.", request_timeout=3)
        result = await decide(gate, "  待って  ")
        assert client.is_closed is False
    assert result.reason == ("decisions_long_hold" if kind == "hold" else "decisions_take_turn")
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/decisions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert all(t == 3 for t in request.extensions["timeout"].values())
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["questions"] == [{"type": "predicate", "name": kind, "instructions": "Custom predicate criteria."}]
    assert isinstance(payload["input"], str)
    assert json.loads(payload["input"]) == (
        {"transcript": "待って", "recorded_duration_seconds": 1.5, "silence_duration_seconds": .3}
        if kind == "hold" else
        {"user_text": "待って", "assistant_spoken_text": "説明の途中", "assistant_full_text": "説明の途中です。"}
    )
    assert "not-uploaded" not in request.content.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("probability,end,timeout", [
    (0, True, None), (.599, True, None), (.6, False, .4), (.799, False, .4),
    (.8, False, 1), (.899, False, 1), (.9, False, 2), (1, False, 2),
])
async def test_turn_end_range_boundaries(probability, end, timeout):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response_body("hold", probability))
    )) as client:
        gate = DecisionsTurnEndGate(http_client=client, api_key="test-key")
        result = await decide(gate)
    assert gate.model == "gpt-6-luna"
    assert gate.name == "decisions"
    assert gate.run_in_background is True
    assert gate.timeout == 2
    assert result.should_end is end
    assert result.confidence == pytest.approx(1 - probability)
    assert result.timeout == timeout
    assert result.pending is False


@pytest.mark.asyncio
@pytest.mark.parametrize("probability,take", [
    (0, False), (.2, False), (.299, False), (.3, False), (.301, True), (.5, True), (1, True),
])
async def test_turn_taking_default_boundary(probability, take):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response_body("take_turn", probability))
    )) as client:
        gate = DecisionsTurnTakingGate(http_client=client, api_key="test-key")
        result = await decide(gate)
    assert gate.model == "gpt-6-luna"
    assert gate.discard_threshold == .3
    assert result.should_take_turn is take
    assert result.probability == probability
    assert result.reason == ("decisions_take_turn" if take else "decisions_backchannel")


@pytest.mark.asyncio
async def test_explicit_previous_turn_taking_threshold_still_overrides_default():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response_body("take_turn", .3))
    )) as client:
        gate = DecisionsTurnTakingGate(http_client=client, api_key="test-key", discard_threshold=.2)
        result = await decide(gate)
    assert result.should_take_turn is True
    assert result.probability == .3


@pytest.mark.asyncio
async def test_custom_threshold_and_hold_ranges_are_respected():
    def respond(request):
        name = json.loads(request.content)["questions"][0]["name"]
        return httpx.Response(200, json=response_body(name, .5))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        ranges = [[.3, .2], [.5, .7], [.95, 3]]
        end_gate = DecisionsTurnEndGate(http_client=client, api_key="test-key", hold_ranges=ranges)
        ranges[1][0] = .8
        ranges.clear()
        end = await decide(end_gate)
        take = await decide(DecisionsTurnTakingGate(http_client=client, api_key="test-key", discard_threshold=.5))
    assert end.should_end is False and end.timeout == .7
    assert end_gate.timeout == 3
    assert take.should_take_turn is False


@pytest.mark.asyncio
async def test_legacy_two_range_configuration_matches_jev():
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response_body("hold", .5))
    )) as client:
        gate = DecisionsTurnEndGate(http_client=client, api_key="test-key", hold_timeout=.6)
        result = await decide(gate)
    assert gate.hold_ranges == ((.5, .6), (.8, 2))
    assert result.timeout == .6


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "", " \n\t"])
async def test_empty_text_passes_without_request(kind, text):
    def unexpected(_):
        pytest.fail("Empty text must not call the API")
    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        result = await decide(gate_type(kind)(http_client=client, api_key="test-key"), text)
    assert_pass_without_score(result, "decisions_no_text")


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    [], {}, {"answers": {}}, {"answers": []}, {"answers": [None]},
    {"answers": [{"name": "unrelated", "type": "predicate", "probability": .9}]},
])
async def test_malformed_answer_containers_pass_without_score(kind, body):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=body)
    )) as client:
        result = await decide(gate_type(kind)(http_client=client, api_key="test-key"))
    assert_pass_without_score(result, "decisions_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, True, "0.5", -.1, 1.1, float("nan"), float("inf"), 10 ** 1000])
async def test_invalid_probabilities_pass_without_score(kind, invalid):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=json.dumps(response_body(kind, invalid)))
    )) as client:
        result = await decide(gate_type(kind)(http_client=client, api_key="test-key"))
    assert_pass_without_score(result, "decisions_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("malformation", ["duplicate", "wrong_type", "missing_probability", "not_json"])
async def test_ambiguous_or_incomplete_answers_are_not_accepted(kind, malformation):
    body = response_body(kind, .9)
    if malformation == "duplicate":
        body["answers"] *= 2
    elif malformation == "wrong_type":
        body["answers"][0]["type"] = "score"
    elif malformation == "missing_probability":
        del body["answers"][0]["probability"]
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content="not json" if malformation == "not_json" else json.dumps(body))
    )) as client:
        result = await decide(gate_type(kind)(http_client=client, api_key="test-key"))
    assert_pass_without_score(result, "decisions_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 429, 500, 529])
async def test_http_failure_does_not_retry_follow_redirects_or_log_secrets(kind, status, caplog):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": "https://must-not-follow.invalid"}, text="private-body")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        result = await decide(gate_type(kind)(http_client=client, api_key="test-key"))
        assert client.is_closed is False
    assert_pass_without_score(result, "decisions_error")
    assert len(calls) == 1
    assert "test-key" not in caplog.text and "private-body" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failure_and_timeout_have_safe_fallbacks(kind, error_class, caplog):
    def respond(request):
        raise error_class("private-exception-test-key", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await decide(gate_type(kind)(http_client=client, api_key="test-key"))
    reason = "decisions_timeout" if kind == "take_turn" and error_class == httpx.ReadTimeout else "decisions_error"
    assert_pass_without_score(result, reason)
    assert "private-exception" not in caplog.text and "test-key" not in caplog.text


@pytest.mark.asyncio
async def test_total_timeout_cancels_http_work_and_keeps_client_open(kind):
    cancelled = asyncio.Event()
    async def respond(_):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await decide(gate_type(kind)(http_client=client, api_key="test-key", request_timeout=.01))
        assert client.is_closed is False
    assert cancelled.is_set()
    assert_pass_without_score(result, "decisions_error" if kind == "hold" else "decisions_timeout")


@pytest.mark.asyncio
async def test_external_cancellation_propagates(kind):
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def respond(_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(decide(gate_type(kind)(http_client=client, api_key="test-key")))
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()
            assert client.is_closed is False
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("debug", [False, True])
async def test_transcripts_are_logged_only_when_debug_enabled(kind, debug, caplog):
    caplog.set_level(logging.INFO, logger="aiavatar.sts.vad")
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response_body(kind, .9))
    )) as client:
        await decide(gate_type(kind)(http_client=client, api_key="test-key", debug=debug), "private-transcript")
    assert ("private-transcript" in caplog.text) is debug
    assert "test-key" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("option,value", [
    ("api_key", " "), ("api_key", None), ("model", ""), ("model", 1), ("instructions", 1),
    *[("request_timeout", v) for v in (0, -1, True, "1", float("nan"), float("inf"), 10 ** 1000)],
])
async def test_invalid_common_configuration_rejected(kind, option, value):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)) as client:
        with pytest.raises(ValueError):
            gate_type(kind)(**{"http_client": client, "api_key": "test-key", option: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"hold_ranges": []}, {"hold_ranges": [(.4, .2), (.4, .3)]},
    {"hold_ranges": [(.6, .4), (.8, .2)]}, {"hold_ranges": [(.6, 0)]},
    {"hold_ranges": [(True, 1)]}, {"hold_ranges": [(1.1, 1)]},
    {"hold_ranges": [(.5, float("inf"))]}, {"hold_ranges": [(.5,)]},
    {"hold_ranges": 3}, {"hold_ranges": [(.6, 1)], "hold_timeout": .3},
])
async def test_invalid_hold_policy_rejected(options):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)) as client:
        with pytest.raises(ValueError):
            DecisionsTurnEndGate(http_client=client, api_key="test-key", **options)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [-.1, 1.1, True, "0.2", float("nan"), float("inf")])
async def test_invalid_discard_threshold_rejected(value):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None)) as client:
        with pytest.raises(ValueError):
            DecisionsTurnTakingGate(http_client=client, api_key="test-key", discard_threshold=value)


@pytest.mark.asyncio
async def test_turn_end_manager_replaces_pending_timeout_and_releases_hold():
    session = SimpleNamespace(session_id="test", turn_end_gate_hold_active=False,
                              turn_end_gate_hold_timeout=None, turn_end_gate_hold_reasons=[])
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response_body("hold", .8))
    )) as client:
        gate = DecisionsTurnEndGate(http_client=client, api_key="test-key")
        manager = TurnEndGateManager([gate])
        async def evaluate(silence):
            return await manager.should_end_turn(
                session=session, audio=b"", sample_rate=16000, channels=1,
                recorded_duration=2, silence_duration=silence, silence_duration_threshold=.2, text="えっと",
            )
        tasks = []
        try:
            assert await evaluate(.3) is False
            assert session.turn_end_gate_hold_reasons == ["decisions_pending"]
            assert session.turn_end_gate_hold_timeout == 2
            tasks = list(manager._states[session.session_id].pending_tasks.values())
            await asyncio.gather(*tasks)
            assert await evaluate(.4) is False
            assert session.turn_end_gate_hold_timeout == 1
            assert await evaluate(1.2) is True
            assert session.turn_end_gate_hold_active is False
        finally:
            manager.reset_session(session.session_id, session=session)
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_turn_taking_session_sends_spoken_prefix_and_full_text(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_body("take_turn", .1))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = DecisionsTurnTakingGate(http_client=client, api_key="test-key")
        session = gate.get_session("test", create=True)
        try:
            assert session.start_playback("chunk", "今日の予定を教えてください", 10, transaction_id="response")
            clock.now += 5
            result = await gate.evaluate("test", "うん")
            assert result.should_take_turn is False
            state = json.loads(requests[0]["input"])
            question = "今日の予定を教えてください"
            assert state["assistant_spoken_text"] == question[:len(question) // 2]
            assert state["assistant_full_text"] == "今日の予定を教えてください"
        finally:
            await gate.close_session("test").aclose()
        assert client.is_closed is False


@pytest.mark.asyncio
async def test_turn_taking_manager_and_disabled_bypass_work_with_idle_playback():
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=response_body("take_turn", .1))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        child = DecisionsTurnTakingGate(http_client=client, api_key="test-key")
        manager = TurnTakingGateManager([child], bypass_enabled=False)
        manager.get_session("managed", create=True)
        child.get_session("standalone", create=True)
        try:
            assert (await child.evaluate("standalone", "うん")).should_take_turn is True
            assert not calls  # Default idle-playback bypass.
            result = await manager.evaluate("managed", "うん")
            assert result.should_take_turn is False
            assert len(calls) == 1
        finally:
            await manager.close_session("managed").aclose()
            await child.close_session("standalone").aclose()


@pytest.mark.asyncio
async def test_session_close_cancels_only_its_pending_request():
    started = {name: asyncio.Event() for name in ("a", "b")}
    release = asyncio.Event()
    async def respond(request):
        name = json.loads(json.loads(request.content)["input"])["user_text"]
        started[name].set()
        await release.wait()
        return httpx.Response(200, json=response_body("take_turn", .9))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = DecisionsTurnTakingGate(http_client=client, api_key="test-key", bypass_enabled=False)
        sessions = {name: gate.get_session(name, create=True) for name in started}
        tasks = {name: asyncio.create_task(gate.evaluate(name, name)) for name in started}
        try:
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())), 1)
            await gate.close_session("a").aclose()
            with pytest.raises(asyncio.CancelledError):
                await tasks["a"]
            assert not tasks["b"].done()
            release.set()
            assert (await tasks["b"]).should_take_turn is True
            assert client.is_closed is False
        finally:
            for session in sessions.values():
                await session.aclose()
            await asyncio.gather(*tasks.values(), return_exceptions=True)


@pytest.fixture
def create_live_decisions_gate():
    __tracebackhide__ = True
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        config = configparser.ConfigParser(interpolation=None)
        try:
            config.read(Path(__file__).resolve().parents[3] / "pytest.ini")
        except (configparser.Error, OSError):
            pytest.fail("Unable to read pytest.ini for OPENAI_API_KEY", pytrace=False)
        for line in config.get("pytest", "env", fallback="").splitlines():
            name, separator, value = line.strip().partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                api_key = value.strip()
                break
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required in the environment or pytest.ini")

    # Return a factory so pytest never displays the credential as a test argument.
    def create(gate_class, http_client):
        __tracebackhide__ = True
        # Allow cold connection setup; keep the default model, prompt and thresholds.
        return gate_class(http_client=http_client, api_key=api_key, request_timeout=5.0)

    return create


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,transcript,expected_end", [
    ("complete_question", "東京の明日の天気を教えてください。", True),
    ("polite_request", "予約を変更したいんですけど。", True),
    ("hesitation", "えっと、あの、その、うーん", False),
    ("explicit_wait", "少し考えるので、まだ返事をせずに待っていてください。", False),
], ids=["complete_question", "polite_request", "hesitation", "explicit_wait"])
async def test_live_turn_end(create_live_decisions_gate, case_id, transcript, expected_end):
    async with httpx.AsyncClient() as client:
        gate = create_live_decisions_gate(DecisionsTurnEndGate, client)
        started_at = time.perf_counter()
        decision = await gate.should_end_turn(
            audio=b"", sample_rate=16000, channels=1,
            recorded_duration=2.0, silence_duration=0.5,
            session_id=f"live-end-{case_id}", text=transcript,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000

    # A fail-open fallback must not count as a successful end-of-turn prediction.
    assert decision.confidence is not None, f"No API probability: {decision.reason}"
    assert 0 <= decision.confidence <= 1
    print(f"turn_end/{case_id}: hold={1 - decision.confidence:.4f}, elapsed={elapsed_ms:.1f}ms")
    assert decision.should_end is expected_end, decision
    if expected_end:
        assert decision.reason == "decisions_complete"
        assert decision.timeout is None
    else:
        assert decision.reason in ("decisions_incomplete", "decisions_long_hold")
        assert decision.timeout in (0.4, 1.0, 2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,user_text,spoken_text,full_text,expected_take", [
    (
        "backchannel", "うんうん",
        "植物は光合成で光のエネルギーを使って養分を作り、その養分を使ってさらに",
        "植物は光合成で光のエネルギーを使って養分を作り、その養分を使ってさらに成長していきます。",
        False,
    ),
    (
        "answer", "はい、お願いします。",
        "この内容で予約してもよろしいですか？", "この内容で予約してもよろしいですか？", True,
    ),
    (
        "stop_request", "ちょっと待って、説明を止めてください。",
        "これから手続きの流れを順番に説明します。まず最初に",
        "これから手続きの流れを順番に説明します。まず最初に申込書を記入します。", True,
    ),
    (
        "unheard_question", "うんうん",
        "植物は光合成で光のエネルギーを使って養分を作り、その養分を使ってさらに",
        "植物は光合成で光のエネルギーを使って養分を作り、その養分を使ってさらに成長していきます。ここまでの説明は分かりましたか？",
        False,
    ),
], ids=["backchannel", "answer", "stop_request", "unheard_question"])
async def test_live_turn_taking(
    create_live_decisions_gate, case_id, user_text, spoken_text, full_text, expected_take,
):
    async with httpx.AsyncClient() as client:
        gate = create_live_decisions_gate(DecisionsTurnTakingGate, client)
        started_at = time.perf_counter()
        # Direct classification ensures idle-playback bypass cannot skip the API.
        decision = await gate.should_take_turn(
            user_text, spoken_text, assistant_full_text=full_text,
            session_id=f"live-take-{case_id}",
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000

    assert decision.probability is not None, f"No API probability: {decision.reason}"
    assert 0 <= decision.probability <= 1
    print(f"turn_taking/{case_id}: take={decision.probability:.4f}, elapsed={elapsed_ms:.1f}ms")
    assert decision.should_take_turn is expected_take, decision
    assert decision.reason == ("decisions_take_turn" if expected_take else "decisions_backchannel")
