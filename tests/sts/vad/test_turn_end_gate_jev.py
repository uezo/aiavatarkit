"""Local Jev contract and lifecycle tests; no credentials or network required."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from aiavatar.sts.vad.turn_end_gates.jev import JevTurnEndGate
from aiavatar.sts.vad.turn_end_gates.manager import TurnEndGateManager


async def decide(gate, text="明日の予定は、えっと", session_id="jev_test"):
    return await gate.should_end_turn(
        audio=b"\x01\x00",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.5,
        silence_duration=0.3,
        session_id=session_id,
        text=text,
    )


def make_session(session_id):
    return SimpleNamespace(
        session_id=session_id,
        turn_end_gate_hold_active=False,
        turn_end_gate_hold_timeout=None,
        turn_end_gate_hold_reasons=[],
    )


async def manager_decide(manager, session, silence_duration=0.3):
    return await manager.should_end_turn(
        session=session,
        audio=b"\x01\x00",
        sample_rate=16000,
        channels=1,
        recorded_duration=1.5,
        silence_duration=silence_duration,
        silence_duration_threshold=0.2,
        text=session.session_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probability,should_end,timeout,reason",
    [
        (0, True, None, "jev_complete"),
        (0.599, True, None, "jev_complete"),
        (0.6, False, 0.4, "jev_incomplete"),
        (0.799, False, 0.4, "jev_incomplete"),
        (0.8, False, 1.0, "jev_incomplete"),
        (0.899, False, 1.0, "jev_incomplete"),
        (0.9, False, 2.0, "jev_long_hold"),
        (1, False, 2.0, "jev_long_hold"),
    ],
)
async def test_probability_boundaries(probability, should_end, timeout, reason):
    def respond(request):
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": probability}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key")
        decision = await decide(gate)

        assert decision.should_end is should_end
        assert decision.timeout == timeout
        assert decision.reason == reason
        assert decision.confidence == pytest.approx(1 - probability)
        assert decision.pending is False
        assert client.is_closed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probability,timeout",
    [(0, None), (0.199, None), (0.2, 0.15), (0.399, 0.15),
     (0.4, 0.4), (0.699, 0.4), (0.7, 0.8), (0.949, 0.8),
     (0.95, 3.0), (1, 3.0)],
)
async def test_custom_four_ranges_select_highest_matching_threshold(probability, timeout):
    def respond(request):
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": probability}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(
            http_client=client,
            api_key="test-key",
            hold_ranges=((0.2, 0.15), (0.4, 0.4), (0.7, 0.8), (0.95, 3.0)),
        )
        decision = await decide(gate)

    assert decision.should_end is (timeout is None)
    assert decision.timeout == timeout
    expected_reason = "jev_complete" if timeout is None else (
        "jev_long_hold" if probability >= 0.95 else "jev_incomplete"
    )
    assert decision.reason == expected_reason
    assert decision.confidence == pytest.approx(1 - probability)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hold_ranges,probability,timeout",
    [
        ([(0, 0.2)], 0, 0.2),
        ([(1, 0.2)], 0.999, None),
        ([(1, 0.2)], 1, 0.2),
        ([[0.3, 0.7]], 0.3, 0.7),
        ([[0.3, 0.7], [0.9, 0.7]], 0.9, 0.7),
    ],
)
async def test_single_range_endpoint_thresholds_and_equal_durations(hold_ranges, probability, timeout):
    def respond(request):
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": probability}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key", hold_ranges=hold_ranges)
        decision = await decide(gate)

    assert decision.should_end is (timeout is None)
    assert decision.timeout == timeout
    assert decision.reason == ("jev_complete" if timeout is None else "jev_long_hold")


@pytest.mark.asyncio
async def test_caller_mutation_does_not_change_configured_ranges_or_pending_timeout():
    def respond(request):
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": 0.65}}})

    ranges = [[0.6, 0.4], [0.9, 2.5]]
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key", hold_ranges=ranges)
        ranges[0][0] = 0.7
        ranges[0][1] = 50
        ranges.clear()
        decision = await decide(gate)

    assert decision.should_end is False
    assert decision.timeout == 0.4
    assert decision.reason == "jev_incomplete"
    assert gate.timeout == 2.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configuration,probability,timeout,reason",
    [
        ({"hold_threshold": 0.4}, 0.4, 0.8, "jev_incomplete"),
        ({"hold_threshold": 0.4}, 0.8, 2.0, "jev_long_hold"),
        ({"long_hold_threshold": 0.9}, 0.5, 0.8, "jev_incomplete"),
        ({"long_hold_threshold": 0.9}, 0.9, 2.0, "jev_long_hold"),
        ({"hold_timeout": 0.6}, 0.5, 0.6, "jev_incomplete"),
        ({"hold_timeout": 0.6}, 0.8, 2.0, "jev_long_hold"),
        ({"long_hold_timeout": 3.0}, 0.5, 0.8, "jev_incomplete"),
        ({"long_hold_timeout": 3.0}, 0.8, 3.0, "jev_long_hold"),
        ({"hold_threshold": 0.8}, 0.8, 2.0, "jev_long_hold"),
    ],
)
async def test_partial_legacy_configuration_preserves_old_defaults(configuration, probability, timeout, reason):
    def respond(request):
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": probability}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key", **configuration)
        decision = await decide(gate)

    assert decision.should_end is False
    assert decision.timeout == timeout
    assert decision.reason == reason


@pytest.mark.asyncio
async def test_request_contract_and_custom_configuration():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": 0.4}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(
            http_client=client,
            api_key="test-key",
            model="test-model",
            instructions="Wait when the speaker is still thinking.",
            hold_threshold=0.25,
            long_hold_threshold=0.4,
            hold_timeout=0.3,
            long_hold_timeout=1.2,
            request_timeout=3.0,
        )
        decision = await decide(gate, text="  確認するので待ってください。  ")

    assert decision.should_end is False
    assert decision.timeout == 1.2
    assert gate.timeout == 3.0
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-key"
    assert all(value == 3.0 for value in request.extensions["timeout"].values())
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["state"] == {
        "transcript": "確認するので待ってください。",
        "recorded_duration_seconds": 1.5,
        "silence_duration_seconds": 0.3,
    }
    assert payload["questions"]["hold"]["type"] == "noul"
    assert payload["questions"]["hold"]["instructions"] == gate.instructions
    assert set(payload["questions"]["hold"]["criteria"]) == {"true", "false"}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "", " \n\t"])
async def test_empty_transcript_passes_without_request(text):
    def unexpected_request(request):
        pytest.fail("Empty transcripts must not be sent to Jev")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key")
        decision = await decide(gate, text=text)

    assert decision.should_end is True
    assert decision.reason == "jev_no_text"
    assert decision.timeout is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"answers": {}},
        {"answers": {"hold": {}}},
        {"answers": {"hold": {"noul": 0.9}}},
        {"answers": {"hold": {"type": "score", "noul": 0.9}}},
        {"answers": {"hold": {"type": "noul", "noul": None}}},
        {"answers": {"hold": {"type": "noul", "noul": True}}},
        {"answers": {"hold": {"type": "noul", "noul": "0.9"}}},
        {"answers": {"hold": {"type": "noul", "noul": -0.1}}},
        {"answers": {"hold": {"type": "noul", "noul": 1.1}}},
        {"answers": {"hold": {"type": "noul", "noul": float("nan")}}},
        {"answers": {"hold": {"type": "noul", "noul": float("inf")}}},
        {"answers": {"hold": []}},
        [],
    ],
)
async def test_invalid_response_passes_without_hold(body):
    def respond(request):
        # Raw JSON also permits testing a provider sending non-finite numbers.
        return httpx.Response(200, content=json.dumps(body))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        decision = await decide(JevTurnEndGate(http_client=client, api_key="test-key"))

    assert decision.should_end is True
    assert decision.reason == "jev_error"
    assert decision.timeout is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["status", "invalid_json", "network", "timeout", "redirect"])
async def test_request_errors_fail_open_without_logging_sensitive_content(failure, caplog):
    private_marker = "private-provider-error-content"
    key = "test-sensitive-key"
    requests = []

    def respond(request):
        requests.append(request)
        if failure == "network":
            raise httpx.ConnectError(private_marker, request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout(private_marker, request=request)
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "https://api.typesafe.ai/redirected"})
        status = 503 if failure == "status" else 200
        return httpx.Response(status, text=f"{private_marker} {key}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        decision = await decide(JevTurnEndGate(http_client=client, api_key=key, debug=True))

    assert decision.should_end is True
    assert decision.reason == "jev_error"
    assert decision.timeout is None
    assert len(requests) == 1
    assert private_marker not in caplog.text
    assert key not in caplog.text


@pytest.mark.asyncio
async def test_total_request_deadline_cancels_stalled_transport():
    canceled = asyncio.Event()

    async def respond(request):
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key", request_timeout=0.01)
        decision = await asyncio.wait_for(decide(gate), timeout=1.0)
        assert canceled.is_set()
        assert not client.is_closed

    assert decision.should_end is True
    assert decision.reason == "jev_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configuration",
    [
        {"api_key": "   "},
        {"hold_threshold": -0.1},
        {"long_hold_threshold": 1.1},
        {"hold_threshold": 0.9, "long_hold_threshold": 0.8},
        {"hold_threshold": float("nan")},
        {"long_hold_threshold": float("inf")},
        {"hold_timeout": 0},
        {"long_hold_timeout": -1},
        {"hold_timeout": 3, "long_hold_timeout": 2},
        {"request_timeout": 0},
        {"request_timeout": float("inf")},
        {"hold_timeout": float("nan")},
        {"long_hold_timeout": float("inf")},
    ],
)
async def test_invalid_configuration_is_rejected(configuration):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ValueError):
            JevTurnEndGate(http_client=client, **{"api_key": "test-key", **configuration})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ranges",
    [
        [],
        0.6,
        "0.6,0.4",
        {0.6: 0.4},
        [None],
        [0.6],
        ["0.6,0.4"],
        [(0.6,)],
        [(0.6, 0.4, 1)],
        [("0.6", 0.4)],
        [(0.6, "0.4")],
        [(None, 0.4)],
        [(0.6, None)],
        [(True, 0.4)],
        [(0.6, True)],
        [(float("nan"), 0.4)],
        [(float("inf"), 0.4)],
        [(0.6, float("nan"))],
        [(0.6, float("inf"))],
        [(-0.1, 0.4)],
        [(1.1, 0.4)],
        [(0.6, 0)],
        [(0.6, -0.4)],
        [(0.8, 0.4), (0.6, 1)],
        [(0.6, 0.4), (0.6, 1)],
        [(0.6, 1), (0.8, 0.4)],
    ],
)
async def test_invalid_hold_ranges_are_rejected(ranges):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ValueError):
            JevTurnEndGate(http_client=client, api_key="test-key", hold_ranges=ranges)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configuration",
    [
        {"hold_threshold": 0.5},
        {"long_hold_threshold": 0.8},
        {"hold_timeout": 0.8},
        {"long_hold_timeout": 2.0},
    ],
)
async def test_range_configuration_cannot_be_mixed_with_legacy_options(configuration):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ValueError):
            JevTurnEndGate(
                http_client=client,
                api_key="test-key",
                hold_ranges=((0.6, 0.4),),
                **configuration,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("probability,timeout", [(0.6, 0.4), (0.8, 1.0), (0.9, 2.0)])
async def test_background_result_replaces_pending_hold_timeout(probability, timeout):
    async def respond(request):
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": probability}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key")
        manager = TurnEndGateManager([gate])
        session = make_session("background")
        try:
            assert await manager_decide(manager, session) is False
            assert session.turn_end_gate_hold_reasons == ["jev_pending"]
            assert session.turn_end_gate_hold_timeout == 2.0
            await manager._states[session.session_id].pending_tasks["jev"]

            assert await manager_decide(manager, session, silence_duration=0.4) is False
            assert session.turn_end_gate_hold_timeout == timeout
            assert await manager_decide(manager, session, silence_duration=0.2 + timeout) is True
            assert session.turn_end_gate_hold_active is False
        finally:
            state = manager._states.get(session.session_id)
            tasks = list(state.pending_tasks.values()) if state else []
            manager.reset_session(session.session_id, session=session)
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_timeout,expected_pending_timeout", [(0.1, 3.0), (5.0, 5.0)])
async def test_custom_ranges_pending_timeout_covers_request_and_longest_hold(request_timeout, expected_pending_timeout):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": 0.8}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(
            http_client=client,
            api_key="test-key",
            hold_ranges=((0.2, 0.15), (0.4, 0.4), (0.7, 0.8), (0.95, 3.0)),
            request_timeout=request_timeout,
        )
        manager = TurnEndGateManager([gate])
        session = make_session("custom_pending")
        task = None
        try:
            assert gate.timeout == expected_pending_timeout
            assert await manager_decide(manager, session) is False
            task = manager._states[session.session_id].pending_tasks["jev"]
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            assert session.turn_end_gate_hold_timeout == expected_pending_timeout

            release.set()
            await task
            assert await manager_decide(manager, session, silence_duration=0.4) is False
            assert session.turn_end_gate_hold_timeout == 0.8
            assert session.turn_end_gate_hold_reasons == ["jev_incomplete"]
        finally:
            manager.reset_session(session.session_id, session=session)
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        assert session.session_id not in manager._states
        assert not client.is_closed


@pytest.mark.asyncio
async def test_short_hold_already_elapsed_when_response_arrives_releases_immediately():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": 0.7}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(http_client=client, api_key="test-key")
        manager = TurnEndGateManager([gate])
        session = make_session("elapsed_before_response")
        task = None
        try:
            assert await manager_decide(manager, session) is False
            task = manager._states[session.session_id].pending_tasks["jev"]
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            # Advance reported audio silence deterministically without wall-clock sleeps.
            assert await manager_decide(manager, session, silence_duration=0.7) is False
            assert session.turn_end_gate_hold_reasons == ["jev_pending"]
            release.set()
            await task

            assert await manager_decide(manager, session, silence_duration=0.7) is True
            assert session.turn_end_gate_hold_active is False
            assert session.turn_end_gate_hold_timeout is None
            assert session.turn_end_gate_hold_reasons == []
            assert session.session_id not in manager._states
        finally:
            manager.reset_session(session.session_id, session=session)
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_reset_cancels_only_its_session_and_client_remains_usable():
    entered = {name: asyncio.Event() for name in ("reset_me", "finish_me")}
    release = asyncio.Event()
    canceled = asyncio.Event()

    async def respond(request):
        transcript = json.loads(request.content)["state"]["transcript"]
        entered[transcript].set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            canceled.set()
            raise
        return httpx.Response(200, json={"answers": {"hold": {"type": "noul", "noul": 0.1}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = JevTurnEndGate(
            http_client=client,
            api_key="test-key",
            hold_ranges=((0.2, 0.15), (0.4, 0.4), (0.7, 0.8), (0.95, 3.0)),
        )
        manager = TurnEndGateManager([gate])
        reset_session = make_session("reset_me")
        finish_session = make_session("finish_me")
        tasks = []
        try:
            assert await manager_decide(manager, reset_session) is False
            assert await manager_decide(manager, finish_session) is False
            tasks = [manager._states[s.session_id].pending_tasks["jev"] for s in (reset_session, finish_session)]
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), timeout=1.0)

            manager.reset_session(reset_session.session_id, session=reset_session)
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert canceled.is_set()
            assert reset_session.turn_end_gate_hold_active is False
            assert not tasks[1].done()

            release.set()
            await tasks[1]
            assert await manager_decide(manager, finish_session, silence_duration=0.4) is True
            assert not client.is_closed
            assert (await decide(gate, text="finish_me")).should_end is True
        finally:
            for session in (reset_session, finish_session):
                manager.reset_session(session.session_id, session=session)
            await asyncio.gather(*tasks, return_exceptions=True)
