"""Exercise the public engine/session API with an in-process HTTP transport."""

import json
from pathlib import Path

import httpx
import pytest

from examples.nod import NodEngine, NodSession


@pytest.mark.asyncio
async def test_public_api_sends_plain_history_and_records_only_delivered_nods():
    requests, delivered = [], []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = "<nod_assistant>そっか</nod_assistant>" if len(requests) == 1 else "<nod_assistant/>"
        return httpx.Response(200, json={"choices": [
            {"message": {"content": content}, "finish_reason": "stop"}]})

    async def emit(decision):
        assert session.is_current(decision)
        delivered.append((decision.utterance_id, decision.id, decision.phrase))
        return True

    profile = Path(__file__).resolve().parents[3] / "examples/nod/profiles/imouto_ja.toml"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with NodEngine.from_profile(profile, client=client) as engine:
            async with NodSession(engine, emit) as session:
                session.update_assistant("a0", "実際に行ってみたいね。")
                session.start_input("u1")
                session.update_user("u1", "分かんないけどね、")
                assert (await session.on_pause("u1")).outcome == "sent"
                session.end_input("u1")
                session.update_user("u1", "分かんないけどね、行ってみないと。")
                session.update_assistant("a1", "見てから決められるといいね。")
                session.start_input("u2")
                session.update_user("u2", "まあね")
                assert (await session.on_pause("u2")).outcome == "none"
        assert not client.is_closed

    assert delivered == [("u1", "understood", "そっか")]
    assert len(requests) == 2
    assert all([m["role"] for m in p["messages"]] == ["system", "user"] for p in requests)
    assert requests[0]["messages"][0] == requests[1]["messages"][0]
    assert "分かんないけどね、<nod_assistant>そっか</nod_assistant>行ってみないと。" in requests[1]["messages"][1]["content"]
    assert "【今回のユーザー発言】\nユーザー：まあね" in requests[1]["messages"][1]["content"]
    assert all(p["reasoning_effort"] == "none" and p["max_completion_tokens"] == 32 for p in requests)


@pytest.mark.asyncio
async def test_invalid_xml_does_not_emit_or_enter_history_or_start_cooldown():
    replies = iter(["<nod_assistant>うん", "<nod_assistant>うん</nod_assistant>という相槌です。",
                    "<nod_assistant/>", "<nod_assistant>うん</nod_assistant>"])
    delivered = []

    def respond(request):
        return httpx.Response(200, json={"choices": [
            {"message": {"content": next(replies)}, "finish_reason": "stop"}]})

    async def emit(decision):
        delivered.append(decision.phrase)
        return True

    profile = Path(__file__).resolve().parents[3] / "examples/nod/profiles/imouto_ja.toml"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with NodEngine.from_profile(profile, client=client) as engine:
            async with NodSession(engine, emit) as session:
                session.start_input("u1")
                session.update_user("u1", "公園に行ったんだ")
                for expected in ("invalid", "invalid", "none"):
                    assert (await session.on_pause("u1")).outcome == expected
                    assert not delivered
                    assert "<nod_assistant>" not in session.build_input("u1")
                assert (await session.on_pause("u1")).outcome == "sent"
                assert delivered == ["うん"]
                assert "公園に行ったんだ<nod_assistant>うん</nod_assistant>" in session.build_input("u1")
