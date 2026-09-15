"""Engine tests use an in-process HTTP transport and placeholder credentials only."""

import asyncio
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tomllib

import httpx
import pytest

from examples.nod.engine import NodDecision, NodEngine, load_profile


PROFILE = Path(__file__).resolve().parents[3] / "examples/nod/profiles/imouto_ja.toml"
CANDIDATES = [{"id": "neutral", "phrase": "うん", "description": "軽く受け止める"},
              {"id": "engaged", "phrase": "うんうん", "description": "しっかり聞く"},
              {"id": "understood", "phrase": "I see", "description": "Acknowledge understanding"}]
CANDIDATE_TOML = '''
[[candidates]]
id = "neutral"
phrase = "うん"
description = "軽く受け止める"
'''


def test_profile_sends_phrases_and_xml_rules_without_internal_ids():
    engine = NodEngine.from_profile(PROFILE)
    profile = load_profile(PROFILE)
    assert len(engine.candidates) == len(profile["candidates"]) == 9
    payload = engine.build_payload("会話データ")
    assert len(payload["messages"]) == 2
    assert payload["messages"][1] == {"role": "user", "content": "会話データ"}
    assert payload["messages"][0]["content"].startswith(profile["prompt"].strip())
    assert "<nod_assistant>" in payload["messages"][0]["content"]
    assert "<backchannel>" not in payload["messages"][0]["content"]
    for item in profile["candidates"]:
        assert f'「{item["phrase"]}」: {item["description"]}' in payload["messages"][0]["content"]
        assert item["id"] not in payload["messages"][0]["content"]
    assert "<nod_assistant/>" in payload["messages"][0]["content"]
    assert payload["max_completion_tokens"] == 32 and payload["reasoning_effort"] == "none"
    assert payload["temperature"] == 0 and payload["stream"] is False
    assert "max_tokens" not in payload and "provider" not in payload


@pytest.mark.parametrize("model,base_url,router", [
    ("gpt-4.1-mini", "https://api.openai.com/v1", False),
    ("google/gemma-4-26b-a4b-it", "https://openrouter.ai/api/v1", True),
    ("gpt-5.6-luna", "https://example.test/v1", False),
])
def test_explicit_provider_configuration(model, base_url, router):
    payload = NodEngine(CANDIDATES, "Return one phrase in a nod_assistant XML element", model=model, base_url=base_url).build_payload("input")
    assert payload["max_tokens"] == 32
    assert "reasoning_effort" not in payload and "max_completion_tokens" not in payload
    if router:
        assert payload["provider"] == {"sort": "latency", "allow_fallbacks": False}
        assert payload["reasoning"] == {"enabled": False}
    else:
        assert "provider" not in payload and "reasoning" not in payload


@pytest.mark.parametrize("content,identifier,phrase,outcome", [
    ("<nod_assistant>うん</nod_assistant>", "neutral", "うん", "candidate"),
    (" \n<nod_assistant>うんうん</nod_assistant>\n", "engaged", "うんうん", "candidate"),
    ("<nod_assistant>&#x3046;&#x3093;</nod_assistant>", "neutral", "うん", "candidate"),
    ("<nod_assistant>I see</nod_assistant>", "understood", "I see", "candidate"),
    ("<nod_assistant/>", "none", None, "none"),
    ("<nod_assistant />", "none", None, "none"),
    ("<nod_assistant></nod_assistant>", "none", None, "none"),
    ("neutral", "invalid", None, "invalid"),
    ("none", "invalid", None, "invalid"),
    ("うん", "invalid", None, "invalid"),
    ("0", "invalid", None, "invalid"),
    ("neutral or engaged", "invalid", None, "invalid"),
    ("\"engaged\"", "invalid", None, "invalid"),
    ("ENGAGED", "invalid", None, "invalid"),
    ("invented", "invalid", None, "invalid"),
    ("<nod_assistant>知らない文言</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>neutral</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>none</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>「うん」</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>うん。</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant> うん </nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>うん", "invalid", None, "invalid"),
    ("<nod_assistant>うん</nod>", "invalid", None, "invalid"),
    ("<nod>うん</nod>", "invalid", None, "invalid"),
    ("<nod_assistant phrase='うん'/>", "invalid", None, "invalid"),
    ("<nod_assistant><phrase>うん</phrase></nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>うん</nod_assistant><nod_assistant/>", "invalid", None, "invalid"),
    ("選択：<nod_assistant>うん</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>うん</nod_assistant>と返します。", "invalid", None, "invalid"),
    ("```xml\n<nod_assistant>うん</nod_assistant>\n```", "invalid", None, "invalid"),
    ("<!-- comment --><nod_assistant>うん</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>う<!-- comment -->ん</nod_assistant>", "invalid", None, "invalid"),
    ("<?xml version='1.0'?><nod_assistant>うん</nod_assistant>", "invalid", None, "invalid"),
    ("<!DOCTYPE nod_assistant [<!ENTITY x 'うん'>]><nod_assistant>&x;</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>&unknown;</nod_assistant>", "invalid", None, "invalid"),
    ("<nod_assistant>&#0;</nod_assistant>", "invalid", None, "invalid"),
    (None, "empty", None, "empty"),
    (" \n", "empty", None, "empty"),
])
@pytest.mark.asyncio
async def test_strict_output_parser_and_injected_client_ownership(content, identifier, phrase, outcome):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with NodEngine(CANDIDATES, "Return one phrase in a nod_assistant XML element", client=client,
                             api_key="placeholder-test-key", base_url="https://example.test/v1/") as engine:
            result = await engine.decide("会話データ")
        assert not client.is_closed
        assert result.id == identifier and result.phrase == phrase and result.outcome == outcome
        assert result.finish_reason == "stop" and result.elapsed_ms >= 0
        assert result.utterance_id is None
        with pytest.raises(FrozenInstanceError):
            result.id = "changed"
        assert str(requests[0].url) == "https://example.test/v1/chat/completions"
        assert requests[0].headers["Authorization"] == "Bearer placeholder-test-key"
        assert json.loads(requests[0].content) == engine.build_payload("会話データ")
        with pytest.raises(RuntimeError, match="closed"):
            await engine.decide("next")


@pytest.mark.asyncio
@pytest.mark.parametrize("profile_name", ["imouto_ja.toml", "imouto_en.toml"])
async def test_all_profile_phrases_map_back_to_the_existing_internal_ids(profile_name):
    profile = PROFILE.parent / profile_name
    candidates = load_profile(profile)["candidates"]
    replies = iter(candidates)

    def respond(request):
        candidate = next(replies)
        return httpx.Response(200, json={"choices": [{"message": {
            "content": f'<nod_assistant>{candidate["phrase"]}</nod_assistant>'}, "finish_reason": "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with NodEngine.from_profile(profile, client=client) as engine:
            for candidate in candidates:
                result = await engine.decide("会話データ")
                assert (result.id, result.phrase, result.outcome) == (
                    candidate["id"], candidate["phrase"], "candidate")


@pytest.mark.asyncio
async def test_lazy_owned_client_is_reused_and_closed_exactly_once(monkeypatch):
    created = []

    class Client:
        def __init__(self, **options):
            self.closed = 0
            created.append(self)

        async def post(self, url, **kwargs):
            return httpx.Response(200, request=httpx.Request("POST", url), json={"choices": [
                {"message": {"content": "<nod_assistant/>"}, "finish_reason": "private provider message"}]})

        async def aclose(self):
            self.closed += 1

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    engine = NodEngine(CANDIDATES, "Return one phrase in a nod_assistant XML element")
    assert created == []
    engine.build_payload("input")
    assert created == []
    async with engine:
        assert (await engine.decide("input")).finish_reason == "unknown"
        await engine.decide("next")
    await engine.close()
    assert len(created) == 1 and created[0].closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "malformed", "cancel"])
async def test_failures_propagate_and_do_not_emit_diagnostics(failure, caplog):
    entered = asyncio.Event()

    async def respond(request):
        entered.set()
        if failure == "cancel":
            await asyncio.Event().wait()
        return httpx.Response(401 if failure == "http" else 200,
                              json={"private": "private provider diagnostics"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with NodEngine(CANDIDATES, "Return one phrase in a nod_assistant XML element", client=client) as engine:
            if failure == "cancel":
                task = asyncio.create_task(engine.decide("input"))
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(httpx.HTTPStatusError if failure == "http" else ValueError):
                    await engine.decide("input")
    assert "private provider diagnostics" not in caplog.text


@pytest.mark.parametrize("invalid", [
    [], [{"id": "none", "phrase": "うん", "description": "聞く"}],
    [{"id": "Bad ID", "phrase": "うん", "description": "聞く"}],
    [{"id": "ok", "phrase": " ", "description": "聞く"}],
    [{"id": "ok", "phrase": "うん", "description": ""}],
    [CANDIDATES[0], CANDIDATES[0]],
    [CANDIDATES[0], dict(CANDIDATES[1], phrase="うん")],
])
def test_invalid_candidate_configurations_fail_before_network(invalid):
    with pytest.raises(ValueError):
        NodEngine(invalid, "Return one phrase in a nod_assistant XML element")


@pytest.mark.parametrize("options", [{"messages": []}, {"model": "other"}, {"stream": True},
                                     {"max_tokens": 1, "max_completion_tokens": 1}, []])
def test_request_options_cannot_replace_protocol_fields(options):
    with pytest.raises(ValueError):
        NodEngine(CANDIDATES, "Return one phrase in a nod_assistant XML element", request_options=options)


def test_request_options_and_candidates_are_copied_and_customizable():
    candidates = [dict(item) for item in CANDIDATES]
    options = {"temperature": 0.1, "max_completion_tokens": 32, "metadata": {"test": "local"}}
    engine = NodEngine(candidates, "Return one phrase in a nod_assistant XML element", request_options=options)
    candidates[0]["phrase"] = "wrong"
    engine.candidates[0]["phrase"] = "also wrong"
    options["metadata"]["test"] = "changed"
    payload = engine.build_payload("input")
    assert "「うん」: 軽く受け止める" in payload["messages"][0]["content"]
    assert "neutral" not in payload["messages"][0]["content"]
    assert payload["temperature"] == 0.1 and payload["max_completion_tokens"] == 32
    assert payload["metadata"] == {"test": "local"}
    payload["metadata"]["test"] = "mutated payload"
    assert engine.build_payload("input")["metadata"] == {"test": "local"}


def test_standalone_profile_loads_multiline_prompt_from_another_working_directory(tmp_path, monkeypatch):
    profile = tmp_path / "profiles/listener.toml"
    profile.parent.mkdir()
    profile.write_text("prompt = '''\n判定方針：\n途中は「うん」。\n'''\n" + CANDIDATE_TOML,
                       encoding="utf-8")
    working_directory = tmp_path / "run"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    loaded = load_profile(profile)
    assert loaded["prompt"] == "判定方針：\n途中は「うん」。\n"
    assert loaded["candidates"] == [CANDIDATES[0]]
    engine = NodEngine.from_profile(profile)
    assert engine.prompt == "判定方針：\n途中は「うん」。"
    assert engine.build_payload("会話データ")["messages"][0] == {
        "role": "system",
        "content": "判定方針：\n途中は「うん」。\n相槌の候補文言：\n「うん」: 軽く受け止める",
    }


@pytest.mark.parametrize("prompt_field", [
    "", 'prompt = ""', 'prompt = "   "', "prompt = 42", "prompt = false", "prompt = []",
])
def test_missing_empty_or_nonstring_profile_prompt_is_rejected(tmp_path, prompt_field):
    profile = tmp_path / "profile.toml"
    profile.write_text(prompt_field + "\n" + CANDIDATE_TOML, encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_profile(profile)
    with pytest.raises(ValueError, match="prompt"):
        NodEngine.from_profile(profile)


@pytest.mark.parametrize("candidate_field", [
    "", "candidates = []", 'candidates = "うん"', 'candidates = ["うん"]',
    'candidates = [{id = "neutral", phrase = "うん"}]',
])
def test_missing_empty_or_invalid_profile_candidates_are_rejected(tmp_path, candidate_field):
    profile = tmp_path / "profile.toml"
    profile.write_text('prompt = "判断方針"\n' + candidate_field, encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(profile)


def test_malformed_toml_profile_is_rejected(tmp_path):
    profile = tmp_path / "profile.toml"
    profile.write_text('prompt = "unterminated\n' + CANDIDATE_TOML, encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        NodEngine.from_profile(profile)
