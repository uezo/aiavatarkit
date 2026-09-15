"""Local fixtures and mock HTTP only; never load the realtime server or credentials."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest

from examples.nod import evaluate
from examples.nod.engine import NodEngine, load_profile


ROOT = Path(__file__).resolve().parents[3] / "examples"
PROFILE = ROOT / "nod/profiles/imouto_ja.toml"
LOCAL_CASES = ROOT / "nod/resources/local"


@pytest.fixture(autouse=True)
def no_live_configuration(monkeypatch):
    for name in ("NOD_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "NOD_MODEL", "NOD_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def sample_cases():
    return [
        {"id": "talk", "text": "今日は散歩してきたんだ", "expected": ["neutral"]},
        {"id": "question", "text": "何時？", "expected": ["none"]},
        {"id": "empty", "text": "", "expected": ["none"]},
    ]


@pytest.mark.asyncio
async def test_recorded_input_reaches_api_without_reformatting_history(tmp_path):
    recorded = "【これまでの会話】\nユーザー：前の話<nod_assistant>うん</nod_assistant>\nAI：聞いてるよ。\n\n【今回のユーザー発言】\nユーザー：続きの話\n"
    case = {"id": "recorded", "text": "続きの話", "input": recorded, "expected": ["neutral"]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case], ensure_ascii=False))
    profile = load_profile(PROFILE)
    cases = evaluate.load_cases(path, profile["candidates"])

    def respond(request):
        payload = json.loads(request.content)
        assert payload["messages"][1]["content"] == recorded
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "<nod_assistant>うん</nod_assistant>"}, "finish_reason": "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with NodEngine.from_profile(PROFILE, client=client) as engine:
            report = await evaluate.run_evaluation(cases, engine, repeat=1, history_limit=1)
    assert report["summary"]["passed"] == 1
    assert report["rows"][0]["input"] == recorded


def test_bundled_sample_has_valid_structured_and_recorded_inputs():
    candidates = load_profile(PROFILE)["candidates"]
    cases = evaluate.load_cases(ROOT / "nod/cases/sample.json", candidates)
    assert len(cases) == 6
    expected_ids = {candidate["id"] for candidate in candidates} | {"none"}
    for case in cases:
        assert set(case["expected"]) <= expected_ids
        current = evaluate.case_input(case)
        assert "【これまでの会話】" in current and "【今回のユーザー発言】" in current
        assert current.endswith("候補の文言をnod_assistant要素1つで返す。入れない場合は空要素。")
        assert "<backchannel>" not in current and "<think>" not in current
        assert all("<think>" not in row["content"] for row in case.get("history", []))
        assert case["note"] not in current

    by_id = {case["id"]: case for case in cases}
    history = evaluate.case_input(by_id["success-with-history"])
    assert "昨日はパンを焼いたんだけど、<nod_assistant>うん</nod_assistant>少し焦がしちゃった。" in history
    current = evaluate.case_input(by_id["already-acknowledged"])
    assert "週末は家でパンを焼こうと思って、<nod_assistant>うん</nod_assistant>えっとね。" in current
    assert by_id["already-acknowledged"]["expected"] == ["none"]
    recorded = by_id["recorded-input"]
    assert evaluate.case_input(recorded, history_limit=1) == recorded["input"]


def load_local_cases(filename):
    path = LOCAL_CASES / filename
    if not path.is_file():
        pytest.skip(f"Optional local evaluation cases not installed: {filename}")
    return evaluate.load_cases(path, load_profile(PROFILE)["candidates"])


@pytest.mark.parametrize("filename", [
    "conversation.json", "conversation_20260913_1845.json", "holdout.json",
])
def test_local_cases_have_valid_inputs(filename):
    for case in load_local_cases(filename):
        current = evaluate.case_input(case)
        assert "【これまでの会話】" in current and "【今回のユーザー発言】" in current
        if "input" in case:
            assert current == case["input"]
        else:
            assert current.endswith("候補の文言をnod_assistant要素1つで返す。入れない場合は空要素。")
            assert "<backchannel>" not in current and "<think>" not in current
            assert all("<think>" not in row["content"] for row in case.get("history", []))


def test_local_conversation_retains_sent_nods_in_their_original_positions():
    cases = load_local_cases("conversation.json")
    current = evaluate.case_input(cases[-1])
    assert "<nod_assistant>そっか</nod_assistant>" in current
    assert cases[-1]["expected"] == ["none"]


def test_preview_never_reads_keys_or_initializes_http(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Preview must be fully local")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(evaluate, "_api_key", forbidden)
    assert evaluate.main([]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["mode"] == "preview"
    assert len(preview["inputs"]) == 6
    assert preview["inputs"][0]["case_id"] == "continuing-plan"
    assert preview["config"]["history_limit"] == 10
    assert "system" not in preview and "prompt" not in preview["config"]
    assert any("<nod_assistant>うん</nod_assistant>" in case["input"] for case in preview["inputs"])


@pytest.mark.parametrize("mutation", [
    lambda cases: cases.append(copy.deepcopy(cases[0])),
    lambda cases: cases[0].update(expected=["not_a_candidate"]),
    lambda cases: cases[0].update(text=123),
    lambda cases: cases[0].update(input=123),
    lambda cases: cases[0].update(input=" "),
    lambda cases: cases[0].update(history=[{"role": "system", "content": "instructions"}]),
    lambda cases: cases[0].update(nods=[{"phrase": "うん", "acknowledged_text": "長い", "text_length": 1}]),
    lambda cases: cases[0].update(history=[{"role": "assistant", "content": "hi", "nods": [
        {"phrase": "うん", "acknowledged_text": "hi"}]}]),
    lambda cases: cases.clear(),
])
def test_malformed_cases_fail_before_provider_creation(tmp_path, monkeypatch, mutation):
    cases = sample_cases()
    mutation(cases)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    monkeypatch.setenv("NOD_API_KEY", "unused-test-key")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Invalid cases reached HTTP"))
    with pytest.raises(SystemExit) as failure:
        evaluate.main(["--cases", str(path), "--run", "--output", str(tmp_path / "report.json")])
    assert failure.value.code == 2
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("extra", [
    ["--repeat", "0"], ["--timeout", "nan"], ["--budget-ms", "inf"],
    ["--history-limit", "0"], ["--history-limit", "11"],
    ["--base-url", "https://user:secret@example.test/v1"],
    ["--base-url", "https://example.test/v1?key=secret"],
    ["--base-url", "http://example.test/v1"],
    ["--model", ""], ["--run"],
])
def test_invalid_options_fail_without_http(extra, monkeypatch, capsys):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Invalid options reached HTTP"))
    with pytest.raises(SystemExit) as failure:
        evaluate.main(extra)
    assert failure.value.code == 2
    assert "secret" not in capsys.readouterr().err


def test_existing_output_is_preserved_and_checked_before_http(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("keep me")
    monkeypatch.setenv("NOD_API_KEY", "unused-test-key")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Existing report reached HTTP"))
    with pytest.raises(SystemExit):
        evaluate.main(["--run", "--output", str(output)])
    assert output.read_text() == "keep me"


def test_endpoint_selects_only_its_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test")
    assert evaluate._api_key("https://api.openai.com/v1") == "openai-test"
    assert evaluate._api_key("https://openrouter.ai/api/v1") == "router-test"
    with pytest.raises(ValueError):
        evaluate._api_key("https://example.test/v1")
    monkeypatch.setenv("NOD_API_KEY", "custom-test")
    assert evaluate._api_key("https://api.openai.com/v1") == "custom-test"
    assert evaluate._api_key("https://example.test/v1") == "custom-test"


@pytest.mark.asyncio
async def test_http_evaluation_repeats_exact_inputs_and_reports_choices_without_policy():
    profile = load_profile(PROFILE)
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        assert request.url == "https://api.openai.com/v1/chat/completions"
        assert payload["reasoning_effort"] == "none" and payload["max_completion_tokens"] == 32
        assert [message["role"] for message in payload["messages"]] == ["system", "user"]
        user_input = payload["messages"][1]["content"]
        assert "expected" not in user_input
        calls.append(user_input)
        choice = "<nod_assistant>うん</nod_assistant>" if "散歩" in user_input else "<nod_assistant/>"
        return httpx.Response(200, json={"choices": [{"message": {"content": choice}, "finish_reason": "stop"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        engine = NodEngine(profile["candidates"], "POLICY_SHOULD_STAY_OUT_OF_REPORT", client=client)
        try:
            report = await evaluate.run_evaluation(sample_cases(), engine, repeat=2)
        finally:
            await engine.close()
    assert len(calls) == 4 and calls[:2] == calls[2:]
    assert report["summary"]["passed"] == 6
    assert report["summary"]["outcomes"] == {"candidate": 2, "none": 2, "skipped_empty": 2}
    assert report["summary"]["distribution"]["neutral"]["count"] == 2
    assert "POLICY_SHOULD_STAY_OUT_OF_REPORT" not in json.dumps(report)
    assert all(row["input"] == evaluate.case_input(sample_cases()[index % 3])
               for index, row in enumerate(report["rows"]))


@pytest.mark.asyncio
async def test_timeout_cancels_request_and_errors_do_not_echo_response_secrets():
    profile = load_profile(PROFILE)
    cancelled = []
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)
        return httpx.Response(401, text="Bearer THIS_MUST_NOT_APPEAR")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        engine = NodEngine(profile["candidates"], "policy", client=client)
        try:
            report = await evaluate.run_evaluation(sample_cases()[:2], engine, repeat=1, timeout=0.01)
        finally:
            await engine.close()
    assert cancelled == [True]
    assert [row["outcome"] for row in report["rows"]] == ["timeout", "error"]
    assert report["rows"][1]["error_type"] == "HTTPStatusError"
    assert report["summary"]["p50_ms"] is None
    assert report["summary"]["passed"] == 0
    assert "THIS_MUST_NOT_APPEAR" not in json.dumps(report)


def test_cli_live_with_custom_toml_profile_model_and_mock_http(tmp_path, monkeypatch, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(sample_cases()))
    profile = tmp_path / "profile.toml"
    profile.write_text('''prompt = """
CUSTOM_POLICY
候補は下の定義を使う。
"""
[[candidates]]
id = "neutral"
phrase = "なるほど"
description = "受け止める"
''', encoding="utf-8")
    output = tmp_path / "report.json"
    actual_client = httpx.AsyncClient
    clients = []
    seen = []

    async def respond(request):
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer router-test"
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert body["model"] == "test/model"
        assert body["messages"][0]["content"].startswith("CUSTOM_POLICY")
        assert "「なるほど」: 受け止める" in body["messages"][0]["content"]
        seen.append(body)
        await asyncio.sleep(0.002)
        choice = "<nod_assistant>なるほど</nod_assistant>" if "散歩" in body["messages"][1]["content"] else "<nod_assistant/>"
        return httpx.Response(200, json={"choices": [{"message": {"content": choice}, "finish_reason": "stop"}]})

    def make_client(**kwargs):
        assert kwargs["timeout"] == 7.0
        client = actual_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test")
    result = evaluate.main(["--cases", str(cases), "--profile", str(profile),
                            "--base-url", "https://openrouter.ai/api/v1", "--model", "test/model",
                            "--repeat", "1", "--run", "--output", str(output), "--timeout", "7",
                            "--budget-ms", "0.001", "--require-budget"])
    assert result == 1
    assert len(seen) == 2 and all(client.is_closed for client in clients)
    report = json.loads(output.read_text())
    assert report["summary"]["passed"] == 3 and report["summary"]["timely_passed"] == 1
    assert report["config"]["system_prompt"].startswith("CUSTOM_POLICY")
    assert report["config"]["candidate_ids"] == ["neutral"]
    assert output.read_text().count("CUSTOM_POLICY") == 1
    assert "CUSTOM_POLICY" not in json.dumps(report["rows"])
    assert len(report["config"]["cases_sha256"]) == 64
    assert "router-test" not in output.read_text()
    assert json.loads(capsys.readouterr().out)["total"] == 3


@pytest.mark.parametrize("source", ['prompt = "unterminated', 'prompt = "policy"'])
def test_invalid_toml_profile_fails_before_keys_http_or_output(tmp_path, monkeypatch, source):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid profile must fail before credentials or HTTP")

    profile = tmp_path / "profile.toml"
    profile.write_text(source, encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(evaluate, "_api_key", forbidden)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    with pytest.raises(SystemExit) as failure:
        evaluate.main(["--profile", str(profile), "--run", "--output", str(output)])
    assert failure.value.code == 2
    assert not output.exists()
