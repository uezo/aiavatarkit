"""Preview nod inputs locally, or explicitly run a paid model regression.

Run from the repository root with ``python -m examples.nod.evaluate``.
The default preview never creates an HTTP client or loads credentials.
"""

import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from .engine import NodEngine, load_profile
from .input import plain_input


ROOT = Path(__file__).resolve().parent


def _nods(value):
    if not isinstance(value, list):
        raise ValueError("nods must be a list")
    for nod in value:
        if not isinstance(nod, dict) or any(
            not isinstance(nod.get(key), str) for key in ("phrase", "acknowledged_text")
        ):
            raise ValueError("each nod requires phrase and acknowledged_text strings")
        length = nod.get("text_length", len(nod["acknowledged_text"]))
        if type(length) is not int or length < len(nod["acknowledged_text"]):
            raise ValueError("nod text_length must be an integer at least as long as acknowledged_text")


def load_cases(path, candidates):
    """Validate every input and expected ID before any provider request."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    choices = {candidate["id"] for candidate in candidates} | {"none"}
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("case IDs must be nonempty, unique strings")
        seen.add(case_id)
        if not isinstance(case.get("text"), str):
            raise ValueError("case text must be a string")
        if "input" in case and (not isinstance(case["input"], str) or not case["input"].strip()):
            raise ValueError("recorded input must be a nonempty string")
        expected = case.get("expected")
        if not isinstance(expected, list) or not expected or any(
            not isinstance(choice, str) or choice not in choices for choice in expected
        ):
            raise ValueError("expected must be a nonempty list of profile IDs or none")
        history = case.get("history", [])
        if not isinstance(history, list):
            raise ValueError("history must be a list")
        for message in history:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                raise ValueError("history roles must be user or assistant")
            if not isinstance(message.get("content"), str):
                raise ValueError("history content must be a string")
            _nods(message.get("nods", []))
            if message["role"] == "assistant" and message.get("nods"):
                raise ValueError("nod annotations belong to user utterances")
        _nods(case.get("nods", []))
    return cases


def case_input(case, history_limit=10):
    if "input" in case:
        return case["input"]  # Replay the exact logged message, including its nod history.
    return plain_input(case.get("history", []), case["text"], case.get("nods", []),
                       history_limit=history_limit)


def _validate_limits(repeat, timeout, budget_ms, history_limit):
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0
           for value in (timeout, budget_ms)):
        raise ValueError("timeout and budget-ms must be positive finite numbers")
    if type(history_limit) is not int or not 1 <= history_limit <= 10:
        raise ValueError("history-limit must be an integer from 1 to 10")


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return round(values[left] + (values[right] - values[left]) * (position - left), 2)


def summarize(rows):
    # Timeout/error observations are censored; do not mix them into API latency.
    completed = [row for row in rows if row["outcome"] not in ("timeout", "error", "skipped_empty")]
    decisions = [row for row in rows if row["choice"] is not None]
    counts = Counter(row["choice"] for row in decisions)
    return {
        "total": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "timely_passed": sum(row["timely_passed"] for row in rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "p50_ms": _percentile([row["elapsed_ms"] for row in completed], 0.5),
        "p95_ms": _percentile([row["elapsed_ms"] for row in completed], 0.95),
        "distribution": {
            choice: {"count": count, "percent": round(100 * count / len(decisions), 2)}
            for choice, count in sorted(counts.items())
        },
    }


async def run_evaluation(cases, engine, *, repeat=3, timeout=5.0, budget_ms=1500.0,
                         history_limit=10):
    """Evaluate sequentially; preserve exact user inputs and sanitized failures."""
    _validate_limits(repeat, timeout, budget_ms, history_limit)
    inputs = [(case, case_input(case, history_limit)) for case in cases]
    rows = []
    for run in range(1, repeat + 1):
        for case, user_input in inputs:
            row = {
                "case_id": case["id"], "run": run, "input": user_input,
                "expected": case["expected"], "choice": None, "phrase": None,
                "finish_reason": None, "elapsed_ms": 0.0,
            }
            if not case["text"].strip():
                row.update(outcome="skipped_empty", choice="none")
            else:
                started = time.monotonic()
                try:
                    decision = await asyncio.wait_for(engine.decide(user_input), timeout=timeout)
                    row.update(outcome=decision.outcome, phrase=decision.phrase,
                               finish_reason=decision.finish_reason)
                    if decision.outcome in ("candidate", "none"):
                        row["choice"] = decision.id or "none"
                except TimeoutError:
                    row["outcome"] = "timeout"
                except Exception as exc:
                    # Provider error messages/bodies may contain request credentials.
                    row.update(outcome="error", error_type=type(exc).__name__)
                row["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
            row["passed"] = row["choice"] in row["expected"]
            row["within_budget"] = (
                row["outcome"] not in ("timeout", "error") and row["elapsed_ms"] <= budget_ms
            )
            row["timely_passed"] = row["passed"] and row["within_budget"]
            rows.append(row)
    return {"summary": summarize(rows), "rows": rows}


def _base_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or any(
        (parsed.username, parsed.password, parsed.query, parsed.fragment)
    ):
        raise ValueError("base-url must be an HTTP(S) endpoint without credentials, query or fragment")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("remote API endpoints must use HTTPS")
    return value.rstrip("/")


def _api_key(base_url):
    key = os.environ.get("NOD_API_KEY", "").strip()
    if key:
        return key
    variable = {"api.openai.com": "OPENAI_API_KEY", "openrouter.ai": "OPENROUTER_API_KEY"}.get(
        urlsplit(base_url).hostname
    )
    key = os.environ.get(variable, "").strip() if variable else ""
    if not key:
        raise ValueError("set NOD_API_KEY or the API key for the selected endpoint")
    return key


def _output_file(path):
    # Exclusive creation both validates writability before requests and protects
    # an existing report or source file against accidental replacement.
    return Path(path).open("x", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "cases/sample.json",
                        help="Evaluation cases JSON (default: bundled cases/sample.json)")
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles/imouto_ja.toml",
                        help="TOML profile containing prompt and candidates (default: profiles/imouto_ja.toml)")
    parser.add_argument("--model", default=os.environ.get("NOD_MODEL") or "gpt-5.6-luna")
    parser.add_argument("--base-url", default=os.environ.get("NOD_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5.0, help="Hard per-request deadline in seconds")
    parser.add_argument("--budget-ms", type=float, default=1500.0, help="Report timely success against this budget")
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="Make paid requests (requires --output)")
    parser.add_argument("--output", type=Path, help="New JSON report path; existing files are never overwritten")
    parser.add_argument("--require-budget", action="store_true", help="Exit nonzero for expected but late choices")
    args = parser.parse_args(argv)

    try:
        _validate_limits(args.repeat, args.timeout, args.budget_ms, args.history_limit)
        base_url = _base_url(args.base_url)
        if not args.model.strip():
            raise ValueError("model must be nonempty")
        if args.run and not args.output:
            raise ValueError("--run requires --output to save the exact inputs and results")
        profile = load_profile(args.profile)
        prompt = profile["prompt"].strip()
        cases = load_cases(args.cases, profile["candidates"])
        inputs = [{"case_id": case["id"], "input": case_input(case, args.history_limit),
                   "expected": case["expected"]} for case in cases]
        system_prompt = NodEngine(profile["candidates"], prompt, model=args.model,
                                  base_url=base_url).build_payload("")["messages"][0]["content"]
        key = _api_key(base_url) if args.run else None
        output = _output_file(args.output) if args.output else None
    except (OSError, ValueError, KeyError, TypeError):
        # Do not echo endpoint userinfo, malformed input values or filesystem data.
        parser.error("invalid configuration: check cases/profile/prompt, limits, output path and endpoint API key")

    config = {
        "model": args.model, "base_url": base_url, "repeat": args.repeat,
        "timeout": args.timeout, "budget_ms": args.budget_ms, "history_limit": args.history_limit,
        "prompt_sha256": sha256(prompt.encode()).hexdigest(),
        "candidate_ids": [candidate["id"] for candidate in profile["candidates"]],
        # One report-level snapshot makes later prompt/candidate edits reproducible.
        # Individual result rows and the live summary never include this policy.
        "system_prompt": system_prompt,
        "cases_sha256": sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
    }

    async def run():
        import httpx
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            engine = NodEngine(profile["candidates"], prompt, api_key=key, client=client,
                               model=args.model, base_url=base_url)
            try:
                return await run_evaluation(cases, engine, repeat=args.repeat, timeout=args.timeout,
                                            budget_ms=args.budget_ms, history_limit=args.history_limit)
            finally:
                await engine.close()

    try:
        if args.run:
            report = asyncio.run(run())
            report["config"] = config
        else:
            report = {"mode": "preview", "config": config, "inputs": inputs}
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if output:
            output.write(rendered)
            print(json.dumps(report.get("summary", {"mode": "preview", "cases": len(cases)}), ensure_ascii=False))
        else:
            print(rendered, end="")
    finally:
        if output:
            output.close()
    if not args.run:
        return 0
    return int(any(not row["timely_passed" if args.require_budget else "passed"] for row in report["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
