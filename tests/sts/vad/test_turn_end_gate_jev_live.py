"""Opt-in, billable TypeSafe API smoke test using synthetic Japanese transcripts.

Requires TYPESAFE_RUN_LIVE_TESTS=1 and TYPESAFE_API_KEY in the process environment.
No local configuration is read. Run with -s to see hold probabilities and timings.
"""

import os
import statistics
import time

import httpx
import pytest

from aiavatar.sts.vad.turn_end_gates.jev import JevTurnEndGate


@pytest.mark.asyncio
async def test_jev_turn_end_gate_live_japanese_turns():
    if os.environ.get("TYPESAFE_RUN_LIVE_TESTS") != "1":
        pytest.skip("Set TYPESAFE_RUN_LIVE_TESTS=1 to authorize billable TypeSafe requests")
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        pytest.skip("TYPESAFE_API_KEY is required")

    cases = [
        ("finished", "今日はこれで終わりです。ありがとうございました。", True),
        ("question", "東京の明日の天気を教えてください。", True),
        ("short_answer", "はい、それでお願いします。", True),
        ("hesitation", "えっと、あの、その、うーん", False),
        ("unfinished", "私が一番伝えたかったことは", False),
        ("explicit_wait", "少し考えるので、まだ返事をせずに待っていてください。", False),
        ("checking", "ちょっと待って、今資料を確認しています。", False),
        ("unfinished_request", "それで、次にお願いしたいのが", False),
        ("question_about_waiting", "電車の待ち時間は何分ですか？", True),
        ("polite_request", "予約を変更したいんですけど。", True),
    ]
    elapsed_times = []
    failures = []
    async with httpx.AsyncClient() as http_client:
        # Allow cold connection setup in this smoke test. Production defaults
        # to a 1-second total deadline; the measured latency is printed below.
        gate = JevTurnEndGate(
            http_client=http_client,
            api_key=api_key,
            request_timeout=5.0,
        )
        for name, transcript, expected_end in cases:
            started_at = time.perf_counter()
            decision = await gate.should_end_turn(
                audio=b"",
                sample_rate=16000,
                channels=1,
                recorded_duration=2.0,
                silence_duration=0.5,
                session_id=f"jev_live_{name}",
                text=transcript,
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            elapsed_times.append(elapsed_ms)
            if decision.confidence is None:
                failures.append(f"{name}: no API probability ({decision.reason})")
                print(f"{name}: {decision.reason}, elapsed={elapsed_ms:.1f}ms")
                continue
            hold_probability = 1.0 - decision.confidence
            print(
                f"{name}: hold={hold_probability:.4f}, "
                f"decision={'END' if decision.should_end else 'WAIT'}, "
                f"hold_timeout={decision.timeout}, elapsed={elapsed_ms:.1f}ms"
            )
            if decision.should_end != expected_end:
                failures.append(f"{name}: expected {'END' if expected_end else 'WAIT'}, hold={hold_probability:.4f}")

    print(
        f"API timing: first={elapsed_times[0]:.1f}ms, "
        f"subsequent_median={statistics.median(elapsed_times[1:]):.1f}ms, "
        f"max={max(elapsed_times):.1f}ms ({len(cases)} requests)"
    )
    assert not failures, "; ".join(failures)
