"""Billable Jev semantic tests using synthetic Japanese conversations.

Runs whenever TYPESAFE_API_KEY is available in the environment or pytest.ini's
[pytest] env block. No additional opt-in flag is required. Use -s to see scores.
"""

import configparser
import os
from pathlib import Path
import time

import httpx
import pytest

from aiavatar.sts.vad.turn_taking_gates.jev_wakeword import (
    DEFAULT_INSTRUCTIONS,
    JevWakewordGate,
)


@pytest.fixture
def create_live_gate():
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        config = configparser.ConfigParser(interpolation=None)
        config.read(Path(__file__).resolve().parents[4] / "pytest.ini")
        for line in config.get("pytest", "env", fallback="").splitlines():
            name, separator, value = line.strip().partition("=")
            if separator and name.strip() == "TYPESAFE_API_KEY":
                api_key = value.strip()
                break
    if not api_key:
        pytest.skip("TYPESAFE_API_KEY is required in the environment or pytest.ini")

    # Keep the credential out of pytest's displayed test arguments on failure.
    def create(http_client, **kwargs):
        return JevWakewordGate(http_client=http_client, api_key=api_key, **kwargs)

    return create


@pytest.mark.asyncio
async def test_jev_wakeword_live_japanese_conversations(create_live_gate):
    test_history = [
        {"role": "user", "content": "音声認識のテストを手伝ってほしい。"},
        {"role": "assistant", "content": "一つ目は短い挨拶、二つ目は長い質問を試そう。"},
        {"role": "user", "content": "一つ目は終わったので、いったん休憩するね。"},
        {"role": "assistant", "content": "わかった。戻ったら、どこから再開するか教えてね。"},
    ]
    food_history = [
        {"role": "user", "content": "今日のお昼は冷たいおそばにしようかな。"},
        {"role": "assistant", "content": "揚げ物も添えるといいね。"},
        {"role": "user", "content": "何を添えるのがおすすめ？"},
        {"role": "assistant", "content": "海老天か舞茸天はどう？どっちが好き？"},
    ]
    cases = [
        ("name_variant", "くろはちゃん、今日の天気を教えて。", [], True),
        ("sister_address", "ねえ、いもうとちゃん。ちょっと相談に乗ってくれる？", [], True),
        ("call_and_request", "クロハ、こんにちは。明日の予定を一緒に整理して。", [], True),
        ("resume_from_history", "二つ目からお願い。", test_history, True),
        ("same_text_without_history", "二つ目からお願い。", [], False),
        ("alternative_answer", "コロッケなんかもよくない？", food_history, True),
        ("unaddressed_attention", "ねえねえ。", food_history, False),
        ("topic_word_only", "おそば。", food_history, False),
        ("addressed_to_someone_else", "お母さん、そこのお茶を取って。", food_history, False),
        ("quoted_name", "資料の例文に「クロハ、こんにちは」と書いてある。", [], False),
    ]
    histories = {name: history for name, _, history, _ in cases}

    async def get_conversation_history(session_id):
        return histories[session_id]

    failures = []
    async with httpx.AsyncClient() as http_client:
        gate = create_live_gate(
            http_client=http_client,
            # Exercise Jev even when a name appears; literal matching is tested locally.
            wakewords=[],
            get_conversation_history=get_conversation_history,
            instructions=DEFAULT_INSTRUCTIONS + (
                " このAIの名前はクロハで、ユーザーの妹として会話します。"
                "ユーザーはこのAIを「クロハ」「くろは」「妹ちゃん」「いもうと」と呼びます。"
            ),
            # Allow cold connection setup; the production default remains 1 second.
            request_timeout=5.0,
        )
        for name, transcript, _, expected_wake in cases:
            started_at = time.perf_counter()
            decision = await gate.should_take_turn(transcript, None, session_id=name)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            if decision.probability is None:
                # A failed API request must never count as a correct rejection.
                failures.append(f"{name}: no API probability ({decision.reason})")
                print(f"{name}: {decision.reason}, elapsed={elapsed_ms:.1f}ms")
                continue
            actual_wake = decision.should_take_turn is None
            print(
                f"{name}: wake={decision.probability:.4f}, "
                f"decision={'WAKE' if actual_wake else 'BLOCK'}, elapsed={elapsed_ms:.1f}ms"
            )
            expected_decision = None if expected_wake else False
            if decision.should_take_turn is not expected_decision:
                failures.append(
                    f"{name}: expected {'WAKE' if expected_wake else 'BLOCK'}, "
                    f"wake={decision.probability:.4f}, reason={decision.reason}"
                )

    assert not failures, "; ".join(failures)
