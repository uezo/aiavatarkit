import asyncio
import os
import re
import tempfile
import pytest
from uuid import uuid4
from aiavatar.sts import STSPipeline
from aiavatar.sts.quick_responder.pro import (
    QuickResponderPro,
    DEFAULT_QRP_SYSTEM_PROMPT,
    DEFAULT_QRP_SYSTEM_PROMPT_JA,
    DEFAULT_QRP_PROMPT_PREFIX,
    DEFAULT_QRP_PROMPT_PREFIX_JA,
    DEFAULT_QRP_REQUEST_PREFIX,
    DEFAULT_QRP_REQUEST_PREFIX_JA,
    DEFAULT_QRP_THINK_TAG_CONTENT,
    DEFAULT_QRP_THINK_TAG_CONTENT_JA,
    DEFAULT_QRP_CONTINUATION_MESSAGE,
    DEFAULT_QRP_CONTINUATION_MESSAGE_JA,
    DEFAULT_QRP_FALLBACK_PHRASES,
    DEFAULT_QRP_FALLBACK_PHRASES_JA,
)
from aiavatar.sts.models import STSRequest
from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.sts.llm.context_manager import SQLiteContextManager
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder


def make_pipeline():
    return STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4.1-nano",
        ),
        tts=VoicevoxSpeechSynthesizer(
            base_url=os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021"),
            speaker=46,
        ),
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True,
    )


def make_qrp(**kwargs):
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    defaults = dict(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-nano",
        tts=VoicevoxSpeechSynthesizer(
            base_url=os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021"),
            speaker=46,
        ),
        context_manager=SQLiteContextManager(db_path=f.name),
        language="ja",
        timeout=0,
        debug=True,
    )
    defaults.update(kwargs)
    return QuickResponderPro(**defaults)


def make_request(text="こんにちは"):
    return STSRequest(
        session_id=f"test_qrp_{uuid4()}",
        user_id="test_user",
        context_id=f"test_qrp_ctx_{uuid4()}",
        text=text,
    )


# -- Defaults --

def test_defaults_en():
    qrp = make_qrp(language=None)
    assert qrp.system_prompt == DEFAULT_QRP_SYSTEM_PROMPT
    assert qrp.prompt_prefix == DEFAULT_QRP_PROMPT_PREFIX
    assert qrp.request_prefix == DEFAULT_QRP_REQUEST_PREFIX
    assert qrp.think_tag_content == DEFAULT_QRP_THINK_TAG_CONTENT
    assert qrp.continuation_message == DEFAULT_QRP_CONTINUATION_MESSAGE
    assert qrp.fallback_phrases == DEFAULT_QRP_FALLBACK_PHRASES


def test_defaults_ja():
    qrp = make_qrp(language="ja")
    assert qrp.system_prompt == DEFAULT_QRP_SYSTEM_PROMPT_JA
    assert qrp.prompt_prefix == DEFAULT_QRP_PROMPT_PREFIX_JA
    assert qrp.request_prefix == DEFAULT_QRP_REQUEST_PREFIX_JA
    assert qrp.think_tag_content == DEFAULT_QRP_THINK_TAG_CONTENT_JA
    assert qrp.continuation_message == DEFAULT_QRP_CONTINUATION_MESSAGE_JA
    assert qrp.fallback_phrases == DEFAULT_QRP_FALLBACK_PHRASES_JA


def test_defaults_ja_jp():
    qrp = make_qrp(language="ja-JP")
    assert qrp.system_prompt == DEFAULT_QRP_SYSTEM_PROMPT_JA


def test_custom_overrides():
    qrp = make_qrp(
        language="ja",
        system_prompt="custom system",
        prompt_prefix="custom prefix",
        request_prefix="custom request",
        think_tag_content="custom think",
        continuation_message="custom cont",
        fallback_phrases=["custom fallback"],
    )
    assert qrp.system_prompt == "custom system"
    assert qrp.prompt_prefix == "custom prefix"
    assert qrp.request_prefix == "custom request"
    assert qrp.think_tag_content == "custom think"
    assert qrp.continuation_message == "custom cont"
    assert qrp.fallback_phrases == ["custom fallback"]


# -- Context cleaning: _clean_user_content --

def test_clean_user_prompt_prefix_kept_as_is():
    qrp = make_qrp()
    content = f"{qrp.prompt_prefix}\n\nこんにちは"
    assert qrp._clean_user_content(content) == content


def test_clean_user_request_prefix_replaced_with_continuation_ja():
    qrp = make_qrp()
    content = '$以下の入力に対して、既にあなたが出力済みの「はい。」や類似の表現は再出力せず、\n\nこんにちは'
    assert qrp._clean_user_content(content) == "「はい。」の続きを出力してください"


def test_clean_user_request_prefix_replaced_with_continuation_en():
    qrp = make_qrp(language=None)
    content = '$For the following input, you have already output "Sure."—do NOT repeat\n\nHello'
    assert qrp._clean_user_content(content) == 'Please output the continuation of "Sure.".'


def test_clean_user_dollar_prefix_fallback_strips_to_user_text():
    qrp = make_qrp()
    content = "$unknown prefix without quotes\n\nユーザーの発話"
    assert qrp._clean_user_content(content) == "ユーザーの発話"


def test_clean_user_plain_text_unchanged():
    qrp = make_qrp()
    assert qrp._clean_user_content("こんにちは") == "こんにちは"


def test_clean_user_non_string_passthrough():
    qrp = make_qrp()
    assert qrp._clean_user_content(None) is None
    assert qrp._clean_user_content(123) == 123


# -- Context cleaning: _clean_assistant_content --

def test_clean_assistant_extract_answer_tag():
    qrp = make_qrp()
    assert qrp._clean_assistant_content("<think>思考中</think><answer>はい。</answer>") == "はい。"


def test_clean_assistant_strip_think_without_answer():
    qrp = make_qrp()
    assert qrp._clean_assistant_content("<think>思考中</think>こんにちは！") == "こんにちは！"


def test_clean_assistant_strip_control_tags():
    qrp = make_qrp()
    assert qrp._clean_assistant_content("<answer>[face:Joy]こんにちは！[face:Neutral]元気？</answer>") == "こんにちは！元気？"


def test_clean_assistant_plain_text_unchanged():
    qrp = make_qrp()
    assert qrp._clean_assistant_content("はい。") == "はい。"


def test_clean_assistant_non_string_passthrough():
    qrp = make_qrp()
    assert qrp._clean_assistant_content(None) is None


# -- Context cleaning: _get_clean_histories --

@pytest.mark.asyncio
async def test_histories_strips_leading_assistant():
    qrp = make_qrp()
    await qrp.context_manager.add_histories("ctx", [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])
    histories = await qrp._get_clean_histories("ctx")
    assert histories[0]["role"] == "user"
    assert len(histories) == 2


@pytest.mark.asyncio
async def test_histories_skips_tool_messages():
    qrp = make_qrp()
    await qrp.context_manager.add_histories("ctx", [
        {"role": "user", "content": "call tool"},
        {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": '{"result": "ok"}', "tool_call_id": "1"},
        {"role": "assistant", "content": "done"},
    ])
    histories = await qrp._get_clean_histories("ctx")
    roles = [h["role"] for h in histories]
    assert "tool" not in roles
    assert len([h for h in histories if h["role"] == "assistant"]) == 1


@pytest.mark.asyncio
async def test_histories_cleans_prompt_prefix_and_request_prefix():
    qrp = make_qrp()
    prefix = qrp.prompt_prefix
    await qrp.context_manager.add_histories("ctx", [
        {"role": "user", "content": f"{prefix}\n\nこんにちは"},
        {"role": "assistant", "content": "<think>思考</think><answer>はい。</answer>"},
        {"role": "user", "content": '$以下の入力に対して、既にあなたが出力済みの「はい。」や類似の表現は再出力せず、\n\nこんにちは'},
        {"role": "assistant", "content": "<think>考え</think><answer>[face:Joy]こんにちは！元気ですか？</answer>"},
    ])
    histories = await qrp._get_clean_histories("ctx")

    assert histories[0]["content"] == f"{prefix}\n\nこんにちは"
    assert histories[1]["content"] == "はい。"
    assert histories[2]["content"] == "「はい。」の続きを出力してください"
    assert histories[3]["content"] == "こんにちは！元気ですか？"


@pytest.mark.asyncio
async def test_histories_handles_multimodal_content():
    qrp = make_qrp()
    await qrp.context_manager.add_histories("ctx", [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            {"type": "text", "text": "plain text"},
        ]},
        {"role": "assistant", "content": "response"},
    ])
    histories = await qrp._get_clean_histories("ctx")
    user_content = histories[0]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "image_url"
    assert user_content[1]["text"] == "plain text"


# -- Generation --

@pytest.mark.asyncio
async def test_respond():
    qrp = make_qrp()
    request = make_request("日本の三大祭りを教えてください")

    await qrp.respond(request)

    assert request.quick_response_text
    assert request.quick_response_voice_text
    assert request.quick_response_audio and len(request.quick_response_audio) > 0

    # request.text overwritten with request_prefix
    assert "日本の三大祭り" in request.text
    assert request.text.startswith("$")

    # History saved
    histories = await qrp.context_manager.get_histories(request.context_id)
    assert len(histories) == 2
    assert histories[0]["role"] == "user"
    assert qrp.prompt_prefix in histories[0]["content"]
    assert "<think>" in histories[1]["content"]
    assert "<answer>" in histories[1]["content"]

    await qrp.tts.close()


@pytest.mark.asyncio
async def test_voice_cache():
    qrp = make_qrp()
    request = make_request()

    text1, _, voice1 = await qrp._generate(request.text, request.context_id)
    assert text1 in qrp.voice_cache

    text2, _, voice2 = await qrp._generate(request.text, request.context_id)
    if text1 == text2:
        assert voice1 == voice2

    qrp.clear_voice_cache()
    assert len(qrp.voice_cache) == 0

    await qrp.tts.close()


@pytest.mark.asyncio
async def test_fallback_on_timeout():
    fallback = ["フォールバック。"]
    qrp = make_qrp(timeout=0.001, fallback_phrases=fallback)
    request = make_request("日本の三大祭りを教えてください")

    await qrp.respond(request)

    assert request.quick_response_text == "フォールバック。"
    assert request.quick_response_audio
    assert "フォールバック。" in qrp.voice_cache

    await qrp.tts.close()


@pytest.mark.asyncio
async def test_fallback_voice_cache():
    fallback = ["キャッシュテスト。"]
    qrp = make_qrp(timeout=0.001, fallback_phrases=fallback)
    request = make_request()

    _, _, voice1 = await qrp._generate(request.text, request.context_id)
    assert "キャッシュテスト。" in qrp.voice_cache

    _, _, voice2 = await qrp._generate(request.text, request.context_id)
    assert voice1 == voice2

    await qrp.tts.close()


@pytest.mark.asyncio
async def test_no_timeout():
    qrp = make_qrp(fallback_phrases=["これは出ないはず。"])
    request = make_request()

    await qrp.respond(request)

    assert request.quick_response_text != "これは出ないはず。"
    assert request.quick_response_audio

    await qrp.tts.close()


@pytest.mark.asyncio
async def test_synthesize_with_cache():
    qrp = make_qrp()

    voice1, tts_time1 = await qrp._synthesize_with_cache("テスト。")
    assert voice1
    assert tts_time1 > 0
    assert "テスト。" in qrp.voice_cache

    voice2, tts_time2 = await qrp._synthesize_with_cache("テスト。")
    assert voice2 == voice1
    assert tts_time2 == 0.0

    await qrp.tts.close()


# -- Pipeline integration --

@pytest.mark.asyncio
async def test_pipeline_integration():
    pipeline = make_pipeline()
    qrp = make_qrp(tts=pipeline.tts)
    original_text = "日本の三大祭りをそれぞれの特徴とともに教えてください"
    request = make_request(original_text)

    await qrp.respond(request)

    assert request.quick_response_text
    assert request.quick_response_audio and len(request.quick_response_audio) > 0

    # request.text overwritten with request_prefix
    assert original_text in request.text
    assert request.text.startswith("$")
    assert request.text != original_text

    # Main LLM response via pipeline
    quick_response_text = request.quick_response_text
    main_responses = []
    async for response in pipeline.invoke(request):
        main_responses.append(response)

    final = [r for r in main_responses if r.type == "final"]
    assert len(final) == 1
    assert "祇園" in final[0].text

    # Main response should not repeat quick response
    answer_match = re.search(r"<answer>(.*?)</answer>", final[0].text, re.DOTALL)
    answer_text = answer_match.group(1) if answer_match else final[0].text
    assert quick_response_text not in answer_text

    await pipeline.shutdown()
    await pipeline.tts.close()


# -- generate (no side effects) --

@pytest.mark.asyncio
async def test_generate_no_history():
    """generate() should not save history."""
    qrp = make_qrp()
    context_id = f"test_gen_ctx_{uuid4()}"

    qr_text, qr_voice_text, qr_voice = await qrp.generate("こんにちは", context_id)

    assert qr_text
    assert qr_voice_text
    assert qr_voice and len(qr_voice) > 0

    # No history should be saved
    histories = await qrp.context_manager.get_histories(context_id)
    assert len(histories) == 0

    await qrp.tts.close()


# -- save_history --

@pytest.mark.asyncio
async def test_save_history():
    """save_history() should persist to context_manager."""
    qrp = make_qrp()
    context_id = f"test_hist_ctx_{uuid4()}"

    await qrp.save_history(context_id, "ユーザーの発話", "はい。")

    histories = await qrp.context_manager.get_histories(context_id)
    assert len(histories) == 2
    assert histories[0]["role"] == "user"
    assert qrp.prompt_prefix in histories[0]["content"]
    assert "ユーザーの発話" in histories[0]["content"]
    assert "<think>" in histories[1]["content"]
    assert "<answer>はい。</answer>" in histories[1]["content"]


# -- cancel_generation_task --

@pytest.mark.asyncio
async def test_cancel_generation_task_cancels_running_task():
    """cancel_generation_task() should cancel a running task."""
    qrp = make_qrp()

    async def slow_task():
        await asyncio.sleep(10)
        return "should not reach", "should not reach", b""

    task = asyncio.create_task(slow_task())
    qrp._pending_generation_tasks["session_1"] = task

    qrp.cancel_generation_task("session_1")

    # Allow event loop to process the cancellation
    await asyncio.sleep(0)
    assert task.cancelled()
    assert "session_1" not in qrp._pending_generation_tasks


def test_cancel_generation_task_no_task():
    """cancel_generation_task() should be safe when no task exists."""
    qrp = make_qrp()
    qrp.cancel_generation_task("nonexistent")  # Should not raise


# -- respond with pending generation task --

@pytest.mark.asyncio
async def test_respond_uses_pending_generation_task():
    """respond() should use a completed pending generation task result."""
    qrp = make_qrp()
    request = make_request("日本の三大祭りを教えてください")

    # Pre-generate result via background task
    task = asyncio.create_task(
        qrp.generate(request.text, request.context_id)
    )
    qrp._pending_generation_tasks[request.session_id] = task

    # Wait for generation task to complete
    await task

    # respond should pick up the pre-generated result
    await qrp.respond(request)

    assert request.quick_response_text
    assert request.quick_response_audio and len(request.quick_response_audio) > 0
    assert request.text.startswith("$")

    # History should be saved (by respond, not by generate)
    histories = await qrp.context_manager.get_histories(request.context_id)
    assert len(histories) == 2

    # Pending task should be consumed
    assert request.session_id not in qrp._pending_generation_tasks

    await qrp.tts.close()


@pytest.mark.asyncio
async def test_respond_fallback_on_cancelled_task():
    """respond() should fall back to normal generation when pending task is cancelled."""
    qrp = make_qrp()
    request = make_request("こんにちは")

    async def slow_task():
        await asyncio.sleep(10)
        return "unreachable", "unreachable", b""

    task = asyncio.create_task(slow_task())
    qrp._pending_generation_tasks[request.session_id] = task
    task.cancel()

    # respond should fall back to normal generation
    await qrp.respond(request)

    assert request.quick_response_text
    assert request.quick_response_audio and len(request.quick_response_audio) > 0

    await qrp.tts.close()
