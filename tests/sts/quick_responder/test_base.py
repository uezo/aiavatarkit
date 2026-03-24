import os
import re
import pytest
from uuid import uuid4
from aiavatar.sts import STSPipeline
from aiavatar.sts.quick_responder.base import (
    QuickResponder,
    DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX,
    DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA,
    DEFAULT_REQUEST_PREFIX,
    DEFAULT_REQUEST_PREFIX_JA,
    DEFAULT_FALLBACK_PHRASES,
    DEFAULT_FALLBACK_PHRASES_JA,
)
from aiavatar.sts.models import STSRequest
from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
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


def make_llm(**kwargs):
    defaults = dict(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt="You are a helpful assistant.",
        model="gpt-4.1-nano",
    )
    defaults.update(kwargs)
    return ChatGPTService(**defaults)


def make_qr(**kwargs):
    defaults = dict(
        llm=make_llm(),
        tts=VoicevoxSpeechSynthesizer(
            base_url=os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021"),
            speaker=46,
        ),
        inline_llm_params={"tools": [], "tool_choice": "none"},
        timeout=0,
        debug=True,
    )
    defaults.update(kwargs)
    return QuickResponder(**defaults)


def make_request(text="こんにちは"):
    return STSRequest(
        session_id=f"test_qr_{uuid4()}",
        user_id="test_user",
        context_id=f"test_qr_ctx_{uuid4()}",
        text=text,
    )


# -- Defaults --

def test_defaults():
    qr = make_qr()
    assert qr.quick_response_prompt_prefix == DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX
    assert qr.request_prefix == DEFAULT_REQUEST_PREFIX
    assert qr.fallback_phrases == DEFAULT_FALLBACK_PHRASES


def test_defaults_ja():
    qr = make_qr(
        quick_response_prompt_prefix=DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA,
        request_prefix=DEFAULT_REQUEST_PREFIX_JA,
        fallback_phrases=DEFAULT_FALLBACK_PHRASES_JA,
    )
    assert qr.quick_response_prompt_prefix == DEFAULT_QUICK_RESPONSE_PROMPT_PREFIX_JA
    assert qr.request_prefix == DEFAULT_REQUEST_PREFIX_JA
    assert qr.fallback_phrases == DEFAULT_FALLBACK_PHRASES_JA


def test_custom_overrides():
    qr = make_qr(
        quick_response_prompt_prefix="custom prefix",
        request_prefix="custom request",
        fallback_phrases=["custom fallback"],
    )
    assert qr.quick_response_prompt_prefix == "custom prefix"
    assert qr.request_prefix == "custom request"
    assert qr.fallback_phrases == ["custom fallback"]


def test_inline_llm_params_default():
    qr = QuickResponder(llm=make_llm(), tts=make_qr().tts)
    assert qr.inline_llm_params == {"reasoning_effort": "none", "tools": [], "tool_choice": "none"}


def test_inline_llm_params_custom():
    custom = {"temperature": 0.0}
    qr = make_qr(inline_llm_params=custom)
    assert qr.inline_llm_params == custom


# -- Generation --

@pytest.mark.asyncio
async def test_respond():
    qr = make_qr()
    request = make_request("日本の三大祭りを教えてください")

    await qr.respond(request)

    assert request.quick_response_text
    assert request.quick_response_voice_text
    assert request.quick_response_audio and len(request.quick_response_audio) > 0

    # request.text overwritten with request_prefix
    assert "日本の三大祭り" in request.text
    assert request.text.startswith("$")

    await qr.tts.close()


@pytest.mark.asyncio
async def test_voice_cache():
    qr = make_qr()
    request = make_request()

    text1, _, voice1 = await qr._generate(request)
    assert text1 in qr.voice_cache

    text2, _, voice2 = await qr._generate(request)
    if text1 == text2:
        assert voice1 == voice2

    qr.clear_voice_cache()
    assert len(qr.voice_cache) == 0

    await qr.tts.close()


@pytest.mark.asyncio
async def test_fallback_on_timeout():
    fallback = ["フォールバック。"]
    qr = make_qr(timeout=0.001, fallback_phrases=fallback)
    request = make_request("日本の三大祭りを教えてください")

    await qr.respond(request)

    assert request.quick_response_text == "フォールバック。"
    assert request.quick_response_audio
    assert "フォールバック。" in qr.voice_cache

    await qr.tts.close()


@pytest.mark.asyncio
async def test_fallback_voice_cache():
    fallback = ["キャッシュテスト。"]
    qr = make_qr(timeout=0.001, fallback_phrases=fallback)
    request = make_request()

    _, _, voice1 = await qr._generate(request)
    assert "キャッシュテスト。" in qr.voice_cache

    _, _, voice2 = await qr._generate(request)
    assert voice1 == voice2

    await qr.tts.close()


@pytest.mark.asyncio
async def test_no_timeout():
    qr = make_qr(fallback_phrases=["これは出ないはず。"])
    request = make_request()

    await qr.respond(request)

    assert request.quick_response_text != "これは出ないはず。"
    assert request.quick_response_audio

    await qr.tts.close()


@pytest.mark.asyncio
async def test_request_text_format():
    qr = make_qr()
    request = make_request("テスト入力")

    await qr.respond(request)

    # request_prefix contains {quick_response_text} placeholder, replaced with actual text
    assert request.quick_response_text in request.text
    assert "テスト入力" in request.text

    await qr.tts.close()


# -- Pipeline integration --

@pytest.mark.asyncio
async def test_pipeline_integration():
    pipeline = make_pipeline()
    qr = QuickResponder(
        llm=pipeline.llm,
        tts=pipeline.tts,
        inline_llm_params={"tools": [], "tool_choice": "none"},
        timeout=0,
        debug=True,
    )
    original_text = "日本の三大祭りをそれぞれの特徴とともに教えてください"
    request = make_request(original_text)

    await qr.respond(request)

    assert request.quick_response_text
    assert request.quick_response_audio and len(request.quick_response_audio) > 0

    # request.text overwritten with request_prefix
    assert original_text in request.text
    assert request.quick_response_text in request.text
    assert request.text != original_text

    # Main LLM response via pipeline
    quick_response_text = request.quick_response_text
    quick_response_voice_text = request.quick_response_voice_text
    main_responses = []
    async for response in pipeline.invoke(request):
        main_responses.append(response)

    # is_first_chunk: quick response chunk should be first, LLM chunks should not
    chunks = [r for r in main_responses if r.type == "chunk"]
    assert len(chunks) >= 2
    assert chunks[0].metadata.get("is_quick_response") is True
    assert chunks[0].metadata.get("is_first_chunk") is True
    for c in chunks[1:]:
        assert c.metadata.get("is_first_chunk") is not True

    # final should contain quick response text and voice_text
    final = [r for r in main_responses if r.type == "final"]
    assert len(final) == 1
    assert final[0].text.startswith(quick_response_text)
    assert final[0].voice_text.startswith(quick_response_voice_text)
    assert "祇園" in final[0].text

    # Main response (excluding quick response prefix) should not repeat quick response
    main_response_text = final[0].text[len(quick_response_text):]
    answer_match = re.search(r"<answer>(.*?)</answer>", main_response_text, re.DOTALL)
    answer_text = answer_match.group(1) if answer_match else main_response_text
    assert quick_response_text not in answer_text

    await pipeline.shutdown()
    await pipeline.tts.close()
