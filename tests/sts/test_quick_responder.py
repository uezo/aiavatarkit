import os
import re
import pytest
from uuid import uuid4
from aiavatar.sts import STSPipeline
from aiavatar.sts.quick_responder import QuickResponder
from aiavatar.sts.models import STSRequest, STSResponse
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
            model="gpt-5.2",
        ),
        tts=VoicevoxSpeechSynthesizer(
            base_url="http://127.0.0.1:50021",
            speaker=46,
            debug=True
        ),
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )


@pytest.mark.asyncio
async def test_respond():
    pipeline = make_pipeline()

    handled_responses = []

    async def mock_handle_response(response: STSResponse):
        handled_responses.append(response)

    pipeline.handle_response = mock_handle_response

    qr = QuickResponder(pipeline)
    original_text = "日本の三大祭りをそれぞれの特徴とともに教えてください"
    request = STSRequest(
        session_id=f"test_qr_{str(uuid4())}",
        user_id="test_user",
        context_id=f"test_qr_ctx_{str(uuid4())}",
        text=original_text,
    )

    await qr.respond(request)

    # Quick response should be a short acknowledgment, not the actual answer
    assert len(handled_responses) == 1
    resp = handled_responses[0]
    assert resp.type == "chunk"
    assert resp.text is not None and len(resp.text) > 0
    assert resp.audio_data is not None and len(resp.audio_data) > 0
    assert "祇園" not in resp.text  # Quick response should not contain the answer

    # request.text was overwritten with dedup prefix
    assert original_text in request.text
    assert resp.text in request.text
    assert request.text != original_text

    # Main LLM response should contain the actual answer
    quick_response_text = resp.text
    main_responses = []
    async for response in pipeline.invoke(request):
        main_responses.append(response)

    final = [r for r in main_responses if r.type == "final"]
    assert len(final) == 1
    assert "祇園" in final[0].text  # Main response should contain the answer
    # Main response (answer portion) should not repeat the quick response
    answer_match = re.search(r"<answer>(.*?)</answer>", final[0].text, re.DOTALL)
    answer_text = answer_match.group(1) if answer_match else final[0].text
    assert quick_response_text not in answer_text

    await pipeline.shutdown()
    await pipeline.tts.close()


@pytest.mark.asyncio
async def test_voice_cache():
    pipeline = make_pipeline()
    async def noop_handle_response(r): pass
    pipeline.handle_response = noop_handle_response

    qr = QuickResponder(pipeline)

    request = STSRequest(
        session_id=f"test_qr_{str(uuid4())}",
        user_id="test_user",
        context_id=f"test_qr_ctx_{str(uuid4())}",
        text="こんにちは",
    )

    # First call: cache should be populated
    qr_text1, _, qr_voice1 = await qr._generate(request)
    assert qr_text1 in qr.voice_cache
    assert qr_voice1 is not None

    # Second call with same response: should use cache
    qr_text2, _, qr_voice2 = await qr._generate(request)
    if qr_text1 == qr_text2:
        assert qr_voice1 == qr_voice2  # Same cached audio

    # Clear cache
    qr.clear_voice_cache()
    assert len(qr.voice_cache) == 0

    await pipeline.shutdown()
    await pipeline.tts.close()


@pytest.mark.asyncio
async def test_fallback_on_timeout():
    pipeline = make_pipeline()

    handled_responses = []

    async def mock_handle_response(response: STSResponse):
        handled_responses.append(response)

    pipeline.handle_response = mock_handle_response

    fallback_phrases = ["フォールバック。", "タイムアウト。"]
    qr = QuickResponder(
        pipeline,
        timeout=0.001,  # 1ms - LLM can't possibly respond in time
        fallback_phrases=fallback_phrases,
        debug=True
    )

    request = STSRequest(
        session_id=f"test_qr_{str(uuid4())}",
        user_id="test_user",
        context_id=f"test_qr_ctx_{str(uuid4())}",
        text="日本の三大祭りをそれぞれの特徴とともに教えてください",
    )

    await qr.respond(request)

    assert len(handled_responses) == 1
    resp = handled_responses[0]
    assert resp.type == "chunk"
    assert resp.text in fallback_phrases
    assert resp.audio_data is not None and len(resp.audio_data) > 0

    # Fallback voice should be cached
    assert resp.text in qr.voice_cache

    await pipeline.shutdown()
    await pipeline.tts.close()


@pytest.mark.asyncio
async def test_fallback_voice_cache():
    pipeline = make_pipeline()
    async def noop_handle_response(r): pass
    pipeline.handle_response = noop_handle_response

    fallback_phrases = ["キャッシュテスト。"]
    qr = QuickResponder(
        pipeline,
        timeout=0.001,
        fallback_phrases=fallback_phrases,
        debug=True
    )

    request = STSRequest(
        session_id=f"test_qr_{str(uuid4())}",
        user_id="test_user",
        context_id=f"test_qr_ctx_{str(uuid4())}",
        text="テスト",
    )

    # First call: TTS is invoked
    text1, _, voice1 = await qr._generate(request)
    assert text1 == "キャッシュテスト。"
    assert voice1 is not None
    assert "キャッシュテスト。" in qr.voice_cache

    # Second call: voice from cache (same bytes)
    text2, _, voice2 = await qr._generate(request)
    assert text2 == "キャッシュテスト。"
    assert voice2 == voice1

    await pipeline.shutdown()
    await pipeline.tts.close()


@pytest.mark.asyncio
async def test_no_timeout():
    """timeout=0 disables the timeout, LLM response is used normally."""
    pipeline = make_pipeline()

    handled_responses = []

    async def mock_handle_response(response: STSResponse):
        handled_responses.append(response)

    pipeline.handle_response = mock_handle_response

    qr = QuickResponder(
        pipeline,
        timeout=0,
        fallback_phrases=["これは出ないはず。"],
        debug=True
    )

    request = STSRequest(
        session_id=f"test_qr_{str(uuid4())}",
        user_id="test_user",
        context_id=f"test_qr_ctx_{str(uuid4())}",
        text="こんにちは",
    )

    await qr.respond(request)

    assert len(handled_responses) == 1
    resp = handled_responses[0]
    assert resp.text != "これは出ないはず。"  # LLM response, not fallback
    assert resp.audio_data is not None

    await pipeline.shutdown()
    await pipeline.tts.close()


@pytest.mark.asyncio
async def test_custom_prompts():
    pipeline = make_pipeline()
    async def noop_handle_response(r): pass
    pipeline.handle_response = noop_handle_response

    custom_prefix = "$Reply with exactly one word: OK."
    custom_request_prefix = "$You already said \"{quick_response_text}\". Continue:"

    qr = QuickResponder(
        pipeline,
        quick_response_prompt_prefix=custom_prefix,
        request_prefix=custom_request_prefix,
    )

    assert qr.quick_response_prompt_prefix == custom_prefix
    assert qr.request_prefix == custom_request_prefix

    request = STSRequest(
        session_id=f"test_qr_{str(uuid4())}",
        user_id="test_user",
        context_id=f"test_qr_ctx_{str(uuid4())}",
        text="Hello",
    )

    await qr.respond(request)

    # request_prefix should be formatted with the generated text
    assert "Continue:" in request.text
    assert "Hello" in request.text

    await pipeline.shutdown()
    await pipeline.tts.close()
