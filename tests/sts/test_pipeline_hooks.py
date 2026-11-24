import asyncio

import pytest

from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.sts.pipeline import ResponseHandler, STSPipeline
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy


def make_pipeline() -> STSPipeline:
    pipeline = STSPipeline.__new__(STSPipeline)
    pipeline._on_before_llm_handlers = []
    pipeline._on_before_tts_handlers = []
    pipeline._on_accepted_handlers = []
    pipeline._on_finish_handlers = []
    return pipeline


@pytest.mark.asyncio
async def test_hook_channels_support_global_single_and_multiple_channels():
    pipeline = make_pipeline()
    calls = []

    @pipeline.on_before_llm
    async def global_hook(request):
        calls.append(("global", request.channel))

    @pipeline.on_before_llm(channels="phone")
    async def phone_hook(request):
        calls.append(("phone", request.channel))

    @pipeline.on_before_llm(channels=["phone", "sms"])
    async def telephone_hook(request):
        calls.append(("telephone", request.channel))

    for channel, expected in (
        ("phone", ["global", "phone", "telephone"]),
        ("sms", ["global", "telephone"]),
        ("web", ["global"]),
        (None, ["global"]),
    ):
        calls.clear()
        request = STSRequest(session_id="session", channel=channel)
        await pipeline._execute_hooks(
            pipeline._on_before_llm_handlers,
            channel,
            request,
        )
        assert [name for name, _ in calls] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("register_name", "handlers_name", "include_response"),
    (
        ("on_accepted", "_on_accepted_handlers", False),
        ("on_before_llm", "_on_before_llm_handlers", False),
        ("on_before_tts", "_on_before_tts_handlers", False),
        ("on_finish", "_on_finish_handlers", True),
    ),
)
async def test_all_pipeline_hooks_accept_channels(
    register_name,
    handlers_name,
    include_response,
):
    pipeline = make_pipeline()
    calls = []
    request = STSRequest(session_id="session", channel="phone")
    response = STSResponse(type="final", session_id="session")

    async def hook(*args):
        calls.append(args)

    returned = getattr(pipeline, register_name)(channels=["phone", "sms"])(hook)
    assert returned is hook

    args = (request, response) if include_response else (request,)
    await pipeline._execute_hooks(
        getattr(pipeline, handlers_name),
        "web",
        *args,
    )
    assert calls == []

    await pipeline._execute_hooks(
        getattr(pipeline, handlers_name),
        "phone",
        *args,
    )
    assert calls == [args]


@pytest.mark.asyncio
async def test_hook_channel_is_stable_during_execution():
    pipeline = make_pipeline()
    calls = []
    request = STSRequest(session_id="session", channel="phone")

    @pipeline.on_accepted
    async def change_request_channel(request):
        request.channel = "sms"

    @pipeline.on_accepted(channels="phone")
    async def phone_hook(request):
        calls.append(request.channel)

    await pipeline._execute_hooks(
        pipeline._on_accepted_handlers,
        "phone",
        request,
    )

    assert calls == ["sms"]


@pytest.mark.parametrize(
    "channels",
    ([], (), [""], ["phone", ""], ["phone", None]),
)
def test_hook_channels_reject_invalid_collections(channels):
    pipeline = make_pipeline()

    with pytest.raises(ValueError):
        pipeline.on_before_llm(channels=channels)


@pytest.mark.parametrize(
    ("config", "channel", "expected"),
    (
        (False, "phone", None),
        (True, "phone", "phone"),
        (True, None, None),
        (["phone", "sms"], "phone", "phone"),
        (["phone", "sms"], "websocket", None),
        (["phone", ("websocket_m5", "desktop_robot")], "websocket_m5", "desktop_robot"),
        ([], "phone", None),
    ),
)
def test_insert_channel_tag_resolves_bool_selected_and_renamed_channels(
    config,
    channel,
    expected,
):
    pipeline = make_pipeline()
    pipeline.insert_channel_tag = config

    assert pipeline._resolve_channel_tag_name(channel) == expected


@pytest.mark.parametrize(
    "config",
    (
        None,
        "phone",
        ("phone", "voice"),
        [""],
        [("phone",)],
        [("phone", "")],
        [("phone", "voice"), "phone"],
        [["phone", "voice"]],
    ),
)
def test_insert_channel_tag_rejects_invalid_rules(config):
    pipeline = make_pipeline()

    with pytest.raises((TypeError, ValueError)):
        pipeline.insert_channel_tag = config


@pytest.mark.asyncio
async def test_pipeline_invocation_inserts_only_selected_mapped_channel_tag(tmp_path):
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=LLMServiceDummy(
            response_text="response",
            db_connection_str=str(tmp_path / "context.db"),
        ),
        tts=SpeechSynthesizerDummy(synthesized_bytes=b"voice"),
        db_connection_str=str(tmp_path / "pipeline.db"),
        voice_recorder_enabled=False,
        insert_channel_tag=["phone", ("websocket_m5", "desktop_robot")],
    )

    async def handle_response(response):
        pass

    async def stop_response(session_id, context_id):
        pass

    pipeline.add_response_handler(ResponseHandler(
        can_handle=lambda session_id: session_id in {"m5-session", "websocket-session"},
        handle_response=handle_response,
        stop_response=stop_response,
    ))

    mapped_request = STSRequest(
        session_id="m5-session",
        user_id="user",
        text="Hello",
        channel="websocket_m5",
    )
    unmatched_request = STSRequest(
        session_id="websocket-session",
        user_id="user",
        text="Hello",
        channel="websocket",
    )

    try:
        _ = [response async for response in pipeline.invoke(mapped_request)]
        _ = [response async for response in pipeline.invoke(unmatched_request)]
        await asyncio.sleep(0)

        assert mapped_request.text == "<channel name='desktop_robot' />Hello"
        assert unmatched_request.text == "Hello"
    finally:
        await pipeline.shutdown()
        await pipeline.stt.close()
        await pipeline.tts.close()


@pytest.mark.asyncio
async def test_pipeline_invocation_filters_all_hooks_by_channel(tmp_path):
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=LLMServiceDummy(
            response_text="response",
            db_connection_str=str(tmp_path / "context.db"),
        ),
        tts=SpeechSynthesizerDummy(synthesized_bytes=b"voice"),
        db_connection_str=str(tmp_path / "pipeline.db"),
        voice_recorder_enabled=False,
    )
    calls = []

    async def handle_response(response):
        pass

    async def stop_response(session_id, context_id):
        pass

    pipeline.add_response_handler(ResponseHandler(
        can_handle=lambda session_id: session_id == "session",
        handle_response=handle_response,
        stop_response=stop_response,
    ))

    def register_hooks(channels, prefix):
        for name in ("on_accepted", "on_before_llm", "on_before_tts", "on_finish"):
            async def hook(*args, name=name):
                calls.append(f"{prefix}:{name}")

            getattr(pipeline, name)(channels=channels)(hook)

    register_hooks("phone", "matched")
    register_hooks(["sms", "linebot"], "skipped")

    try:
        responses = [response async for response in pipeline.invoke(STSRequest(
            session_id="session",
            user_id="user",
            text="request",
            channel="phone",
        ))]
        await asyncio.sleep(0)

        assert responses[-1].type == "final"
        assert calls == [
            "matched:on_accepted",
            "matched:on_before_llm",
            "matched:on_before_tts",
            "matched:on_finish",
        ]
    finally:
        await pipeline.shutdown()
        await pipeline.stt.close()
        await pipeline.tts.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("voice_text_tag", "response_text", "expected"),
    (
        pytest.param(
            ["ack", "answer"],
            "<ack>了解</ack>\n<think>内部。</think><answer>最初。続き。最後</answer>",
            [("了解", "ack"), ("最初。", "answer"), ("続き。", "answer"), ("最後", "answer")],
            id="tagged",
        ),
        pytest.param(["answer"], "タグなし。", [("タグなし。", None)], id="fallback"),
        pytest.param(None, "設定なし。", [("設定なし。", None)], id="unconfigured"),
    ),
)
async def test_pipeline_voice_text_tags(tmp_path, voice_text_tag, response_text, expected):
    pipeline = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=LLMServiceDummy(
            response_text=response_text,
            voice_text_tag=voice_text_tag,
            db_connection_str=str(tmp_path / "context.db"),
        ),
        tts=SpeechSynthesizerDummy(synthesized_bytes=b"voice"),
        db_connection_str=str(tmp_path / "pipeline.db"),
        voice_recorder_enabled=False,
        voice_recorder_dir=str(tmp_path / "voices"),
    )
    try:
        responses = [response async for response in pipeline.invoke(STSRequest(
            session_id="session", user_id="user", text="request"
        ))]
    finally:
        await pipeline.shutdown()
        await pipeline.stt.close()
        await pipeline.tts.close()

    chunks = [response for response in responses if response.type == "chunk"]
    assert [(chunk.voice_text, chunk.voice_text_tag) for chunk in chunks if chunk.voice_text] == expected
    assert all(chunk.voice_text_tag is None for chunk in chunks if not chunk.voice_text)
    assert responses[-1].type == "final"
    assert responses[-1].voice_text == "".join(text for text, _ in expected)
    assert responses[-1].voice_text_tag is None
