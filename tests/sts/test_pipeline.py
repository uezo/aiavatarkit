import asyncio
import os
import pytest
import numpy
from datetime import datetime, timezone
from time import sleep
from uuid import uuid4
from aiavatar.sts import STSPipeline
from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.vad.standard import StandardSpeechDetector
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.stt.google import GoogleSpeechRecognizer
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder
from aiavatar.sts.models import STSRequest, STSResponse
from aiavatar.adapter import Adapter

INPUT_VOICE_SAMPLE_RATE = 24000 # using VOICEVOX


class RecordingAdapter(Adapter):
    """
    A custom ResponseHandler that records the final audio data in memory.
    """
    def __init__(self, sts: STSPipeline):
        super().__init__(sts)
        self.user_id = None
        self.final_context_id = None
        self.final_audio = bytes()

    async def handle_request(self, request: STSRequest):
        async for response in self.sts.invoke(request):
            await self.handle_response(response)

    async def handle_response(self, response: STSResponse):
        if response.type == "start":
            self.final_user_id = response.user_id
            self.final_context_id = response.context_id
        if response.type == "chunk" and response.audio_data:
            self.final_audio += response.audio_data
        # We only care about the "final" response which carries the entire synthesized audio
        elif response.type == "final" and response.audio_data:
            self.final_audio = response.audio_data

    async def stop_response(self, session_id: str, context_id: str):
        # For this test, we do not need to do anything special
        pass

@pytest.mark.asyncio
async def test_sts_pipeline():
    """
    Integration test scenario:
      1. Generate audio for "日本の首都は？" via Voicevox
      2. Pass that audio to STSPipeline.invoke()
         -> STT transcribes -> LLM answers (contains "東京") -> TTS re-synthesizes
      3. A custom ResponseHandler captures the final synthesized audio
      4. Use GoogleSpeechRecognizer on the final audio to check if "東京" is present.
    """
    session_id = f"test_sts_pipeline_{str(uuid4())}"

    # TTS for input audio instead of human's speech
    voicevox_for_input = VoicevoxSpeechSynthesizer(
        base_url="http://127.0.0.1:50021",
        speaker=46,
        debug=True
    )

    async def get_input_voice(text: str):
        return await voicevox_for_input.synthesize(text)

    # STT for output audio instead of human's listening
    stt_for_final = GoogleSpeechRecognizer(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        sample_rate=INPUT_VOICE_SAMPLE_RATE,
        language="ja-JP",
        debug=True
    )

    async def get_output_text(voice: bytes):
        return await stt_for_final.transcribe(voice)

    # Initialize pipeline
    sts = STSPipeline(
        vad=StandardSpeechDetector(
            volume_db_threshold=-50.0,
            silence_duration_threshold=0.5,
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            debug=True
        ),
        stt=GoogleSpeechRecognizer(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            language="ja-JP",
            debug=True
        ),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=VoicevoxSpeechSynthesizer(
            base_url="http://127.0.0.1:50021",
            speaker=46,
            debug=True
        ),
        performance_recorder=SQLitePerformanceRecorder(),  # DB記録
        debug=True
    )

    # Adapter for test
    adapter = RecordingAdapter(sts)

    # Invoke pipeline with the first request (Ask capital of Japan)
    await adapter.handle_request(STSRequest(session_id=session_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("日本の首都は？")))
    context_id = adapter.final_context_id

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "東京" in output_text, f"Expected '東京' in recognized text, but got: {output_text}"

    # Invoke pipeline with the successive request (Ask about of US, without using the word 'capital' to check context)
    await adapter.handle_request(STSRequest(session_id=session_id, context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("アメリカは？")))

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "ワシントン" in output_text, f"Expected 'ワシントン' in recognized text, but got: {output_text}"

    await sts.shutdown()
    await voicevox_for_input.close()
    await stt_for_final.close()


@pytest.mark.asyncio
async def test_sts_pipeline_wakeword():
    session_id = f"test_sts_pipeline_wakeword_{str(uuid4())}"

    # TTS for input audio instead of human's speech
    voicevox_for_input = VoicevoxSpeechSynthesizer(
        base_url="http://127.0.0.1:50021",
        speaker=46,
        debug=True
    )

    async def get_input_voice(text: str):
        return await voicevox_for_input.synthesize(text)

    # STT for output audio instead of human's listening
    stt_for_final = GoogleSpeechRecognizer(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        sample_rate=INPUT_VOICE_SAMPLE_RATE,
        language="ja-JP",
        debug=True
    )

    async def get_output_text(voice: bytes):
        return await stt_for_final.transcribe(voice)

    # Initialize pipeline
    sts = STSPipeline(
        vad=StandardSpeechDetector(
            volume_db_threshold=-50.0,
            silence_duration_threshold=0.5,
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            debug=True
        ),
        stt=GoogleSpeechRecognizer(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            language="ja-JP",
            debug=True
        ),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=VoicevoxSpeechSynthesizer(
            base_url="http://127.0.0.1:50021",
            speaker=46,
            debug=True
        ),
        wakewords=["こんにちは"],
        wakeword_timeout=10,
        performance_recorder=SQLitePerformanceRecorder(),  # DB記録
        debug=True
    )

    # Adapter for test
    adapter = RecordingAdapter(sts)

    # First request without wakeword: not invoked
    await adapter.handle_request(STSRequest(
        session_id=session_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("もしもし")
    ))
    context_id = adapter.final_context_id

    # Check no voice generated
    assert adapter.final_audio is b""
    assert context_id is None

    # Second request with wakeword: invoked
    await adapter.handle_request(STSRequest(
        session_id=session_id, context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("やあ、こんにちは！ところで日本の首都は？")
    ))
    context_id = adapter.final_context_id

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "東京" in output_text, f"Expected '東京' in recognized text, but got: {output_text}"
    adapter.final_audio = b""

    # Third request without wakeword, within wakeword timeout: invoked
    await adapter.handle_request(STSRequest(
        session_id=session_id, context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("アメリカは？")
    ))

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "ワシントン" in output_text, f"Expected 'ワシントン' in recognized text, but got: {output_text}"
    adapter.final_audio = b""

    # Wait for timeout
    await asyncio.sleep(10)

    # Fourth request without wakeword, timeout: not invoked
    await adapter.handle_request(STSRequest(
        session_id=session_id, context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("じゃあフランスは？")
    ))

    # Check no voice generated
    assert adapter.final_audio is b""
    assert adapter.final_context_id == context_id   # Wakeword timeout but context is still alive

    # Second request with wakeword: invoked
    await adapter.handle_request(STSRequest(
        session_id=session_id, context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("こんにちは。ドイツはどうだろうか？")
    ))
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "ベルリン" in output_text, f"Expected 'ベルリン' in recognized text, but got: {output_text}"
    adapter.final_audio = b""
    assert adapter.final_context_id == context_id

    await sts.shutdown()
    await voicevox_for_input.close()
    await stt_for_final.close()


@pytest.mark.asyncio
async def test_sts_pipeline_novoice():
    session_id = f"test_sts_pipeline_novoice_{str(uuid4())}"

    # Initialize pipeline
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),  # DB記録
        debug=True
    )

    async for response in sts.invoke(STSRequest(session_id=session_id, context_id="test_pipeline_novoice", user_id="aiavatar.sts_user", text="こんにちは")):
        if response.type == "chunk" or response.type == "final":
            assert response.text is not None and response.text != ""
            assert response.voice_text is not None and response.voice_text != ""
            assert response.audio_data is None


@pytest.mark.asyncio
async def test_sts_pipeline_with_user():
    session_id = f"test_sts_pipeline_with_user_{str(uuid4())}"

    # TTS for input audio instead of human's speech
    voicevox_for_input = VoicevoxSpeechSynthesizer(
        base_url="http://127.0.0.1:50021",
        speaker=46,
        debug=True
    )

    async def get_input_voice(text: str):
        return await voicevox_for_input.synthesize(text)

    # STT for output audio instead of human's listening
    stt_for_final = GoogleSpeechRecognizer(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        sample_rate=INPUT_VOICE_SAMPLE_RATE,
        language="ja-JP",
        debug=True
    )

    async def get_output_text(voice: bytes):
        return await stt_for_final.transcribe(voice)

    # Initialize pipeline
    sts = STSPipeline(
        vad=StandardSpeechDetector(
            volume_db_threshold=-50.0,
            silence_duration_threshold=0.5,
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            debug=True
        ),
        stt=GoogleSpeechRecognizer(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            sample_rate=INPUT_VOICE_SAMPLE_RATE,
            language="ja-JP",
            debug=True
        ),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=VoicevoxSpeechSynthesizer(
            base_url="http://127.0.0.1:50021",
            speaker=46,
            debug=True
        ),
        performance_recorder=SQLitePerformanceRecorder(),  # DB記録
        debug=True
    )

    user_id = "test_sts_pipeline_with_user_user"

    # Set user to vad as recording session data
    sts.vad.set_session_data(session_id, "user_id", user_id, True) # True: Create session if not exists
    assert sts.vad.recording_sessions[session_id].data == {"user_id": user_id}

    # Adapter for test
    adapter = RecordingAdapter(sts)

    # Create input voice data with silent section
    audio_data = await get_input_voice("こんにちは")
    silence_samples = int(INPUT_VOICE_SAMPLE_RATE * 0.5)
    silence_bytes = numpy.zeros(silence_samples, dtype=numpy.int16).tobytes()
    audio_data += silence_bytes

    for i in range(0, len(audio_data), 512):
        chunk = audio_data[i:i+512]
        await sts.vad.process_samples(chunk, session_id)

    # Wait for processing ends
    await asyncio.sleep(5)

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "こんにちは" in output_text, f"Expected 'こんにちは' in recognized text, but got: {output_text}"

    # Check user_id
    assert adapter.final_user_id == user_id

    await sts.shutdown()
    await voicevox_for_input.close()
    await stt_for_final.close()


@pytest.mark.asyncio
async def test_request_merging_enabled():
    session_id = f"test_request_merging_enabled_{str(uuid4())}"

    """Test request merging when merge_request_threshold > 0"""
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        merge_request_threshold=3.0,  # 3 second threshold
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    user_id = "test_user"
    context_id = "test_context"

    # First request
    request1 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="Hello"
    )

    responses1 = []
    async for response in sts.invoke(request1):
        responses1.append(response)

    # Immediately send second request (within threshold)
    request2 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="World"
    )

    responses2 = []
    async for response in sts.invoke(request2):
        responses2.append(response)

    # Check that second request was merged with prefix
    assert sts.merge_request_prefix in request2.text
    assert "Hello" in request2.text  # Previous request text should be included
    assert "World" in request2.text  # Current request text should be included
    
    await sts.shutdown()


@pytest.mark.asyncio
async def test_request_merging_allow_merge_false():
    """Test that requests are not merged when allow_merge=False even within threshold"""
    session_id = f"test_request_merging_allow_merge_false_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        merge_request_threshold=3.0,  # Merging enabled with 3 second threshold
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    user_id = "test_user"
    context_id = "test_context"

    # First request
    request1 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="Hello"
    )

    responses1 = []
    async for response in sts.invoke(request1):
        responses1.append(response)

    # Second request with allow_merge=False (within threshold)
    request2 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="World",
        allow_merge=False
    )

    responses2 = []
    async for response in sts.invoke(request2):
        responses2.append(response)

    # Check that second request was NOT merged because allow_merge=False
    assert sts.merge_request_prefix not in request2.text
    assert request2.text == "World"  # Should remain unchanged

    await sts.shutdown()


@pytest.mark.asyncio
async def test_request_merging_disabled():
    session_id = f"test_request_merging_disabled_{str(uuid4())}"

    """Test that requests are not merged when merge_request_threshold = 0"""
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        merge_request_threshold=0.0,  # Disabled
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    user_id = "test_user"
    context_id = "test_context"

    # First request
    request1 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="Hello"
    )

    responses1 = []
    async for response in sts.invoke(request1):
        responses1.append(response)

    # Immediately send second request
    request2 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="World"
    )

    responses2 = []
    async for response in sts.invoke(request2):
        responses2.append(response)

    # Check that second request was NOT merged
    assert sts.merge_request_prefix not in request2.text
    assert request2.text == "World"  # Should remain unchanged
    
    await sts.shutdown()


@pytest.mark.asyncio
async def test_request_merging_threshold_exceeded():
    session_id = f"test_request_merging_threshold_exceeded_{str(uuid4())}"

    """Test that requests are not merged when time threshold is exceeded"""
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        merge_request_threshold=0.5,  # 0.5 second threshold
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    user_id = "test_user"
    context_id = "test_context"

    # First request
    request1 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="Hello"
    )

    responses1 = []
    async for response in sts.invoke(request1):
        responses1.append(response)

    # Wait longer than threshold
    sleep(0.6)

    # Send second request after threshold
    request2 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="World"
    )

    responses2 = []
    async for response in sts.invoke(request2):
        responses2.append(response)

    # Check that second request was NOT merged due to time threshold
    assert sts.merge_request_prefix not in request2.text
    assert request2.text == "World"  # Should remain unchanged
    
    await sts.shutdown()


@pytest.mark.asyncio
async def test_request_merging_different_sessions():
    session_id = f"test_request_merging_different_sessions_{str(uuid4())}"

    """Test that requests from different sessions are not merged"""
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        merge_request_threshold=1.0,  # 1 second threshold
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    user_id = "test_user"
    context_id = "test_context"

    # First request from session 1
    request1 = STSRequest(
        session_id=f"{session_id}_1",
        user_id=user_id,
        context_id=context_id,
        text="Hello"
    )

    responses1 = []
    async for response in sts.invoke(request1):
        responses1.append(response)

    # Immediately send request from session 2
    request2 = STSRequest(
        session_id=f"{session_id}_2",
        user_id=user_id,
        context_id=context_id,
        text="World"
    )

    responses2 = []
    async for response in sts.invoke(request2):
        responses2.append(response)

    # Check that requests from different sessions are NOT merged
    assert sts.merge_request_prefix not in request2.text
    assert request2.text == "World"  # Should remain unchanged
    
    await sts.shutdown()


@pytest.mark.asyncio
async def test_request_merging_with_files():
    session_id = f"test_request_merging_with_files_{str(uuid4())}"

    """Test that files are preserved during request merging"""
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        merge_request_threshold=1.0,  # 1 second threshold
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    user_id = "test_user"
    context_id = "test_context"

    # First request with files
    request1 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="Hello",
        files={"file1": "content1"}
    )

    responses1 = []
    async for response in sts.invoke(request1):
        responses1.append(response)

    # Second request without files (within threshold)
    request2 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="World"
    )

    responses2 = []
    async for response in sts.invoke(request2):
        responses2.append(response)

    # Check that files from previous request are preserved
    assert request2.files == {"file1": "content1"}
    assert sts.merge_request_prefix in request2.text
    
    await sts.shutdown()


@pytest.mark.asyncio
async def test_request_merging_prefix_removal():
    session_id = f"test_request_merging_prefix_removal_{str(uuid4())}"

    """Test that merge prefix is properly removed from previous request text"""
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        ),
        tts=SpeechSynthesizerDummy(),
        merge_request_threshold=3.0,  # 3 second threshold
        merge_request_prefix="$MERGED:",
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    user_id = "test_user"
    context_id = "test_context"

    # First request
    request1 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="First request"
    )

    responses1 = []
    async for response in sts.invoke(request1):
        responses1.append(response)

    # Second request (will be merged)
    request2 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="Second request"
    )

    responses2 = []
    async for response in sts.invoke(request2):
        responses2.append(response)

    # Third request (should remove prefix from previous merged request)
    request3 = STSRequest(
        session_id=session_id,
        user_id=user_id,
        context_id=context_id,
        text="Third request"
    )

    responses3 = []
    async for response in sts.invoke(request3):
        responses3.append(response)

    # Check that prefix was properly removed and text was merged correctly
    assert "$MERGED:" in request3.text
    # The merged text should not contain duplicate prefixes
    assert request3.text.count("$MERGED:") == 1
    assert "First request" in request3.text
    assert "Second request" in request3.text
    assert "Third request" in request3.text

    await sts.shutdown()


@pytest.mark.asyncio
async def test_invoke_queued_sequential():
    """Test that invoke with use_invoke_queue=True processes requests sequentially"""
    session_id = f"test_invoke_queued_sequential_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),
        use_invoke_queue=True,
        debug=True
    )

    responses_a = []
    responses_b = []

    async def request_a():
        async for response in sts.invoke(STSRequest(
            session_id=session_id,
            user_id="test_user",
            text="Say 'Hello A'",
            wait_in_queue=True
        )):
            responses_a.append(response)

    async def request_b():
        async for response in sts.invoke(STSRequest(
            session_id=session_id,
            user_id="test_user",
            text="Say 'Hello B'",
            wait_in_queue=True
        )):
            responses_b.append(response)

    # Run both requests concurrently
    await asyncio.gather(request_a(), request_b())

    # Both should complete with responses
    assert len(responses_a) > 0
    assert len(responses_b) > 0

    # Check final responses exist
    final_a = [r for r in responses_a if r.type == "final"]
    final_b = [r for r in responses_b if r.type == "final"]
    assert len(final_a) == 1
    assert len(final_b) == 1

    await sts.shutdown()


@pytest.mark.asyncio
async def test_invoke_queued_immediately_cancels_pending():
    """Test that wait_in_queue=False cancels pending requests in queue"""
    session_id = f"test_invoke_queued_immediately_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),
        use_invoke_queue=True,
        debug=True
    )

    responses_a = []
    responses_b = []
    responses_c = []

    async def request_a():
        async for response in sts.invoke(STSRequest(
            session_id=session_id,
            user_id="test_user",
            text="Say 'Hello A' with a long explanation",
            wait_in_queue=True
        )):
            responses_a.append(response)

    async def request_b():
        # Small delay to ensure A starts first
        await asyncio.sleep(0.1)
        async for response in sts.invoke(STSRequest(
            session_id=session_id,
            user_id="test_user",
            text="Say 'Hello B'",
            wait_in_queue=True
        )):
            responses_b.append(response)

    async def request_c():
        # Wait a bit then send with wait_in_queue=False (interrupt)
        await asyncio.sleep(0.2)
        async for response in sts.invoke(STSRequest(
            session_id=session_id,
            user_id="test_user",
            text="Say 'Hello C'",
            wait_in_queue=False
        )):
            responses_c.append(response)

    await asyncio.gather(request_a(), request_b(), request_c())

    # A should complete (it was already processing)
    assert len([r for r in responses_a if r.type == "final"]) == 1

    # B should be cancelled (was in queue when C came with wait_in_queue=False)
    cancelled_b = [r for r in responses_b if r.type == "cancelled"]
    assert len(cancelled_b) == 1

    # C should complete
    assert len([r for r in responses_c if r.type == "final"]) == 1

    await sts.shutdown()


@pytest.mark.asyncio
async def test_invoke_queued_cleanup_on_idle():
    """Test that queue resources are cleaned up after idle timeout"""
    session_id = f"test_invoke_queued_cleanup_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        invoke_queue_idle_timeout=1.0,  # Short timeout for testing
        performance_recorder=SQLitePerformanceRecorder(),
        use_invoke_queue=True,
        debug=True
    )

    # Send a request
    async for response in sts.invoke(STSRequest(
        session_id=session_id,
        user_id="test_user",
        text="Hello"
    )):
        pass

    # Queue resources should exist
    assert session_id in sts._request_queues
    assert session_id in sts._invoke_workers

    # Wait for idle timeout + buffer
    await asyncio.sleep(1.5)

    # Queue resources should be cleaned up
    assert session_id not in sts._request_queues
    assert session_id not in sts._invoke_workers
    assert session_id not in sts._response_queues

    await sts.shutdown()


@pytest.mark.asyncio
async def test_invoke_queued_error_handling():
    """Test that errors in invoke are properly handled and returned"""
    session_id = f"test_invoke_queued_error_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key="invalid_key",  # This will cause an error
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),
        use_invoke_queue=True,
        debug=True
    )

    responses = []
    async for response in sts.invoke(STSRequest(
        session_id=session_id,
        user_id="test_user",
        text="Hello"
    )):
        responses.append(response)

    # Should have an error response
    error_responses = [r for r in responses if r.type == "error"]
    assert len(error_responses) >= 1

    await sts.shutdown()


@pytest.mark.asyncio
async def test_validate_request_cancel():
    """Test that validate_request cancels request when reason is returned"""
    session_id = f"test_validate_request_cancel_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    # Set up validator that rejects short text
    @sts.validate_request
    async def validate_request(request: STSRequest):
        if len(request.text) < 5:
            return "Text too short"
        return None

    responses = []
    async for response in sts.invoke(STSRequest(
        session_id=session_id,
        user_id="test_user",
        text="Hi"  # Too short, should be canceled
    )):
        responses.append(response)

    # Should have a canceled response
    canceled_responses = [r for r in responses if r.type == "canceled"]
    assert len(canceled_responses) == 1
    assert canceled_responses[0].metadata["reason"] == "Text too short"

    # Should not have final response (LLM was not called)
    final_responses = [r for r in responses if r.type == "final"]
    assert len(final_responses) == 0

    await sts.shutdown()


@pytest.mark.asyncio
async def test_validate_request_pass():
    """Test that validate_request allows request when None is returned"""
    session_id = f"test_validate_request_pass_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    # Set up validator that rejects short text
    @sts.validate_request
    async def validate_request(request: STSRequest):
        if len(request.text) < 5:
            return "Text too short"
        return None

    responses = []
    async for response in sts.invoke(STSRequest(
        session_id=session_id,
        user_id="test_user",
        text="Hello, this is a valid request"  # Long enough, should pass
    )):
        responses.append(response)

    # Should not have canceled response
    canceled_responses = [r for r in responses if r.type == "canceled"]
    assert len(canceled_responses) == 0

    # Should have final response
    final_responses = [r for r in responses if r.type == "final"]
    assert len(final_responses) == 1

    await sts.shutdown()


@pytest.mark.asyncio
async def test_validate_request_with_files():
    """Test that validate_request can access files in request"""
    session_id = f"test_validate_request_with_files_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),
        debug=True
    )

    # Set up validator that rejects requests with too many files
    @sts.validate_request
    async def validate_request(request: STSRequest):
        if request.files and len(request.files) > 2:
            return "Too many files"
        return None

    # Request with too many files
    responses = []
    async for response in sts.invoke(STSRequest(
        session_id=session_id,
        user_id="test_user",
        text="Check these files",
        files={"file1": "content1", "file2": "content2", "file3": "content3"}
    )):
        responses.append(response)

    # Should be canceled
    canceled_responses = [r for r in responses if r.type == "canceled"]
    assert len(canceled_responses) == 1
    assert canceled_responses[0].metadata["reason"] == "Too many files"

    await sts.shutdown()


@pytest.mark.asyncio
async def test_invoke_queued_multiple_sessions():
    """Test that different sessions have independent queues"""
    session_id_1 = f"test_invoke_queued_multi_1_{str(uuid4())}"
    session_id_2 = f"test_invoke_queued_multi_2_{str(uuid4())}"

    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(),
        llm=ChatGPTService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        ),
        tts=SpeechSynthesizerDummy(),
        performance_recorder=SQLitePerformanceRecorder(),
        use_invoke_queue=True,
        debug=True
    )

    responses_1 = []
    responses_2 = []

    async def session_1():
        async for response in sts.invoke(STSRequest(
            session_id=session_id_1,
            user_id="user_1",
            text="Hello from session 1"
        )):
            responses_1.append(response)

    async def session_2():
        async for response in sts.invoke(STSRequest(
            session_id=session_id_2,
            user_id="user_2",
            text="Hello from session 2"
        )):
            responses_2.append(response)

    # Run both sessions concurrently
    await asyncio.gather(session_1(), session_2())

    # Both should complete independently
    assert len([r for r in responses_1 if r.type == "final"]) == 1
    assert len([r for r in responses_2 if r.type == "final"]) == 1

    # Each session should have its own queue
    assert session_id_1 in sts._request_queues or session_id_1 not in sts._request_queues  # May be cleaned up
    assert session_id_2 in sts._request_queues or session_id_2 not in sts._request_queues

    await sts.shutdown()
