import asyncio
import os
import pytest
import numpy
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
    await adapter.handle_request(STSRequest(user_id="aiavatar.sts_user", audio_data=await get_input_voice("日本の首都は？")))
    context_id = adapter.final_context_id

    # Check output voice audio
    final_audio = adapter.final_audio
    assert len(final_audio) > 0, "No final audio was captured by the response handler."
    output_text = await get_output_text(final_audio)
    assert "東京" in output_text, f"Expected '東京' in recognized text, but got: {output_text}"

    # Invoke pipeline with the successive request (Ask about of US, without using the word 'capital' to check context)
    await adapter.handle_request(STSRequest(context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("アメリカは？")))

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
        user_id="aiavatar.sts_user", audio_data=await get_input_voice("もしもし")
    ))
    context_id = adapter.final_context_id

    # Check no voice generated
    assert adapter.final_audio is b""
    assert context_id is None

    # Second request with wakeword: invoked
    await adapter.handle_request(STSRequest(
        context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("やあ、こんにちは！ところで日本の首都は？")
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
        context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("アメリカは？")
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
        context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("じゃあフランスは？")
    ))

    # Check no voice generated
    assert adapter.final_audio is b""
    assert adapter.final_context_id == context_id   # Wakeword timeout but context is still alive

    # Second request with wakeword: invoked
    await adapter.handle_request(STSRequest(
        context_id=context_id, user_id="aiavatar.sts_user", audio_data=await get_input_voice("こんにちは。ドイツはどうだろうか？")
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

    async for response in sts.invoke(STSRequest(context_id="test_pipeline_novoice", user_id="aiavatar.sts_user", text="こんにちは")):
        if response.type == "chunk" or response.type == "final":
            assert response.text is not None and response.text != ""
            assert response.voice_text is not None and response.voice_text != ""
            assert response.audio_data is None


@pytest.mark.asyncio
async def test_sts_pipeline_with_user():
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

    session_id = "test_sts_pipeline_with_user_session"
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
