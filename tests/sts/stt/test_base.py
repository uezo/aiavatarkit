import pytest
from aiavatar.sts.stt.base import SpeechRecognizer, SpeechRecognitionResult


class MockSpeechRecognizer(SpeechRecognizer):
    async def transcribe(self, data: bytes) -> str:
        return "test transcription"


@pytest.mark.asyncio
async def test_recognize_with_preprocess_postprocess():
    recognizer = MockSpeechRecognizer()
    
    # Define preprocess function that returns tuple
    @recognizer.preprocess
    async def custom_preprocess(session_id: str, data: bytes):
        processed_data = data + b"_processed"
        metadata = {"session_id": session_id, "original_size": len(data)}
        return processed_data, metadata
    
    # Define postprocess function that returns tuple
    @recognizer.postprocess
    async def custom_postprocess(session_id: str, text: str, data: bytes, preprocess_metadata: dict):
        modified_text = text.upper()
        metadata = {
            "session_id": session_id,
            "original_text": text,
            "data_size": len(data),
            "preprocess_info": preprocess_metadata
        }
        return modified_text, metadata
    
    # Test recognize method
    test_session_id = "test-session-123"
    test_data = b"test audio data"
    
    result = await recognizer.recognize(test_session_id, test_data)
    
    # Assertions
    assert isinstance(result, SpeechRecognitionResult)
    assert result.text == "TEST TRANSCRIPTION"
    assert result.preprocess_metadata == {
        "session_id": test_session_id,
        "original_size": len(test_data)
    }
    assert result.postprocess_metadata == {
        "session_id": test_session_id,
        "original_text": "test transcription",
        "data_size": len(test_data + b"_processed"),
        "preprocess_info": result.preprocess_metadata
    }
    
    await recognizer.close()


@pytest.mark.asyncio
async def test_recognize_with_simple_preprocess_postprocess():
    recognizer = MockSpeechRecognizer()
    
    # Define preprocess function that returns only bytes
    @recognizer.preprocess
    async def simple_preprocess(session_id: str, data: bytes):
        return data + b"_simple"
    
    # Define postprocess function that returns only string
    @recognizer.postprocess
    async def simple_postprocess(session_id: str, text: str, data: bytes, preprocess_metadata: dict):
        return f"[{session_id}] {text}"
    
    # Test recognize method
    test_session_id = "simple-session"
    test_data = b"simple data"
    
    result = await recognizer.recognize(test_session_id, test_data)
    
    # Assertions
    assert isinstance(result, SpeechRecognitionResult)
    assert result.text == "[simple-session] test transcription"
    assert result.preprocess_metadata is None
    assert result.postprocess_metadata is None
    
    await recognizer.close()


@pytest.mark.asyncio
async def test_recognize_with_empty_preprocess_result():
    recognizer = MockSpeechRecognizer()
    
    # Define preprocess that returns empty bytes
    @recognizer.preprocess
    async def empty_preprocess(session_id: str, data: bytes):
        return b""
    
    # Test recognize method
    test_session_id = "empty-session"
    test_data = b"some data"
    
    result = await recognizer.recognize(test_session_id, test_data)
    
    # Assertions - should return early without transcription
    assert isinstance(result, SpeechRecognitionResult)
    assert result.text is None
    assert result.preprocess_metadata is None
    assert result.postprocess_metadata is None
    
    await recognizer.close()


@pytest.mark.asyncio
async def test_recognize_without_custom_processors():
    recognizer = MockSpeechRecognizer()
    
    # Test recognize method without custom pre/post processors
    test_session_id = "default-session"
    test_data = b"default data"
    
    result = await recognizer.recognize(test_session_id, test_data)
    
    # Assertions - should use default processors
    assert isinstance(result, SpeechRecognitionResult)
    assert result.text == "test transcription"
    assert result.preprocess_metadata is None
    assert result.postprocess_metadata is None
    
    await recognizer.close()


@pytest.mark.asyncio
async def test_recognize_with_mixed_return_types():
    recognizer = MockSpeechRecognizer()
    
    # Preprocess returns tuple, postprocess returns string only
    @recognizer.preprocess
    async def mixed_preprocess(session_id: str, data: bytes):
        return data, {"preprocessed": True}
    
    @recognizer.postprocess
    async def mixed_postprocess(session_id: str, text: str, data: bytes, preprocess_metadata: dict):
        return text + " (processed)"
    
    # Test recognize method
    test_session_id = "mixed-session"
    test_data = b"mixed data"
    
    result = await recognizer.recognize(test_session_id, test_data)
    
    # Assertions
    assert isinstance(result, SpeechRecognitionResult)
    assert result.text == "test transcription (processed)"
    assert result.preprocess_metadata == {"preprocessed": True}
    assert result.postprocess_metadata is None
    
    await recognizer.close()
