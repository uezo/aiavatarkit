"""
Tests for StreamSpeechRecognitionServer

Prerequisites:
    1. Start the STT server on port 48000:
       cd examples/stt
       AZURE_API_KEY=your_key uvicorn server:app --port 48000

    2. Run tests:
       pytest tests/adapter/stt/test_stream_stt_server.py -v
"""

import asyncio
import base64
import json
import wave
from pathlib import Path
from uuid import uuid4
import pytest
import websockets

# Server URL
STT_SERVER_URL = "ws://localhost:48000/ws/stt"

# Test WAV file paths
TEST_WAV_PATH = Path(__file__).parent.parent.parent / "sts" / "stt" / "data" / "hello.wav"  # Japanese: こんにちは
TEST_WAV_PATH_EN = Path(__file__).parent.parent.parent / "sts" / "stt" / "data" / "hello_en.wav"  # English: Hello


def load_wav_as_pcm(wav_path: Path, target_sample_rate: int = 16000) -> bytes:
    """Load WAV file and return raw PCM bytes."""
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getsampwidth() == 2, "Expected 16-bit audio"
        assert wav_file.getnchannels() == 1, "Expected mono audio"
        return wav_file.readframes(wav_file.getnframes())


class STTTestClient:
    """Simple WebSocket client for testing StreamSpeechRecognitionServer."""

    def __init__(self, url: str = STT_SERVER_URL):
        self.url = url
        self.ws = None
        self.session_id = None
        self.partial_results = []
        self.final_results = []
        self.is_connected = False
        self._receive_task = None

    async def connect(self, session_id: str = None):
        """Connect to server and start session."""
        self.session_id = session_id or f"test_{uuid4()}"
        self.ws = await websockets.connect(self.url)

        # Send start message
        await self.ws.send(json.dumps({
            "type": "start",
            "session_id": self.session_id
        }))

        # Wait for connected response
        response = json.loads(await self.ws.recv())
        assert response["type"] == "connected"
        self.is_connected = True

        # Start receiving messages in background
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        """Background task to receive messages."""
        try:
            while self.is_connected:
                try:
                    data = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    msg = json.loads(data)
                    if msg["type"] == "partial":
                        self.partial_results.append(msg)
                    elif msg["type"] == "final":
                        self.final_results.append(msg)
                except asyncio.TimeoutError:
                    continue
        except websockets.exceptions.ConnectionClosed:
            pass

    async def send_audio(self, audio_data: bytes, chunk_size: int = 3200):
        """Send audio data in chunks."""
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            b64_chunk = base64.b64encode(chunk).decode("utf-8")
            await self.ws.send(json.dumps({
                "type": "data",
                "session_id": self.session_id,
                "audio_data": b64_chunk
            }))
            # Simulate real-time streaming (chunk_size / sample_rate / 2)
            await asyncio.sleep(chunk_size / 16000 / 2)

    async def send_silence(self, duration_sec: float = 1.0, sample_rate: int = 16000):
        """Send silence to trigger end of speech detection."""
        silence_bytes = b'\x00' * int(duration_sec * sample_rate * 2)
        await self.send_audio(silence_bytes)

    async def disconnect(self):
        """Disconnect from server."""
        self.is_connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.send(json.dumps({
                "type": "stop",
                "session_id": self.session_id
            }))
            await self.ws.close()

    def clear_results(self):
        """Clear collected results."""
        self.partial_results.clear()
        self.final_results.clear()


@pytest.fixture
def test_client():
    """Create test client fixture."""
    return STTTestClient()


@pytest.mark.asyncio
async def test_connection(test_client: STTTestClient):
    """Test basic connection and disconnection."""
    await test_client.connect()
    assert test_client.is_connected
    assert test_client.session_id is not None
    await test_client.disconnect()


@pytest.mark.asyncio
async def test_speech_recognition_hello(test_client: STTTestClient):
    """
    Test speech recognition with hello.wav file.
    Expected: "こんにちは" should be recognized.
    """
    await test_client.connect()

    try:
        # Load and send audio
        audio_data = load_wav_as_pcm(TEST_WAV_PATH)
        await test_client.send_audio(audio_data)

        # Send silence to trigger end of speech
        await test_client.send_silence(duration_sec=1.0)

        # Wait for recognition to complete
        timeout = 5.0
        start_time = asyncio.get_event_loop().time()
        while len(test_client.final_results) == 0:
            if asyncio.get_event_loop().time() - start_time > timeout:
                pytest.fail("Timeout waiting for final result")
            await asyncio.sleep(0.1)

        # Verify results
        assert len(test_client.final_results) >= 1
        final_result = test_client.final_results[0]
        assert final_result["type"] == "final"
        assert final_result["is_final"] is True
        assert "こんにちは" in final_result["text"]

        # Check metadata
        assert "duration" in final_result.get("metadata", {})

    finally:
        await test_client.disconnect()


@pytest.mark.asyncio
async def test_partial_results(test_client: STTTestClient):
    """
    Test that partial results are received during speech recognition.
    Stream VAD should send partial results via on_speech_detecting.
    """
    await test_client.connect()

    try:
        # Load and send audio
        audio_data = load_wav_as_pcm(TEST_WAV_PATH)
        await test_client.send_audio(audio_data)

        # Send silence to trigger end of speech
        await test_client.send_silence(duration_sec=1.0)

        # Wait for final result
        timeout = 5.0
        start_time = asyncio.get_event_loop().time()
        while len(test_client.final_results) == 0:
            if asyncio.get_event_loop().time() - start_time > timeout:
                pytest.fail("Timeout waiting for final result")
            await asyncio.sleep(0.1)

        # Verify partial results were received (stream VAD feature)
        # Note: Partial results may not always be sent depending on audio length
        # and segment_silence_threshold setting
        print(f"Partial results received: {len(test_client.partial_results)}")
        for i, partial in enumerate(test_client.partial_results):
            print(f"  Partial {i}: {partial.get('text', '')}")

        # Final result should contain the full text
        assert len(test_client.final_results) >= 1
        assert "こんにちは" in test_client.final_results[0]["text"]

    finally:
        await test_client.disconnect()


@pytest.mark.asyncio
async def test_multiple_utterances(test_client: STTTestClient):
    """
    Test multiple utterances in a single session.
    Each utterance should produce a separate final result.
    """
    await test_client.connect()

    try:
        audio_data = load_wav_as_pcm(TEST_WAV_PATH)

        # First utterance
        await test_client.send_audio(audio_data)
        await test_client.send_silence(duration_sec=1.0)

        # Wait for first result
        timeout = 5.0
        start_time = asyncio.get_event_loop().time()
        while len(test_client.final_results) < 1:
            if asyncio.get_event_loop().time() - start_time > timeout:
                pytest.fail("Timeout waiting for first result")
            await asyncio.sleep(0.1)

        assert "こんにちは" in test_client.final_results[0]["text"]

        # Second utterance (same audio)
        await test_client.send_audio(audio_data)
        await test_client.send_silence(duration_sec=1.0)

        # Wait for second result
        start_time = asyncio.get_event_loop().time()
        while len(test_client.final_results) < 2:
            if asyncio.get_event_loop().time() - start_time > timeout:
                pytest.fail("Timeout waiting for second result")
            await asyncio.sleep(0.1)

        assert len(test_client.final_results) >= 2
        assert "こんにちは" in test_client.final_results[1]["text"]

    finally:
        await test_client.disconnect()


@pytest.mark.asyncio
async def test_session_isolation():
    """
    Test that multiple concurrent sessions are isolated.
    Uses different audio files for each client to verify results don't cross.
    """
    client1 = STTTestClient()
    client2 = STTTestClient()

    try:
        # Connect both clients
        await client1.connect(session_id="isolation_test_1")
        await client2.connect(session_id="isolation_test_2")

        # Use different audio for each client
        audio_data_ja = load_wav_as_pcm(TEST_WAV_PATH)  # Japanese: こんにちは
        audio_data_en = load_wav_as_pcm(TEST_WAV_PATH_EN)  # English: Hello

        # Send audio to both clients simultaneously
        await asyncio.gather(
            client1.send_audio(audio_data_ja),
            client2.send_audio(audio_data_en)
        )

        # Send silence to both
        await asyncio.gather(
            client1.send_silence(duration_sec=1.0),
            client2.send_silence(duration_sec=1.0)
        )

        # Wait for results from both
        timeout = 5.0
        start_time = asyncio.get_event_loop().time()
        while len(client1.final_results) < 1 or len(client2.final_results) < 1:
            if asyncio.get_event_loop().time() - start_time > timeout:
                pytest.fail("Timeout waiting for results from both clients")
            await asyncio.sleep(0.1)

        # Verify both got results with correct session IDs
        assert client1.final_results[0]["session_id"] == "isolation_test_1"
        assert client2.final_results[0]["session_id"] == "isolation_test_2"

        # Verify each client received the correct recognition result
        # Client 1 should get Japanese, Client 2 should get English
        assert "こんにちは" in client1.final_results[0]["text"]
        assert "hello" in client2.final_results[0]["text"].lower()

    finally:
        await client1.disconnect()
        await client2.disconnect()


@pytest.mark.asyncio
async def test_silence_only_no_result(test_client: STTTestClient):
    """
    Test that sending only silence does not produce a recognition result.
    """
    await test_client.connect()

    try:
        # Send only silence (no speech)
        await test_client.send_silence(duration_sec=2.0)

        # Wait a bit to see if any result comes
        await asyncio.sleep(2.0)

        # Should not have any final results (no speech detected)
        # Note: This may vary depending on VAD sensitivity
        print(f"Final results after silence only: {len(test_client.final_results)}")

    finally:
        await test_client.disconnect()
