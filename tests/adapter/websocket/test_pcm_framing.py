"""In-memory WAV/PCM framing; no providers, models, services or files."""

import base64
import io
import json
import wave

import pytest

from aiavatar.adapter.models import AIAvatarResponse, AvatarControlRequest, ControlTag
from aiavatar.adapter.websocket.server import (
    AIAvatarWebSocketServer, WebSocketSessionData, iter_audio_responses,
)
from aiavatar.sts.models import STSResponse


def make_wav(pcm_data, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


@pytest.mark.parametrize("channels", [1, 2])
def test_pcm_descriptor_preserves_controls_and_identifies_all_frames(channels):
    pcm_data = b"\x00\x01" * (11 * channels)
    response = AIAvatarResponse(
        type="chunk", session_id="s", user_id="u", context_id="c", transaction_id="t",
        text="Hello", voice_text="Hello!", language="en",
        control_tags=[ControlTag(name="face", attributes={"name": "happy"})],
        avatar_control_request=AvatarControlRequest(face_name="happy"),
        structured_content={"example": True}, metadata={"original": True},
    )
    descriptor, *chunks = list(iter_audio_responses(response, make_wav(pcm_data, channels), 8))

    assert descriptor.audio_data is None
    assert descriptor.metadata["audio_frame_count"] == 11
    assert descriptor.metadata["audio_id"]
    assert descriptor.metadata["original"] is True
    assert descriptor.metadata["pcm_format"] == {
        "sample_rate": 24000, "channels": channels, "sample_width": 2,
    }
    for field in ("text", "voice_text", "language", "control_tags",
                  "avatar_control_request", "structured_content"):
        assert getattr(descriptor, field) == getattr(response, field)
    assert response.metadata == {"original": True}
    assert b"".join(base64.b64decode(chunk.audio_data) for chunk in chunks) == pcm_data
    assert all(chunk.metadata == {
        "audio_id": descriptor.metadata["audio_id"],
        "pcm_format": descriptor.metadata["pcm_format"],
    } for chunk in chunks)
    assert all(chunk.text is None and chunk.control_tags is None for chunk in chunks)
    assert all(chunk.session_id == "s" and chunk.user_id == "u" and chunk.context_id == "c"
               and chunk.transaction_id == "t" for chunk in chunks)
    assert all(len(base64.b64decode(chunk.audio_data)) % (channels * 2) == 0 for chunk in chunks)
    second_descriptor = next(iter_audio_responses(response, make_wav(pcm_data, channels), 8))
    assert second_descriptor.metadata["audio_id"] != descriptor.metadata["audio_id"]


def test_complete_audio_mode_preserves_wav_and_metadata():
    wav_data = make_wav(b"\x01\x02" * 11)
    response = AIAvatarResponse(type="chunk", text="hello", metadata={"nod": True})
    message, = list(iter_audio_responses(response, wav_data, 0))
    assert base64.b64decode(message.audio_data) == wav_data
    assert message.text == "hello"
    assert message.metadata == {"nod": True}
    assert response.audio_data is None


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [None, "transaction", "disconnect"])
async def test_server_streams_frames_and_stops_after_interruption(interrupt):
    server = AIAvatarWebSocketServer.__new__(AIAvatarWebSocketServer)
    connection = WebSocketSessionData()
    connection.id = "s"
    connection.active_transaction_id = "t"
    messages = []

    class Socket:
        async def send_text(self, text):
            messages.append(json.loads(text))
            if len(messages) == 2:
                if interrupt == "transaction":
                    connection.active_transaction_id = "next"
                elif interrupt == "disconnect":
                    server.sessions.pop("s")
                    server.websockets.pop("s")

    server.sessions = {"s": connection}
    server.websockets = {"s": Socket()}
    server.response_audio_chunk_size = 8
    server._on_response_handlers = []
    server.control_tag_pattern = r'\[{tag}:(\w+)\]|<{tag}\s[^>]*{attr}=["\'](\w+)["\']'
    server.debug = False
    await server.handle_response(STSResponse(
        type="chunk", session_id="s", transaction_id="t", text="hello", voice_text="hello",
        audio_data=make_wav(b"\x01\x02" * 11), metadata={},
    ))
    assert len(messages) == (4 if interrupt is None else 2)
    assert messages[0]["metadata"]["audio_frame_count"] == 11
    assert all(message["metadata"]["audio_id"] == messages[0]["metadata"]["audio_id"]
               for message in messages)
