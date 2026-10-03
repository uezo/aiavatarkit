"""OpenAI Realtime transport, session handshake, and history restoration."""

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
import json
from urllib.parse import urlencode
from uuid import uuid4

from websockets.asyncio.client import connect


class ProviderError(RuntimeError):
    """A provider error without its potentially private diagnostic payload."""

    def __init__(self, code: str):
        super().__init__("OpenAI Realtime error")
        self.code = code


def make_session_config(
    *,
    turn_detection: dict,
    transcription_model: str | None = None,
    language: str | None = None,
    instructions: str | None = None,
) -> dict:
    audio_input = {
        "format": {"type": "audio/pcm", "rate": 24000},
        "turn_detection": deepcopy(turn_detection),
    }
    if transcription_model:
        audio_input["transcription"] = {"model": transcription_model}
        if language:
            if transcription_model in ("gpt-transcribe", "gpt-live-transcribe"):
                audio_input["transcription"]["languages"] = [language]
            else:
                audio_input["transcription"]["language"] = language
    config = {"type": "realtime", "output_modalities": ["text"], "audio": {"input": audio_input}}
    if instructions is not None:
        config["instructions"] = instructions
    return config


class _Writer:
    """Frame writes while the connection's send lock is already held."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def send(self, event: dict):
        await self.websocket.send(json.dumps(event))

    async def send_audio(self, audio: bytes):
        await self.send({
            "type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode("ascii"),
        })


class RealtimeConnection:
    def __init__(self, *, model: str, api_key: str, startup_timeout: float, close_timeout: float):
        self.model = model
        self.api_key = api_key
        self.startup_timeout = startup_timeout
        self.close_timeout = close_timeout
        self.websocket = None
        self._transport = None
        self._reader = None
        self._send_lock = asyncio.Lock()

    async def __aenter__(self):
        self._transport = connect(
            "wss://api.openai.com/v1/realtime?" + urlencode({"model": self.model}),
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=self.startup_timeout,
            close_timeout=self.close_timeout,
        )
        self.websocket = await self._transport.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return await self._transport.__aexit__(exc_type, exc, traceback)

    @asynccontextmanager
    async def writer(self):
        """Serialize a complete write operation, including caller-side PCM work."""
        async with self._send_lock:
            yield _Writer(self.websocket)

    async def send(self, event: dict):
        async with self.writer() as writer:
            await writer.send(event)

    async def close(self):
        await self.websocket.close()

    async def _receive(self) -> dict:
        while True:
            event = json.loads(await anext(self._reader))
            if event["type"] == "session.created":
                continue
            if event["type"] == "error":
                code = event.get("error", {}).get("code", "realtime_error")
                # A cancellation can race with the provider finishing.
                if code == "response_cancel_not_active":
                    continue
                raise ProviderError(code)
            return event

    async def configure(self, config: dict, history_messages: list[dict]) -> bool:
        """Wait for the requested format and every restored message's acknowledgement."""
        await self.send({"type": "session.update", "session": config})
        self._reader = self.websocket.__aiter__()
        configured = False
        pending_history_items = set()
        while True:
            try:
                event = await self._receive()
            except StopAsyncIteration:
                return False
            if event["type"] == "session.updated":
                if configured:
                    continue
                accepted = event["session"]
                if (accepted.get("output_modalities") != ["text"] or
                        accepted.get("audio", {}).get("input", {}).get("format") != {"type": "audio/pcm", "rate": 24000}):
                    raise RuntimeError("Realtime returned an unexpected audio or output format")
                configured = True
                previous_item_id = "root"
                for message in history_messages:
                    item = {
                        "id": uuid4().hex, "type": "message", "role": message["role"],
                        "content": [{
                            "type": "input_text" if message["role"] == "user" else "output_text",
                            "text": message["content"],
                        }],
                    }
                    pending_history_items.add(item["id"])
                    await self.send({
                        "type": "conversation.item.create", "event_id": "restore_" + uuid4().hex,
                        "previous_item_id": previous_item_id, "item": item,
                    })
                    previous_item_id = item["id"]
                history_messages.clear()
                if not pending_history_items:
                    return True
            elif configured and event["type"] in (
                "conversation.item.added", "conversation.item.created", "conversation.item.done",
            ):
                item_id = (event.get("item") or {}).get("id")
                if item_id in pending_history_items:
                    pending_history_items.remove(item_id)
                    if not pending_history_items:
                        return True
            else:
                raise RuntimeError("Realtime sent output before the session was ready")

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict:
        while True:
            event = await self._receive()
            if event["type"] != "session.updated":
                return event
