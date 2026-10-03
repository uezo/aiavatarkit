"""Continuous GPT-Live audio, independent of the VAD/STT/LLM pipeline."""

import asyncio
import base64
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import math
import os
from time import monotonic
from typing import Any, Callable

from websockets.asyncio.client import connect

from aiavatar.sts.models import STSRequest, STSResponse
from .streaming import StreamingSTSPipeline

logger = logging.getLogger(__name__)
event_logger = logging.getLogger(__name__ + ".events")


@dataclass
class _Session:
    session_id: str
    user_id: str | None
    context_id: str | None
    channel: str | None
    ready: asyncio.Future
    websocket: Any = None
    task: asyncio.Task | None = None
    finalizer: asyncio.Task | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started: bool = False
    closing: bool = False
    pending_audio: bytes = b""
    output_format_sent: bool = False
    transcript_role: str | None = None
    transcript_text: str = ""
    failure: str | None = None
    close_event: dict | None = None
    created_at: float = field(default_factory=monotonic)
    last_output_audio_at: float | None = None
    first_output_audio_at: float | None = None
    output_audio_bytes: int = 0


class OpenAILivePipeline(StreamingSTSPipeline):
    """Relay mono PCM16 at 16 or 24 kHz to one Live connection per session.

    Native PCM output carries pcm_format metadata; adapters must relay it as-is.
    Captions are display snapshots, grouped by speaker changes; they do not
    define turns or trigger synthesis. No per-utterance final event is emitted.

    output_audio_callback optionally diverts PCM to a synchronous, enqueue-only
    callback. Return processed mono PCM16 at input_sample_rate with
    send_output_audio(); omitting session_id broadcasts to active sessions.

    Responses delegation supports provider-managed tools. Client delegation and
    application function tools are intentionally outside this initial backend.
    """

    def __init__(
        self,
        *,
        openai_api_key: str = None,
        model: str = "gpt-live-1",
        instructions: str = None,
        voice: str = "marin",
        input_sample_rate: int = 16000,
        delegation: dict = None,
        startup_timeout: float = 15.0,
        close_timeout: float = 5.0,
        output_audio_callback: Callable[[bytes], None] | None = None,
        debug: bool = False,
    ):
        super().__init__()
        if type(input_sample_rate) is not int or input_sample_rate not in (16000, 24000):
            raise ValueError("input_sample_rate must be 16000 or 24000 (mono PCM16)")
        for name, value in (("startup_timeout", startup_timeout), ("close_timeout", close_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.delegation = deepcopy(delegation) if delegation is not None else {
            "type": "responses", "responses": {"model": "gpt-5.6-luna"},
        }
        if self.delegation.get("type") != "responses":
            raise ValueError("Only Responses delegation is supported")
        responses = self.delegation.get("responses", {})
        if not responses.get("model"):
            raise ValueError("delegation.responses.model is required")
        if any(tool.get("type") == "function" for tool in responses.get("tools", [])):
            raise ValueError("Application function tools are not supported by OpenAILivePipeline")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.instructions = instructions
        self.voice = voice
        self.input_sample_rate = input_sample_rate
        self.vad.sample_rate = input_sample_rate
        self.startup_timeout = startup_timeout
        self.close_timeout = close_timeout
        self.output_audio_callback = output_audio_callback
        self.debug = debug
        self._sessions: dict[str, _Session] = {}

    async def start_session(self, request: STSRequest):
        if not self.openai_api_key:
            raise ValueError("openai_api_key or OPENAI_API_KEY is required")
        if not request.session_id:
            raise ValueError("session_id is required")
        if request.session_id in self._sessions:
            raise ValueError("Live session is already active")
        rate = (request.metadata or {}).get("input_sample_rate", self.input_sample_rate)
        if type(rate) is not int or rate != self.input_sample_rate:
            raise ValueError(f"Microphone input_sample_rate must be {self.input_sample_rate}")
        session = _Session(
            session_id=request.session_id,
            user_id=request.user_id,
            context_id=request.context_id,
            channel=request.channel,
            ready=asyncio.get_running_loop().create_future(),
        )
        self._remember_session(request)
        self._sessions[session.session_id] = session
        session.task = asyncio.create_task(self._run_session(session))
        try:
            await asyncio.wait_for(asyncio.shield(session.ready), self.startup_timeout)
        except BaseException:
            # A canceled browser handshake must not leave a paid Live session running.
            session.ready.cancel()
            session.task.cancel()
            await asyncio.gather(session.task, return_exceptions=True)
            if self._sessions.get(session.session_id) is session:
                del self._sessions[session.session_id]
                self._release_session(session.session_id)
            raise

    async def _emit(self, session: _Session, type: str, *, metadata: dict = None, **kwargs):
        await self.handle_response(STSResponse(
            type=type,
            session_id=session.session_id,
            user_id=session.user_id,
            context_id=session.context_id,
            transaction_id=None,
            metadata={**({"channel": session.channel} if session.channel else {}), **(metadata or {})},
            **kwargs,
        ))

    async def _send(self, session: _Session, event: dict):
        async with session.send_lock:
            await session.websocket.send(json.dumps(event))
            self._log_event(session, event, direction="send")

    def _log_event(self, session: _Session, event: dict, *, direction: str = "recv"):
        """Trace receipt timing without logging audio, session config, or headers."""
        if not self.debug or not event_logger.isEnabledFor(logging.DEBUG):
            return
        now = monotonic()
        event_type = event.get("type")
        details = {}
        if event_type == "session.output_audio.delta":
            delta = event.get("delta", "")
            if isinstance(delta, str):
                size = len(delta) // 4 * 3 - (len(delta) - len(delta.rstrip("=")))
                details.update(audio_bytes=size, audio_ms=round(size / (2 * self.input_sample_rate) * 1000, 1))
                if size > 0:
                    if session.first_output_audio_at is None:
                        session.first_output_audio_at = now
                    session.output_audio_bytes += size
                if session.first_output_audio_at is not None:
                    # Compare differences between two receipts, not a session-wide
                    # average that includes waiting for the user or silent gaps.
                    details.update(
                        audio_total_ms=round(session.output_audio_bytes / (2 * self.input_sample_rate) * 1000, 1),
                        audio_elapsed_ms=round((now - session.first_output_audio_at) * 1000, 1),
                    )
            if session.last_output_audio_at is not None:
                details["interval_ms"] = round((now - session.last_output_audio_at) * 1000, 1)
            session.last_output_audio_at = now
        elif event_type in ("session.input_transcript.delta", "session.output_transcript.delta"):
            details = {key: event.get(key) for key in ("delta", "start_ms", "end_ms")}
            details["role"] = "user" if event_type == "session.input_transcript.delta" else "assistant"
            if session.last_output_audio_at is not None:
                details["since_audio_ms"] = round((now - session.last_output_audio_at) * 1000, 1)
        elif event_type == "response.event":
            if isinstance(event.get("event"), dict):
                details["backend_type"] = event["event"].get("type")
        elif event_type == "error":
            if isinstance(event.get("error"), dict):
                details["code"] = event["error"].get("code")
        event_logger.debug(
            "session=%r t=%.3fs %s %s %s", session.session_id,
            now - session.created_at, direction, event_type,
            json.dumps(details, ensure_ascii=False),
        )

    async def _report_error(self, session: _Session, message: str, **metadata):
        try:
            await self._emit(session, "error", metadata={"error": message, **metadata})
        except Exception:
            logger.warning("Live response handler failed while reporting an error")

    async def _run_session(self, session: _Session):
        config = {
            "model": self.model,
            "audio": {
                "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
                "output": {"voice": self.voice},
            },
            "delegation": deepcopy(self.delegation),
        }
        if self.instructions is not None:
            config["instructions"] = self.instructions
        try:
            async with connect(
                "wss://api.openai.com/v1/live/sessions",
                additional_headers={"Authorization": f"Bearer {self.openai_api_key}"},
                open_timeout=self.startup_timeout,
                close_timeout=self.close_timeout,
            ) as websocket:
                session.websocket = websocket
                await self._send(session, {"type": "session.start", "session": config})
                async for message in websocket:
                    event = json.loads(message)
                    self._log_event(session, event)
                    event_type = event["type"]
                    if event_type == "session.closed":
                        session.close_event = event
                        break
                    if event_type == "error":
                        # Provider errors can contain echoed client input. Forward the
                        # error code only; never expose credentials or session config.
                        code = event.get("error", {}).get("code", "live_error")
                        await self._report_error(session, "GPT-Live error", code=code)
                        if not session.started:
                            raise RuntimeError("GPT-Live rejected session startup")
                        continue
                    if event_type == "session.started":
                        if session.started:
                            continue
                        audio_format = event["session"].get("audio", {}).get("format", {})
                        if audio_format != config["audio"]["format"]:
                            raise RuntimeError("GPT-Live returned an unexpected audio format")
                        session.started = True
                        await self._session_connected(STSResponse(
                            type="connected", session_id=session.session_id,
                            user_id=session.user_id, context_id=session.context_id,
                            metadata={**({"channel": session.channel} if session.channel else {}),
                                      "input_sample_rate": self.input_sample_rate},
                        ), session.ready)
                        continue
                    if not session.started:
                        raise RuntimeError("GPT-Live sent output before session.started")
                    if not session.closing:
                        await self._receive_event(session, event)
                if session.close_event is None and session.failure is None:
                    session.failure = "GPT-Live connection closed without session.closed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Avoid logging raw provider/transport exceptions (they may echo keys).
            if self.debug:
                event_logger.debug("session=%r connection_failed exception=%s", session.session_id, type(exc).__name__)
            session.failure = session.failure or "GPT-Live connection failed"
            if session.started:
                await self._report_error(session, session.failure)
        finally:
            session.closing = True
            try:
                if not session.ready.done():
                    session.ready.set_exception(RuntimeError(session.failure or "GPT-Live startup was interrupted"))
                if session.started:
                    terminal = session.close_event or {}
                    try:
                        await self.stop_response(session.session_id, session.context_id)
                    except Exception:
                        logger.warning("Live response handler failed while stopping playback")
                    try:
                        await self._emit(session, "session_closed", metadata={
                            "reason": terminal.get("reason", session.failure or "connection_closed"),
                            "finalized": session.close_event is not None,
                            **({"usage": terminal["usage"]} if "usage" in terminal else {}),
                        })
                    except Exception:
                        logger.warning("Live response handler failed while closing the session")
            finally:
                if self.debug:
                    event_logger.debug(
                        "session=%r session_ended finalized=%s failure=%r",
                        session.session_id, session.close_event is not None, session.failure,
                    )
                if self._sessions.get(session.session_id) is session:
                    del self._sessions[session.session_id]
                    self._release_session(session.session_id)

    async def _receive_event(self, session: _Session, event: dict):
        event_type = event["type"]
        if event_type == "session.output_audio.delta":
            audio = base64.b64decode(event["delta"], validate=True)
            if len(audio) % 2:
                raise ValueError("GPT-Live PCM output must contain complete 16-bit samples")
            if audio:
                if self.output_audio_callback is not None:
                    self.output_audio_callback(audio)
                else:
                    await self.send_output_audio(audio, session.session_id)
        elif event_type in ("session.input_transcript.delta", "session.output_transcript.delta"):
            role = "user" if event_type == "session.input_transcript.delta" else "assistant"
            text = event["delta"]
            if role != session.transcript_role:
                session.transcript_role = role
                session.transcript_text = ""
            session.transcript_text += text
            if session.transcript_text.strip():
                # Display snapshots do not define voice turns or carry controls.
                key = "partial_request_text" if role == "user" else "partial_response_text"
                await self._emit(session, "info", metadata={
                    "role": role, "start_ms": event["start_ms"], "end_ms": event["end_ms"],
                    key: session.transcript_text,
                })
        elif event_type == "response.event":
            # Delegated response.done is not a voice turn completion.
            await self._emit(session, "backend_event", metadata={"event": event})
        elif event_type == "session.usage.updated":
            await self._emit(session, "usage", metadata={"usage": event.get("usage")})

    async def send_output_audio(self, audio: bytes, session_id: str | None = None):
        """Deliver PCM16 at input_sample_rate, bypassing output_audio_callback.

        Without session_id, share the same audio with all active sessions. This
        deliberately provides no audio isolation for a shared external device.
        """
        if len(audio) % 2:
            raise ValueError("Output PCM must contain complete 16-bit samples")
        if not audio:
            return
        sessions = list(self._sessions.values()) if session_id is None else [self._sessions.get(session_id)]
        metadata = {"pcm_format": {"sample_rate": self.input_sample_rate, "channels": 1, "sample_width": 2}}
        for session in sessions:
            if (session is None or not session.started or not session.ready.done() or session.ready.cancelled()
                    or session.closing or session.finalizer is not None):
                continue
            gate = self._prepared_sessions.get(session.session_id)
            if gate is not None and not gate.is_set():
                continue
            try:
                if not session.output_format_sent:
                    # Native clients initialize their player before consuming PCM.
                    await self._emit(session, "chunk", metadata=metadata)
                    session.output_format_sent = True
                if (self._sessions.get(session.session_id) is session
                        and not session.closing and session.finalizer is None):
                    await self._emit(session, "chunk", audio_data=audio, metadata=metadata)
            except Exception:
                if session_id is not None:
                    raise
                logger.warning("Could not deliver shared Live audio to session %s", session.session_id)

    async def process_audio_samples(self, samples: bytes, context_id: str):
        """Send continuous raw PCM. ``context_id`` is the adapter's session ID."""
        session = self._sessions.get(context_id)
        if session is None or session.closing or not session.started:
            raise RuntimeError("Live session is not ready")
        try:
            async with session.send_lock:
                if session.closing:
                    return
                audio = session.pending_audio + samples
                complete = len(audio) - len(audio) % 2
                session.pending_audio = audio[complete:]
                if complete:
                    await session.websocket.send(json.dumps({
                        "type": "session.input_audio.append",
                        "audio": base64.b64encode(audio[:complete]).decode("ascii"),
                    }))
        except Exception:
            session.failure = "GPT-Live audio send failed"
            if self._sessions.get(context_id) is session:
                await self.finalize(context_id)
            raise RuntimeError(session.failure) from None

    async def finalize(self, session_id: str):
        session = self._sessions.get(session_id)
        if session is None:
            self._release_session(session_id)
            return
        if session.finalizer is None:
            session.finalizer = asyncio.create_task(self._finalize_session(session))
        # A response callback cannot await teardown of the worker running it.
        # A canceled adapter request must still release the connection.
        if asyncio.current_task() is not session.task:
            await asyncio.shield(session.finalizer)

    async def _finalize_session(self, session: _Session):
        if not session.closing:
            session.closing = True
            # Unblock a prepared receiver so it can consume the close ack.
            self.activate_session(session.session_id)
            if session.started and session.websocket is not None:
                try:
                    await asyncio.wait_for(self._send(session, {"type": "session.close"}), self.close_timeout)
                except Exception:
                    session.task.cancel()
            else:
                session.task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(session.task, return_exceptions=True)),
                self.close_timeout,
            )
        except TimeoutError:
            session.failure = "Timed out waiting for GPT-Live session.closed"
            session.task.cancel()
        except asyncio.CancelledError:
            session.task.cancel()
            raise
        finally:
            await asyncio.gather(session.task, return_exceptions=True)

    async def shutdown(self):
        await asyncio.gather(*(self.finalize(session_id) for session_id in list(self._sessions)))
