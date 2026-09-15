"""Connect a stream VAD/WebSocket pipeline to Nod."""

import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import uuid4

from ..session import NodSession


@dataclass
class _Conversation:
    connection: object
    nod: NodSession
    recording_id: str | None = None
    recordings: OrderedDict = field(default_factory=OrderedDict)
    transaction_id: str | None = None
    response_text: str = ""
    pending: asyncio.Task | None = None
    tasks: set = field(default_factory=set)

    def remember(self, recording_id):
        self.recordings[recording_id] = True
        while len(self.recordings) > 32:
            self.recordings.popitem(last=False)


class NodPipelineBridge:
    """Own per-connection history and tasks; register event hooks by default.

    Set auto_register_hooks=False to wire the event methods manually.
    Automatic registration replaces the app's single on_disconnect callback;
    later on_disconnect registration replaces Nod's cleanup callback. Use manual
    wiring to combine an existing disconnect callback with close_session.
    The app and engine remain caller-owned. Call prepare_audio at startup for
    the default WAV sender, or supply an async emit(session_id, decision).
    Custom emitters must recheck is_current after acquiring their send lock.
    """

    def __init__(self, app, engine, *, emit=None, timeout=1.5, min_interval=2.0,
                 history_limit=10, log_inputs=False, auto_register_hooks=True):
        self.app, self.engine = app, engine
        self.emit = emit
        self._options = dict(timeout=timeout, min_interval=min_interval,
                             history_limit=history_limit, log_inputs=log_inputs)
        self._sessions = {}
        self._audio = {}
        self._closed = False
        if auto_register_hooks:
            self._register_hooks()

    def _register_hooks(self):
        @self.app.on_session_start
        async def on_session_start(request, session):
            await self.open_session(session)

        @self.app.sts.vad.on_speech_detecting
        async def on_partial(text, session):
            self.on_partial(text, session)

        @self.app.sts.on_accepted
        async def on_accepted(request):
            self.on_accepted(request)

        @self.app.on_response
        async def on_response(response, sts_response):
            self.on_response(response, sts_response)

        @self.app.on_disconnect
        async def on_disconnect(session):
            await self.close_session(session)

    async def prepare_audio(self):
        """Cache all candidate audio once; the decision path does no TTS."""
        audio = {}
        for candidate in self.engine.candidates:
            data = await self.app.sts.tts.synthesize(candidate["phrase"])
            if not data:
                raise ValueError(f"No nod audio for {candidate['id']}")
            audio[candidate["id"]] = base64.b64encode(data).decode("ascii")
        self._audio = audio

    async def open_session(self, connection):
        if self._closed:
            raise RuntimeError("NodPipelineBridge is closed")
        previous = self._sessions.get(connection.id)
        if previous:
            if previous.connection is connection:
                return
            await self.close_session(previous.connection)
        async def emit(decision):
            return await self._emit(connection.id, decision)
        self._sessions[connection.id] = _Conversation(
            connection, NodSession(self.engine, emit, **self._options))

    def _session(self, session_id):
        state = self._sessions.get(session_id)
        if (self._closed or state is None
                or self.app.sessions.get(session_id) is not state.connection):
            return None
        return state

    def on_partial(self, text, recording):
        """Consume cumulative text and schedule a decision without awaiting it."""
        state = self._session(recording.session_id)
        recording_id = recording.recording_id
        if state is None or not recording_id:
            return
        previous = state.pending
        if recording_id != state.recording_id:
            # A late update must not reopen an input already accepted/replaced.
            if recording_id in state.recordings:
                return
            if previous and not previous.done():
                previous.cancel()
            state.nod.start_input(recording_id)
            state.recording_id = recording_id
            state.remember(recording_id)
        state.nod.update_user(recording_id, text)
        if not text.strip():
            return
        if previous and not previous.done() and not previous.cancelling():
            return  # One decision at a time; no pending-pause queue.

        async def decide():
            if previous and not previous.done():
                await asyncio.gather(previous, return_exceptions=True)
            if self._session(recording.session_id) is state:
                await state.nod.on_pause(recording_id)

        task = asyncio.create_task(decide())
        state.pending = task
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)

    def on_accepted(self, request):
        """End only this request's recording; do not end a newer recording."""
        state = self._session(request.session_id)
        if state is None:
            return
        recording_id = (request.metadata or {}).get("recording_id")
        if recording_id:
            state.nod.update_user(recording_id, request.text or "", create=True)
            state.nod.end_input(recording_id)
            state.remember(recording_id)
            if state.recording_id == recording_id:
                state.recording_id = None
                if state.pending:
                    state.pending.cancel()
        else:
            # Explicit text requests also supersede a pending spoken input.
            if state.pending:
                state.pending.cancel()
            if state.recording_id:
                state.nod.end_input(state.recording_id)
                state.recording_id = None
            identifier = "text:" + request.transaction_id
            if identifier not in state.recordings:
                state.nod.start_input(identifier)
                state.nod.update_user(identifier, request.text or "")
                state.nod.end_input(identifier)
                state.remember(identifier)
        state.transaction_id = request.transaction_id
        state.response_text = ""

    def on_response(self, response, sts_response):
        """Collect spoken chunks; a cumulative final replaces rather than appends."""
        state = self._session(sts_response.session_id)
        if (state is None or (response.metadata or {}).get("nod") is True
                or not sts_response.transaction_id
                or sts_response.transaction_id != state.transaction_id):
            return
        text = response.voice_text if response.voice_text is not None else response.text
        if response.type == "chunk" and text:
            state.response_text = (state.response_text + text)[-2400:]
        elif response.type == "final" and text is not None:
            state.response_text = text[-2400:]
        else:
            return
        if state.response_text:
            state.nod.update_assistant(state.transaction_id, state.response_text)

    def is_current(self, session_id, decision):
        state = self._session(session_id)
        return state is not None and state.nod.is_current(decision)

    async def _emit(self, session_id, decision):
        state = self._session(session_id)
        if state is None or not state.nod.is_current(decision):
            return False
        if self.emit:
            return await self.emit(session_id, decision)
        audio = self._audio.get(decision.id)
        if audio is None:
            return False
        # Import the optional framework only when using its wire format.
        from aiavatar.adapter.models import AIAvatarResponse
        socket = self.app.websockets.get(session_id)
        async with state.connection.send_lock:
            if (not self.is_current(session_id, decision) or socket is None
                    or self.app.websockets.get(session_id) is not socket):
                return False
            await socket.send_text(AIAvatarResponse(
                type="chunk", session_id=session_id, text="",
                voice_text="", audio_data=audio,
                metadata={"nod": True, "nod_id": str(uuid4()),
                          "recording_id": decision.utterance_id},
            ).model_dump_json())
            return True

    async def close_session(self, connection):
        state = self._sessions.get(connection.id)
        if state is None or state.connection is not connection:
            return
        del self._sessions[connection.id]
        for task in tuple(state.tasks):
            task.cancel()
        await state.nod.close()
        await asyncio.gather(*state.tasks, return_exceptions=True)

    async def close(self):
        self._closed = True
        for state in list(self._sessions.values()):
            await self.close_session(state.connection)
        self._audio.clear()
