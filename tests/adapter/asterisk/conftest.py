from typing import Any


class DummyVAD:
    def __init__(self):
        self.recording_started_handlers = []
        self.session_data = {}
        self.samples = []
        self.finalized = []

    def on_recording_started(self, func):
        self.recording_started_handlers.append(func)
        return func

    def set_session_data(self, session_id, key, value, create_session=False):
        if create_session:
            self.session_data.setdefault(session_id, {})
        if session_id in self.session_data:
            self.session_data[session_id][key] = value

    def get_session_data(self, session_id, key):
        return self.session_data.get(session_id, {}).get(key)

    async def process_samples(self, samples, session_id):
        self.samples.append((session_id, samples))
        return False

    async def finalize_session(self, session_id):
        self.finalized.append(session_id)
        self.session_data.pop(session_id, None)


class DummyTTS:
    async def synthesize(self, text):
        return b"\x00\x00" * 320


class DummySTS:
    def __init__(self):
        self.vad = DummyVAD()
        self.tts = DummyTTS()
        self.response_handlers = []
        self.finalized = []

    def add_response_handler(self, handler):
        self.response_handlers.append(handler)

    async def finalize(self, session_id):
        self.finalized.append(session_id)
        await self.vad.finalize_session(session_id)


class FakeMediaWebSocket:
    def __init__(self, *, headers=None, query_params=None):
        self.headers = headers or {"sec-websocket-protocol": "media"}
        self.query_params = query_params or {}
        self.text_messages = []
        self.binary_messages = []
        self.closed = []

    async def send_text(self, source):
        self.text_messages.append(source)

    async def send_bytes(self, source):
        self.binary_messages.append(source)

    async def close(self, code=1000):
        self.closed.append(code)


class FakeResponse:
    def __init__(self, status_code=204, payload: Any = None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.content = b"" if payload is None else b"{}"
        self.text = text

    def json(self):
        return self._payload


class FakeARIClient:
    def __init__(self):
        self.calls = []
        self.variables = {}

    async def request(self, method, path, params=None, json=None):
        params = dict(params or {})
        self.calls.append((method, path, params, json))
        if method == "GET" and path.endswith("/variable"):
            key = (path, params.get("variable"))
            if key in self.variables:
                return FakeResponse(200, {"value": self.variables[key]})
            return FakeResponse(404)
        if method == "POST" and path == "channels/externalMedia":
            return FakeResponse(201, {"id": params["channelId"]})
        if method == "POST" and path == "channels":
            return FakeResponse(201, {"id": params["channelId"]})
        return FakeResponse()


class FakeAdapter:
    def __init__(self):
        self.sessions = {}
        self.manager = None
        self.started = []
        self.completed = []
        self.failed = []
        self.unknown = []
        self.unregistered = []
        self.prepared = []
        self.transfer_variables = {}

    def bind_call_manager(self, manager):
        self.manager = manager

    def register_session(self, session_id, **values):
        from aiavatar.adapter.asterisk.models import AsteriskSessionData

        session = self.sessions.setdefault(
            session_id,
            AsteriskSessionData(session_id=session_id),
        )
        for name, value in values.items():
            setattr(session, name, value)
        return session

    async def unregister_session(self, session_id):
        self.unregistered.append(session_id)
        self.sessions.pop(session_id, None)

    async def prepare_transfer(
        self,
        session,
        *,
        destination_alias,
        destination,
        transfer_strategy,
    ):
        from aiavatar.adapter.asterisk.models import AsteriskTransferRequest

        request = AsteriskTransferRequest(
            session_id=session.session_id,
            user_id=session.user_id or session.caller_number or session.session_id,
            context_id=session.context_id,
            destination_alias=destination_alias,
            destination=destination,
            transfer_strategy=transfer_strategy,
            variables=dict(self.transfer_variables),
        )
        self.prepared.append(request)
        return request

    async def notify_transfer_started(self, session_id, destination):
        self.started.append((session_id, destination))

    async def notify_transfer_completed(self, session_id, destination, method):
        self.completed.append((session_id, destination, method))

    async def notify_transfer_failed(self, session_id, destination, reason):
        self.failed.append((session_id, destination, reason))

    async def notify_transfer_unknown(self, session_id, destination, reason):
        self.unknown.append((session_id, destination, reason))
