from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from aiavatar.adapter.base import Adapter
from aiavatar.admin import BasicAdminAuthenticator, setup_admin_panel
from aiavatar.admin_legacy import setup_admin_panel as setup_legacy_admin_panel
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder
from aiavatar.sts.performance_recorder import PerformanceRecord


class Configurable:
    def __init__(self):
        self.enabled = True

    def get_config(self):
        return {"enabled": self.enabled}

    def set_config(self, config):
        if config.get("enabled") is not None:
            self.enabled = config["enabled"]
        return {"enabled": self.enabled}


class Pipeline(Configurable):
    def __init__(self, recorder):
        super().__init__()
        self.performance_recorder = recorder
        self.voice_recorder_enabled = False
        self.voice_recorder = None
        self.vad = Configurable()
        self.stt = Configurable()
        self.llm = Configurable()
        self.tts = Configurable()


class VoiceRecorderFake:
    async def get_request_voice(self, transaction_id):
        return b"RIFF-request"

    async def get_response_voices(self, transaction_id):
        return [b"RIFF-response"]

    async def get_voice(self, voice_id):
        if voice_id.endswith("_response_0"):
            return b"RIFF-response"
        return None


class AIAvatarTestServer(Adapter):
    def __init__(self, recorder):
        self.sts = Pipeline(recorder)

    async def handle_response(self, response):
        pass

    async def stop_response(self, session_id, context_id):
        pass


def test_new_admin_uses_one_replaceable_authenticator_and_new_routes(tmp_path):
    recorder = SQLitePerformanceRecorder(str(tmp_path / "admin.db"))
    try:
        app = FastAPI()
        adapter = AIAvatarTestServer(recorder)
        adapter.sts.voice_recorder_enabled = True
        adapter.sts.voice_recorder = VoiceRecorderFake()
        setup_admin_panel(
            app,
            adapter=adapter,
            authenticator=BasicAdminAuthenticator("admin", "secret"),
        )
        client = TestClient(app)
        recorder.record(PerformanceRecord(
            transaction_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            user_id="user",
            session_id="session-new",
            context_id="context",
            request_text="hello admin",
            speech_end_at=datetime.now(timezone.utc),
            silence_threshold_time=0.1,
            stt_after_threshold_time=0.1,
            turn_end_gate_time=0.1,
            stt_time=0.1,
            stop_response_time=0.2,
            before_llm_time=0.3,
            llm_first_chunk_time=0.4,
            llm_first_voice_chunk_time=0.5,
            tts_first_chunk_time=0.6,
        ))
        recorder.record_queue.join()

        assert client.get("/admin", follow_redirects=False).status_code == 401
        assert client.get("/admin/api/capabilities").status_code == 401
        assert client.get("/admin/assets/admin-app.js").status_code == 401

        auth = ("admin", "secret")
        redirect = client.get("/admin", auth=auth, follow_redirects=False)
        assert redirect.status_code == 307
        page = client.get("/admin/", auth=auth)
        assert page.status_code == 200
        assert "assets/admin-app.js" in page.text
        assert client.get("/admin/assets/admin-app.js", auth=auth).status_code == 200
        assert client.get("/admin/api/capabilities", auth=auth).json() == {"evaluation": False}
        assert client.get("/admin/api/metrics/summary", auth=auth).status_code == 200
        logs = client.get("/admin/api/logs?session_id=session-new", auth=auth)
        assert logs.status_code == 200
        log = logs.json()["groups"][0]["logs"][0]
        assert log["session_id"] == "session-new"
        timing = log["timing_breakdown"]
        assert timing["total_first_response"] == pytest.approx(0.9)
        assert [
            timing[key]
            for key in (
                "silence_detection", "streaming_stt_finalization", "turn_end_gate",
                "stt", "stop_response", "before_llm", "llm", "processing", "tts",
            )
        ] == pytest.approx([0.1] * 9)
        assert client.get(
            "/admin/api/logs/voice/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/request",
            auth=auth,
        ).content == b"RIFF-request"
        assert client.get(
            "/admin/api/logs/voice/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/response",
            auth=auth,
        ).json() == {"count": 1}
        assert client.get("/metrics/summary", auth=auth).status_code == 404

        paths = {route.path for route in app.routes}
        assert not any("character" in path or "control" in path or path == "/conversation" for path in paths)
    finally:
        recorder.close()


def test_new_admin_accepts_custom_authenticator(tmp_path):
    recorder = SQLitePerformanceRecorder(str(tmp_path / "custom-auth.db"))
    try:
        async def sso_auth(request: Request):
            if request.headers.get("X-SSO-User") != "uezo":
                raise HTTPException(status_code=401)
            return "uezo"

        app = FastAPI()
        setup_admin_panel(app, adapter=AIAvatarTestServer(recorder), authenticator=sso_auth)
        client = TestClient(app)
        assert client.get("/admin/api/capabilities").status_code == 401
        assert client.get("/admin/api/capabilities", headers={"X-SSO-User": "uezo"}).status_code == 200
    finally:
        recorder.close()


def test_legacy_admin_remains_selectable_with_old_routes(tmp_path):
    recorder = SQLitePerformanceRecorder(str(tmp_path / "legacy-admin.db"))
    try:
        app = FastAPI()
        setup_legacy_admin_panel(
            app,
            adapter=AIAvatarTestServer(recorder),
            api_key="legacy-key",
            basic_auth_username="admin",
            basic_auth_password="secret",
        )
        client = TestClient(app)
        recorder.record(PerformanceRecord(
            transaction_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            session_id="legacy-session",
            context_id="legacy-context",
        ))
        recorder.record_queue.join()
        assert client.get("/admin").status_code == 401
        assert client.get("/admin", auth=("admin", "secret")).status_code == 200
        headers = {"Authorization": "Bearer legacy-key"}
        assert client.get("/metrics/summary", headers=headers).status_code == 200
        legacy_log = client.get("/logs", headers=headers).json()["groups"][0]["logs"][0]
        assert "session_id" not in legacy_log
        paths = {route.path for route in app.routes}
        assert "/avatar/perform" in paths
        assert "/conversation" in paths
    finally:
        recorder.close()
