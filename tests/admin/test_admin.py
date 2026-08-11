from datetime import datetime, timezone
import threading

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from aiavatar.adapter.base import Adapter
from aiavatar.admin import BasicAdminAuthenticator, setup_admin_panel
from aiavatar.admin_legacy import setup_admin_panel as setup_legacy_admin_panel
from aiavatar.sts.performance_recorder.sqlite import SQLitePerformanceRecorder
from aiavatar.sts.performance_recorder import PerformanceRecord
from aiavatar.sts.tts import SpeechSynthesizerRouter
from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.sts.tts.voicevox import VoicevoxSpeechSynthesizer
from aiavatar.sts.vad.silero import SileroSpeechDetector
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector


class Configurable:
    def __init__(
        self,
        enabled: bool = True,
        sample_rate: int = 16000,
        api_key: str = None,
        max_tokens: int = 100,
        language: str | None = None,
    ):
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.language = language

    def get_config(self):
        raise AssertionError("Admin must read actual members instead of get_config()")

    def set_config(self, config):
        raise AssertionError("Admin must update actual members instead of set_config()")


class Pipeline(Configurable):
    def __init__(self, recorder, debug: bool = False):
        super().__init__()
        self.debug = debug
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
    def __init__(self, recorder, response_audio_chunk_size: int = 0):
        self.sts = Pipeline(recorder)
        self.response_audio_chunk_size = response_audio_chunk_size

    def can_handle(self, session_id):
        return False

    async def handle_response(self, response):
        pass

    async def stop_response(self, session_id, context_id):
        pass


def test_admin_routes_take_priority_over_root_static_files(tmp_path):
    recorder = SQLitePerformanceRecorder(str(tmp_path / "root-mount.db"))
    try:
        ui_dir = tmp_path / "ui"
        ui_dir.mkdir()
        (ui_dir / "index.html").write_text("WebSocket UI", encoding="utf-8")

        app = FastAPI()
        setup_admin_panel(
            app,
            adapter=AIAvatarTestServer(recorder),
            authenticator=BasicAdminAuthenticator("admin", "secret"),
        )
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
        client = TestClient(app)

        assert client.get("/").text == "WebSocket UI"
        assert client.get("/admin").status_code == 401
        assert client.get("/admin/assets/admin-app.js").status_code == 401
        assert client.get("/admin/api/capabilities").status_code == 401

        auth = ("admin", "secret")
        assert client.get(
            "/admin",
            auth=auth,
            follow_redirects=False,
        ).status_code == 307
        assert client.get("/admin/", auth=auth).status_code == 200
        assert client.get(
            "/admin/assets/admin-app.js",
            auth=auth,
        ).status_code == 200
        assert client.get("/admin/api/capabilities", auth=auth).json() == {
            "evaluation": False,
        }
    finally:
        recorder.close()


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
            channel="websocket",
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
        recorder.record(PerformanceRecord(
            transaction_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            session_id="session-text",
            context_id="context",
            channel="linebot",
            request_text="hello text",
            before_llm_time=0.1,
            llm_first_chunk_time=0.2,
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
        assert client.get("/admin/api/capabilities", auth=auth).json() == {
            "evaluation": False,
        }
        assert client.get("/admin/api/metrics/summary", auth=auth).status_code == 200
        channel_metrics = client.get(
            "/admin/api/metrics/by-channel?period=24h&interval=1h",
            auth=auth,
        )
        assert channel_metrics.status_code == 200
        metrics = channel_metrics.json()
        assert metrics["total_requests"] == 2
        by_channel = {item["channel"]: item for item in metrics["channels"]}
        assert by_channel["linebot"]["pipeline_summary"]["avg_first_response_time"] == pytest.approx(0.2)
        assert by_channel["linebot"]["speech_summary"]["measured_count"] == 0
        assert by_channel["websocket"]["pipeline_summary"]["avg_first_response_time"] == pytest.approx(0.6)
        assert by_channel["websocket"]["speech_summary"]["avg_first_response_time"] == pytest.approx(0.9)
        assert client.get(
            "/admin/api/metrics/by-channel?period=24h&interval=invalid",
            auth=auth,
        ).status_code == 400
        logs = client.get("/admin/api/logs?session_id=session-new", auth=auth)
        assert logs.status_code == 200
        log = logs.json()["groups"][0]["logs"][0]
        assert log["session_id"] == "session-new"
        assert log["channel"] == "websocket"
        linebot_logs = client.get("/admin/api/logs?channel=linebot", auth=auth)
        assert linebot_logs.status_code == 200
        context_logs = linebot_logs.json()["groups"][0]["logs"]
        assert {item["session_id"] for item in context_logs} == {"session-new", "session-text"}
        assert {item["channel"] for item in context_logs} == {"websocket", "linebot"}
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


def test_new_admin_updates_safe_members_without_persisting(tmp_path):
    recorder = SQLitePerformanceRecorder(str(tmp_path / "member-config.db"))
    try:
        app = FastAPI()
        adapter = AIAvatarTestServer(recorder)
        setup_admin_panel(app, adapter=adapter)
        client = TestClient(app)

        schema = client.get("/admin/api/config/runtime").json()
        sections = {section["name"]: section for section in schema["sections"]}
        assert set(sections) == {
            "pipeline", "vad", "stt", "llm", "tts", "adapter:test",
        }
        vad = sections["vad"]
        assert vad["component"] == "Configurable"
        fields = {field["name"]: field for field in vad["fields"]}
        assert fields["enabled"]["value"] is True
        assert "sample_rate" not in fields
        assert fields["api_key"]["secret"] is True
        assert fields["api_key"]["value"] is None
        assert fields["max_tokens"]["secret"] is False
        assert fields["language"]["nullable"] is True

        invalid = client.post(
            "/admin/api/config/runtime/vad",
            json={"config": {"enabled": False, "max_tokens": 1.5}},
        )
        assert invalid.status_code == 400
        assert adapter.sts.vad.enabled is True

        response = client.post(
            "/admin/api/config/runtime/vad",
            json={"config": {"enabled": False}},
        )
        assert response.status_code == 200
        assert response.json() == {"updated": ["enabled"]}
        assert adapter.sts.vad.enabled is False

        language_response = client.post(
            "/admin/api/config/runtime/vad",
            json={"config": {"language": "en"}},
        )
        assert language_response.status_code == 200
        assert adapter.sts.vad.language == "en"
        refreshed = client.get("/admin/api/config/runtime").json()
        refreshed_vad = next(
            section for section in refreshed["sections"] if section["name"] == "vad"
        )
        refreshed_fields = {
            field["name"]: field for field in refreshed_vad["fields"]
        }
        assert refreshed_fields["language"]["nullable"] is True

        clear_language_response = client.post(
            "/admin/api/config/runtime/vad",
            json={"config": {"language": None}},
        )
        assert clear_language_response.status_code == 200
        assert adapter.sts.vad.language is None

        secret_response = client.post(
            "/admin/api/config/runtime/vad",
            json={"config": {"api_key": "volatile-test-key"}},
        )
        assert secret_response.status_code == 200
        assert adapter.sts.vad.api_key == "volatile-test-key"
        assert "volatile-test-key" not in client.get(
            "/admin/api/config/runtime"
        ).text

        refreshed = client.get("/admin/api/config/runtime").json()
        refreshed_vad = next(
            section for section in refreshed["sections"] if section["name"] == "vad"
        )
        refreshed_fields = {
            field["name"]: field for field in refreshed_vad["fields"]
        }
        assert refreshed_fields["enabled"]["value"] is False
    finally:
        recorder.close()


def test_vad_threshold_change_applies_to_new_sessions(monkeypatch, tmp_path):
    class FakeVadIterator:
        def __init__(self, model, threshold, sampling_rate):
            self.model = model
            self.threshold = threshold
            self.sampling_rate = sampling_rate

        def reset_states(self):
            pass

    def initialize_fake_model_pool(self, *_):
        self.model_pool = [object()]
        self.model_locks = [threading.Lock()]
        self.VADIteratorClass = FakeVadIterator

    monkeypatch.setattr(
        SileroSpeechDetector,
        "_init_silero_model",
        initialize_fake_model_pool,
    )
    vad = SileroStreamSpeechDetector(
        speech_recognizer=object(),
        speech_probability_threshold=0.5,
        use_vad_iterator=True,
    )
    existing_session = vad.get_session("existing")

    recorder = SQLitePerformanceRecorder(str(tmp_path / "vad-threshold.db"))
    try:
        adapter = AIAvatarTestServer(recorder)
        adapter.sts.vad = vad
        app = FastAPI()
        setup_admin_panel(app, adapter=adapter)
        client = TestClient(app)

        response = client.post(
            "/admin/api/config/runtime/vad",
            json={"config": {"speech_probability_threshold": 0.7}},
        )

        assert response.status_code == 200
        assert vad.speech_probability_threshold == 0.7
        assert existing_session.vad_iterator.threshold == 0.5
        assert vad.get_session("new").vad_iterator.threshold == 0.7
    finally:
        recorder.close()


@pytest.mark.asyncio
async def test_admin_expands_tts_router_into_route_synthesizers(tmp_path):
    tts_ja = VoicevoxSpeechSynthesizer(
        base_url="http://voicevox.example:50021",
        speaker=46,
    )
    tts_multi = OpenAISpeechSynthesizer(
        openai_api_key="tts-key",
        speaker="sage",
        model="gpt-4o-mini-tts",
        audio_format="wav",
    )
    router = SpeechSynthesizerRouter({
        "ja": tts_ja,
        "multi": tts_multi,
    })
    recorder = SQLitePerformanceRecorder(str(tmp_path / "tts-router.db"))
    try:
        adapter = AIAvatarTestServer(recorder)
        adapter.sts.tts = router
        app = FastAPI()
        setup_admin_panel(app, adapter=adapter)
        client = TestClient(app)

        schema = client.get("/admin/api/config/runtime").json()
        sections = {section["name"]: section for section in schema["sections"]}
        assert "tts" not in sections
        assert sections["tts:ja"]["title"] == "TTS · ja"
        assert sections["tts:ja"]["component"] == "VoicevoxSpeechSynthesizer"
        assert sections["tts:multi"]["title"] == "TTS · multi"
        assert sections["tts:multi"]["component"] == "OpenAISpeechSynthesizer"

        ja_fields = {
            field["name"]: field for field in sections["tts:ja"]["fields"]
        }
        multi_fields = {
            field["name"]: field for field in sections["tts:multi"]["fields"]
        }
        assert ja_fields["base_url"]["value"] == "http://voicevox.example:50021"
        assert ja_fields["speaker"]["value"] == 46
        assert multi_fields["speaker"]["value"] == "sage"
        assert multi_fields["model"]["value"] == "gpt-4o-mini-tts"
        assert multi_fields["openai_api_key"]["secret"] is True
        assert multi_fields["openai_api_key"]["value"] is None
        assert "audio_format" not in multi_fields

        ja_response = client.post(
            "/admin/api/config/runtime/tts%3Aja",
            json={"config": {"speaker": 3}},
        )
        multi_response = client.post(
            "/admin/api/config/runtime/tts%3Amulti",
            json={"config": {"speaker": "coral", "openai_api_key": "new-key"}},
        )
        assert ja_response.status_code == 200
        assert multi_response.status_code == 200
        assert tts_ja.speaker == 3
        assert tts_multi.speaker == "coral"
        assert tts_multi.openai_api_key == "new-key"
        assert router.synthesizers == {"ja": tts_ja, "multi": tts_multi}
    finally:
        await router.close()
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
