"""App selection, shared static pages, and ownership; no API or TTS services."""

from pathlib import Path

from fastapi import APIRouter
import httpx
import pytest

from examples.websocket.realtime import server


@pytest.fixture
def components(monkeypatch):
    created = []

    class Pipeline:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            self.fail_shutdown = False
            created.append(self)

        async def shutdown(self):
            self.closed = True
            if self.fail_shutdown:
                raise RuntimeError("shutdown failed")

    class TTS:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            created.append(self)

        async def close(self):
            self.closed = True

    class Adapter:
        def __init__(self, *, sts, api_key):
            self.sts = sts

        def get_websocket_router(self, path):
            assert path == "/ws"
            return APIRouter()

    monkeypatch.setattr(server, "OpenAIRealtimePipeline", Pipeline)
    monkeypatch.setattr(server, "OpenAILivePipeline", Pipeline)
    monkeypatch.setattr(server, "VoicevoxSpeechSynthesizer", TTS)
    monkeypatch.setattr(server, "RealtimeWebSocketServer", Adapter)
    return created, TTS


@pytest.mark.asyncio
@pytest.mark.parametrize("pipeline", ["realtime", "live"])
async def test_serves_existing_pages_and_shuts_down(pipeline, components):
    created, _ = components
    app = server.create_app(pipeline)
    sts = app.state.adapter.sts
    assert sts.options["input_sample_rate"] == 16000
    if pipeline == "realtime":
        assert sts.options["context_db_path"] is None
        assert sts.options["restore_context"] is False
    else:
        assert len(created) == 1, "Live must not construct a TTS client"
        assert sts.options["voice"] == "marin"

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/")
            assert response.headers["location"] == "html/index.html"
            html_dir = Path(server.__file__).resolve().parents[1] / "html"
            for name in ("index.html", "3d.html", "aiavatar.js", "ui.js"):
                response = await client.get("/html/" + name)
                assert response.status_code == 200
                assert response.content == (html_dir / name).read_bytes()
    assert all(component.closed for component in created)


@pytest.mark.asyncio
async def test_injected_tts_is_caller_owned(components):
    _, TTS = components
    tts = TTS()
    app = server.create_app("realtime", tts=tts, model="test-model", instructions="Test")
    async with app.router.lifespan_context(app):
        assert app.state.adapter.sts.options["tts"] is tts
        assert app.state.adapter.sts.options["model"] == "test-model"
        assert app.state.adapter.sts.options["instructions"] == "Test"
    assert app.state.adapter.sts.closed
    assert not tts.closed


@pytest.mark.asyncio
async def test_owned_tts_closes_even_when_pipeline_shutdown_fails(components):
    app = server.create_app("realtime")
    app.state.adapter.sts.fail_shutdown = True
    tts = app.state.adapter.sts.options["tts"]
    with pytest.raises(RuntimeError, match="shutdown failed"):
        async with app.router.lifespan_context(app):
            pass
    assert tts.closed


def test_rejects_unsupported_pipeline_before_creating_resources(components):
    created, TTS = components
    with pytest.raises(ValueError, match="pipeline"):
        server.create_app("unknown")
    assert created == []
    tts = TTS()
    with pytest.raises(ValueError, match="native audio"):
        server.create_app("live", tts=tts)
    assert created == [tts]
