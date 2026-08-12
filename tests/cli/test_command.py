import os
import subprocess
import sys
import warnings

import pytest

import aiavatar.cli.command as cli_command


REAL_LOAD_DOTENV = cli_command.load_dotenv


@pytest.fixture(autouse=True)
def disable_project_dotenv(monkeypatch):
    """Keep CLI tests isolated from a developer's ignored root .env file."""
    monkeypatch.setattr(cli_command, "load_dotenv", lambda **_: False)


def test_cli_hides_known_websockets_deprecation_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cli_command._configure_cli_warnings()
        warnings.warn(
            "websockets.legacy is deprecated; see the upgrade guide",
            DeprecationWarning,
        )
        warnings.warn(
            "websockets.server.WebSocketServerProtocol is deprecated",
            DeprecationWarning,
        )
        warnings.warn("unrelated deprecation", DeprecationWarning)

    assert [str(item.message) for item in caught] == ["unrelated deprecation"]


def test_command_import_does_not_load_builtin_components():
    code = """
import sys
import aiavatar.cli.command
assert 'aiavatar.cli.components' not in sys.modules
assert 'aiavatar.cli.builtin' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_load_app_from_python_file(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(
        "async def app(scope, receive, send):\n    pass\n",
        encoding="utf-8",
    )

    assert cli_command.load_app(str(script)).__name__ == "app"


def test_load_app_requires_app_export(tmp_path):
    script = tmp_path / "run.py"
    script.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not export 'app'"):
        cli_command.load_app(str(script))


def test_load_app_preserves_application_tracebacks(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text("raise RuntimeError('application bug')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="application bug"):
        cli_command.load_app(str(script))
    assert str(tmp_path) not in sys.path


def test_cli_sets_supplied_key_before_loading_script(monkeypatch):
    observed = {}
    application = object()

    def fake_load_app(target):
        observed["target"] = target
        observed["key"] = os.environ.get("OPENAI_API_KEY")
        observed["base_url"] = os.environ.get("OPENAI_BASE_URL")
        return application

    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example/v1")
    monkeypatch.setattr(cli_command, "load_app", fake_load_app)
    monkeypatch.setattr(
        cli_command.uvicorn,
        "run",
        lambda app, **kwargs: observed.update(app=app, kwargs=kwargs),
    )

    cli_command.main([
        "run.py",
        "--openai-api-key", "test-key",
        "--openai-base-url", "https://argument.example/v1",
        "--port", "9000",
    ])

    assert observed == {
        "target": "run.py",
        "key": "test-key",
        "base_url": "https://argument.example/v1",
        "app": application,
        "kwargs": {"host": "127.0.0.1", "port": 9000},
    }


def test_cli_preserves_openai_environment_without_arguments(monkeypatch):
    observed = {}
    application = object()
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example/v1")
    monkeypatch.setattr(
        cli_command,
        "load_app",
        lambda _: observed.update(
            key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        ) or application,
    )
    monkeypatch.setattr(cli_command.uvicorn, "run", lambda *args, **kwargs: None)

    cli_command.main(["run.py"])

    assert observed == {
        "key": "environment-key",
        "base_url": "https://environment.example/v1",
    }


def test_cli_loads_dotenv_without_overriding_environment(
    monkeypatch,
    tmp_path,
    clean_builtin_environment,
):
    observed = {}
    application = object()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_command, "load_dotenv", REAL_LOAD_DOTENV)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example/v1")
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dotenv-key\n"
        "OPENAI_BASE_URL=https://dotenv.example/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_command,
        "load_app",
        lambda _: observed.update(
            key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        ) or application,
    )
    monkeypatch.setattr(cli_command.uvicorn, "run", lambda *args, **kwargs: None)

    try:
        cli_command.main(["run.py"])
    finally:
        os.environ.pop("OPENAI_API_KEY", None)

    assert observed == {
        "key": "dotenv-key",
        "base_url": "https://environment.example/v1",
    }


def test_cli_tts_arguments_override_environment(
    monkeypatch,
    clean_builtin_environment,
):
    observed = {}
    application = object()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AIAVATAR_JA_TTS", "voicevox")
    monkeypatch.setenv("AIAVATAR_MULTI_TTS", "openai")
    monkeypatch.setenv("AIAVATAR_OPENAI_TTS_SPEAKER", "sage")
    monkeypatch.setenv("AIAVATAR_LLM_EXTRA_BODY", '{"old":true}')
    monkeypatch.setenv("AIAVATAR_LLM_API", "responses-websocket")
    monkeypatch.setattr(
        cli_command,
        "_load_builtin_app",
        lambda: observed.update(
            ja_tts=os.environ.get("AIAVATAR_JA_TTS"),
            multi_tts=os.environ.get("AIAVATAR_MULTI_TTS"),
            openai_tts_speaker=os.environ.get("AIAVATAR_OPENAI_TTS_SPEAKER"),
            llm_extra_body=os.environ.get("AIAVATAR_LLM_EXTRA_BODY"),
            llm_api=os.environ.get("AIAVATAR_LLM_API"),
        ) or application,
    )
    monkeypatch.setattr(cli_command.uvicorn, "run", lambda *args, **kwargs: None)

    cli_command.main([
        "--ja-tts", "instant",
        "--multi-tts", "voicevox",
        "--openai-tts-speaker", "coral",
        "--llm-extra-body", '{"thinking":{"type":"disabled"}}',
        "--llm-api", "chat-completions",
    ])

    assert observed == {
        "ja_tts": "instant",
        "multi_tts": "voicevox",
        "openai_tts_speaker": "coral",
        "llm_extra_body": '{"thinking":{"type":"disabled"}}',
        "llm_api": "chat-completions",
    }


def test_builtin_app_prompts_for_missing_key(
    monkeypatch,
    clean_builtin_environment,
):
    observed = {}
    application = object()

    monkeypatch.setattr(cli_command.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_command.getpass, "getpass", lambda _: "prompt-key")
    monkeypatch.setattr(
        cli_command,
        "_load_builtin_app",
        lambda: observed.update(key=os.environ.get("OPENAI_API_KEY")) or application,
    )
    monkeypatch.setattr(
        cli_command.uvicorn,
        "run",
        lambda app, **kwargs: observed.update(app=app),
    )

    cli_command.main([])

    assert observed == {"key": "prompt-key", "app": application}


def test_builtin_app_accepts_individual_openai_keys(
    monkeypatch,
    clean_builtin_environment,
):
    observed = {}
    application = object()

    monkeypatch.setenv("AIAVATAR_STT_OPENAI_API_KEY", "stt-key")
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("AIAVATAR_TTS_OPENAI_API_KEY", "tts-key")
    monkeypatch.setattr(cli_command.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        cli_command,
        "_load_builtin_app",
        lambda: observed.update(loaded=True) or application,
    )
    monkeypatch.setattr(cli_command.uvicorn, "run", lambda *args, **kwargs: None)

    cli_command.main([])

    assert observed == {"loaded": True}


def test_builtin_app_does_not_require_unused_tts_openai_key(
    monkeypatch,
    clean_builtin_environment,
):
    observed = {}
    application = object()

    monkeypatch.setenv("AIAVATAR_STT_OPENAI_API_KEY", "stt-key")
    monkeypatch.setenv("AIAVATAR_LLM_OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("AIAVATAR_JA_TTS", "voicevox")
    monkeypatch.setenv(
        "AIAVATAR_JA_TTS_CONFIG",
        '{"alphabet_to_kana":false}',
    )
    monkeypatch.setenv("AIAVATAR_MULTI_TTS", "voicevox")
    monkeypatch.setattr(cli_command.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        cli_command,
        "_load_builtin_app",
        lambda: observed.update(loaded=True) or application,
    )
    monkeypatch.setattr(cli_command.uvicorn, "run", lambda *args, **kwargs: None)

    cli_command.main([])

    assert observed == {"loaded": True}
