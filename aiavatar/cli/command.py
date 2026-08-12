import argparse
import getpass
import importlib
import importlib.util
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import uvicorn
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class ApplicationLoadError(ValueError):
    pass


def _configure_cli_warnings() -> None:
    """Hide dependency deprecations that users cannot act on from this CLI."""
    warnings.filterwarnings(
        "ignore",
        message=r"websockets\.legacy is deprecated;.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"websockets\.server\.WebSocketServerProtocol is deprecated",
        category=DeprecationWarning,
    )


def _load_file_module(path: Path):
    resolved = path.resolve()
    if not resolved.is_file():
        raise ApplicationLoadError(f"Application script not found: {path}")
    module_name = f"_aiavatar_user_app_{abs(hash(resolved))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ApplicationLoadError(f"Could not load application script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    script_dir = str(resolved.parent)
    inserted_script_dir = script_dir not in sys.path
    if inserted_script_dir:
        sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        if inserted_script_dir and script_dir in sys.path:
            sys.path.remove(script_dir)
        raise
    return module


def load_app(target: str) -> Any:
    """Load an ASGI ``app`` from a Python file or ``module:attribute``."""
    path = Path(target)
    if path.suffix == ".py" or path.is_file():
        module = _load_file_module(path)
        attribute = "app"
    else:
        module_name, separator, attribute = target.partition(":")
        attribute = attribute if separator else "app"
        if not module_name or not attribute:
            raise ApplicationLoadError(f"Invalid application target: {target}")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as ex:
            if ex.name is None or not (
                ex.name == module_name or module_name.startswith(ex.name + ".")
            ):
                raise
            raise ApplicationLoadError(
                f"Application module not found: {module_name}"
            ) from ex

    if not hasattr(module, attribute):
        raise ApplicationLoadError(
            f"Application target '{target}' does not export '{attribute}'"
        )
    app = getattr(module, attribute)
    if not callable(app):
        raise ApplicationLoadError(
            f"Application target '{target}' is not an ASGI application"
        )
    return app


def _load_builtin_app() -> Any:
    from .builtin import create_app

    return create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiavatar",
        description="Run the default AIAvatarKit app or a Python ASGI application.",
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Python file exporting 'app', or module[:attribute]",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--openai-api-key",
        help=(
            "Set OPENAI_API_KEY for this process before loading the application. "
            "Using the environment variable avoids storing the key in shell history."
        ),
    )
    parser.add_argument(
        "--openai-base-url",
        help="Set OPENAI_BASE_URL for this process before loading the application.",
    )
    parser.add_argument(
        "--ja-tts",
        choices=("voicevox", "openai", "instant"),
        help=(
            "Japanese TTS for the default app. Overrides AIAVATAR_JA_TTS; "
            "defaults to voicevox."
        ),
    )
    parser.add_argument(
        "--multi-tts",
        choices=("voicevox", "openai", "instant"),
        help=(
            "Non-Japanese TTS for the default app. Overrides "
            "AIAVATAR_MULTI_TTS; defaults to openai."
        ),
    )
    parser.add_argument(
        "--openai-tts-speaker",
        help=(
            "OpenAI TTS speaker for the default app. Overrides "
            "AIAVATAR_OPENAI_TTS_SPEAKER; defaults to sage."
        ),
    )
    parser.add_argument(
        "--llm-extra-body",
        metavar="JSON",
        help=(
            "JSON object passed as LLM extra_body by the default app. Overrides "
            "AIAVATAR_LLM_EXTRA_BODY and disables the default reasoning_effort."
        ),
    )
    parser.add_argument(
        "--llm-api",
        choices=("responses-websocket", "chat-completions"),
        help=(
            "LLM API for the default app. Overrides AIAVATAR_LLM_API; "
            "defaults to responses-websocket."
        ),
    )
    return parser


def _prepare_openai_api_key(
    parser: argparse.ArgumentParser,
    *,
    supplied_key: Optional[str],
    using_builtin_app: bool,
) -> None:
    if supplied_key:
        os.environ["OPENAI_API_KEY"] = supplied_key
        return
    if os.getenv("OPENAI_API_KEY") or not using_builtin_app:
        return
    individual_keys = [
        "AIAVATAR_STT_OPENAI_API_KEY",
        "AIAVATAR_LLM_OPENAI_API_KEY",
    ]
    if all(os.getenv(name) for name in individual_keys):
        return
    if not sys.stdin.isatty():
        parser.error(
            "OPENAI_API_KEY is required for the default app; set the environment "
            "variable or pass --openai-api-key"
        )
    api_key = getpass.getpass("OpenAI API Key: ").strip()
    if not api_key:
        parser.error("OpenAI API Key must not be empty")
    os.environ["OPENAI_API_KEY"] = api_key


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_cli_warnings()
    logging.basicConfig(level=logging.INFO)
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    if args.ja_tts:
        os.environ["AIAVATAR_JA_TTS"] = args.ja_tts
    if args.multi_tts:
        os.environ["AIAVATAR_MULTI_TTS"] = args.multi_tts
    if args.openai_tts_speaker:
        os.environ["AIAVATAR_OPENAI_TTS_SPEAKER"] = args.openai_tts_speaker
    if args.llm_extra_body:
        os.environ["AIAVATAR_LLM_EXTRA_BODY"] = args.llm_extra_body
    if args.llm_api:
        os.environ["AIAVATAR_LLM_API"] = args.llm_api
    _prepare_openai_api_key(
        parser,
        supplied_key=args.openai_api_key,
        using_builtin_app=args.script is None,
    )

    try:
        app = load_app(args.script) if args.script else _load_builtin_app()
    except ApplicationLoadError as ex:
        parser.error(str(ex))

    # Application imports may install their own warning filters.
    _configure_cli_warnings()
    if args.script:
        logger.info("Application: http://%s:%s/", args.host, args.port)
    else:
        logger.info("Admin Panel: http://%s:%s/admin/", args.host, args.port)
        logger.info("WebSocket example: http://%s:%s/", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
