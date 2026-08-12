"""Helpers for the built-in CLI application and generated starter scripts.

The public helpers are loaded lazily so ``aiavatar path/to/app.py`` does not
import any of the built-in speech providers.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components import ComponentSet, build_components
    from .config import AppConfig


__all__ = [
    "AppConfig",
    "ComponentSet",
    "build_components",
]


def __getattr__(name: str):
    if name == "AppConfig":
        from .config import AppConfig

        return AppConfig
    if name in {"ComponentSet", "build_components"}:
        from .components import ComponentSet, build_components

        return {
            "ComponentSet": ComponentSet,
            "build_components": build_components,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
