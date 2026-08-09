from contextlib import AsyncExitStack
import inspect
from typing import Callable, Dict, Mapping, Optional

from .base import SpeechSynthesizer


RouteFunction = Callable[[str, Optional[dict], Optional[str]], Optional[str]]


class SpeechSynthesizerRouter(SpeechSynthesizer):
    """Route synthesis requests to one of several speech synthesizers.

    The router itself does not cache or transform audio. Each registered
    synthesizer retains its own preprocessing, caching, generation, and
    postprocessing flow.
    """

    def __init__(
        self,
        synthesizers: Mapping[str, SpeechSynthesizer],
        *,
        default: str = None,
        debug: bool = False,
    ):
        if not synthesizers:
            raise ValueError("synthesizers must not be empty")

        invalid_keys = [
            key for key in synthesizers
            if not isinstance(key, str) or not key
        ]
        if invalid_keys:
            raise ValueError("synthesizer route keys must be non-empty strings")

        invalid_synthesizers = [
            key for key, synthesizer in synthesizers.items()
            if not isinstance(synthesizer, SpeechSynthesizer)
        ]
        if invalid_synthesizers:
            raise TypeError(
                "registered synthesizers must be SpeechSynthesizer instances: "
                f"{invalid_synthesizers}"
            )

        if default is not None and default not in synthesizers:
            raise ValueError(f"Unknown default TTS route: {default}")

        # No cache, preprocessors, or sample-rate conversion is configured on
        # the router. Those concerns remain with each registered synthesizer.
        super().__init__(cache_dir=None, sample_rate=None, debug=debug)
        self.synthesizers: Dict[str, SpeechSynthesizer] = dict(synthesizers)
        self.default = default
        self._route_func: Optional[RouteFunction] = None

    def route(self, func: RouteFunction) -> RouteFunction:
        """Register the synchronous function used to select a TTS route."""
        if not callable(func):
            raise TypeError("route must be callable")
        if inspect.iscoroutinefunction(func):
            raise TypeError("route must be a synchronous function")
        self._route_func = func
        return func

    async def generate(
        self,
        text: str,
        style_info: dict = None,
        language: str = None,
    ) -> bytes:
        if self._route_func is None:
            raise RuntimeError("TTS route function is not configured")

        route = self._route_func(text, style_info, language)
        if route is None:
            route = self.default
        if route is None:
            raise ValueError("TTS route returned None and no default is configured")

        try:
            synthesizer = self.synthesizers[route]
        except (KeyError, TypeError):
            raise ValueError(f"Unknown TTS route: {route}") from None

        return await synthesizer.synthesize(
            text,
            style_info=style_info,
            language=language,
        )

    async def close(self):
        """Close the router and every registered synthesizer exactly once."""
        synthesizers = []
        seen = set()
        for synthesizer in self.synthesizers.values():
            identity = id(synthesizer)
            if identity not in seen:
                seen.add(identity)
                synthesizers.append(synthesizer)

        stack = AsyncExitStack()
        stack.push_async_callback(super().close)
        for synthesizer in synthesizers:
            stack.push_async_callback(synthesizer.close)
        await stack.aclose()
