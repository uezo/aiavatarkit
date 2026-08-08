from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..base import SpeechSynthesizer


class TTSPostprocessor(ABC):
    @abstractmethod
    async def process(
        self,
        audio: bytes,
        *,
        synthesizer: "SpeechSynthesizer",
    ) -> bytes:
        pass

    def get_cache_config(
        self,
        synthesizer: "SpeechSynthesizer",
    ) -> Optional[dict]:
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        }
