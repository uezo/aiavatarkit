import abc
import base64
from collections import defaultdict
import logging
from pathlib import Path
import time
from typing import Optional
import aiofiles
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)


class HistoryProvider(abc.ABC):
    """Abstract base class for image history backends.

    Subclass this to implement custom storage (e.g., S3, Redis).
    """

    def __init__(self, image_dir: Optional[str] = None, image_size: Optional[int] = None):
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size

    async def _save_image(self, context_id: str, image_bytes: bytes, image_id: str):
        """Save original image to disk."""
        dir_path = self.image_dir / context_id
        dir_path.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(dir_path / f"{image_id}.jpg", "wb") as f:
            await f.write(image_bytes)

    def _resize_image(self, image_bytes: bytes) -> bytes:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        long_edge = max(img.size)
        if long_edge <= self.image_size:
            return image_bytes
        scale = self.image_size / long_edge
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    @abc.abstractmethod
    async def store_image(self, context_id: str, image_bytes: bytes, image_id: str) -> str:
        """Store an image and return a URL (data URL, HTTP URL, etc.) for the LLM."""
        ...

    @abc.abstractmethod
    def get(self, context_id: str) -> list[tuple[float, str, str]]:
        """Return history entries as [(timestamp, image_url, raw_text), ...]."""
        ...

    @abc.abstractmethod
    def add(self, context_id: str, timestamp: float, image_url: str, raw_text: str):
        """Append a new entry to the history."""
        ...

    @abc.abstractmethod
    def cleanup(self):
        """Evict stale entries (called before each request)."""
        ...


class InlineMemoryHistoryProvider(HistoryProvider):
    """In-memory image history with base64 data URLs. Simple, no HTTP serving needed."""

    def __init__(
        self,
        max_size: int = 20,
        truncate_to: int = 5,
        ttl: float = 600.0,
        image_dir: Optional[str] = "vision_stream_images",
        image_size: Optional[int] = 1024,
    ):
        super().__init__(image_dir=image_dir, image_size=image_size)
        self.max_size = max_size
        self.truncate_to = truncate_to
        self.ttl = ttl
        self._store: dict[str, list] = defaultdict(list)

    async def store_image(self, context_id: str, image_bytes: bytes, image_id: str) -> str:
        if self.image_dir:
            await self._save_image(context_id, image_bytes, image_id)

        if self.image_size:
            image_bytes = self._resize_image(image_bytes)

        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def get(self, context_id: str) -> list[tuple[float, str, str]]:
        return self._store[context_id]

    def add(self, context_id: str, timestamp: float, image_url: str, raw_text: str):
        history = self._store[context_id]
        history.append((timestamp, image_url, raw_text))
        if len(history) >= self.max_size:
            logger.info(
                f"[{context_id}] History truncated: {len(history)} -> {self.truncate_to}"
            )
            self._store[context_id] = history[-self.truncate_to:]

    def cleanup(self):
        now = time.time()
        expired = [
            cid for cid, history in self._store.items()
            if history and (now - history[-1][0]) > self.ttl
        ]
        for cid in expired:
            logger.info(f"Evicting stale image history: {cid}")
            del self._store[cid]


class LinkedFileHistoryProvider(HistoryProvider):
    """File-based image history served via HTTP. OpenAI can cache by URL."""

    def __init__(
        self,
        base_url: str,
        image_dir: str = "vision_stream_images",
        image_path: str = "/vision/images",
        max_size: int = 20,
        truncate_to: int = 5,
        ttl: float = 600.0,
        image_size: Optional[int] = 1024,
    ):
        super().__init__(image_dir=image_dir, image_size=image_size)
        self.base_url = base_url.rstrip("/")
        self.image_path = image_path
        self.max_size = max_size
        self.truncate_to = truncate_to
        self.ttl = ttl
        self._store: dict[str, list] = defaultdict(list)

    async def store_image(self, context_id: str, image_bytes: bytes, image_id: str) -> str:
        await self._save_image(context_id, image_bytes, image_id)
        # Save resized version and return its URL if image_size is set
        if self.image_size:
            resized = self._resize_image(image_bytes)
            async with aiofiles.open(self.image_dir / context_id / f"{image_id}_sm.jpg", "wb") as f:
                await f.write(resized)
            return f"{self.base_url}{self.image_path}/{context_id}/{image_id}_sm"
        return f"{self.base_url}{self.image_path}/{context_id}/{image_id}"

    def get(self, context_id: str) -> list[tuple[float, str, str]]:
        return self._store[context_id]

    def add(self, context_id: str, timestamp: float, image_url: str, raw_text: str):
        history = self._store[context_id]
        history.append((timestamp, image_url, raw_text))
        if len(history) >= self.max_size:
            logger.info(
                f"[{context_id}] History truncated: {len(history)} -> {self.truncate_to}"
            )
            self._store[context_id] = history[-self.truncate_to:]

    def cleanup(self):
        now = time.time()
        expired = [
            cid for cid, history in self._store.items()
            if history and (now - history[-1][0]) > self.ttl
        ]
        for cid in expired:
            logger.info(f"Evicting stale image history: {cid}")
            del self._store[cid]

    def get_router(self) -> APIRouter:
        router = APIRouter()
        image_dir = self.image_dir

        @router.get(f"{self.image_path}/{{context_id}}/{{image_id}}")
        async def get_image(context_id: str, image_id: str):
            file_path = image_dir / context_id / f"{image_id}.jpg"
            if not file_path.is_file():
                raise HTTPException(status_code=404)
            return FileResponse(file_path, media_type="image/jpeg")

        return router
