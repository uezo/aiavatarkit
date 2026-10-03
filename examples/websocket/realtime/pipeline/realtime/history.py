"""Context storage initialization, portable history loading, and queued writes."""

import asyncio
import logging

from aiavatar.sts.llm.context_manager import ContextManager, SQLiteContextManager
from .turn import TurnHistory, TurnHistoryTracker

logger = logging.getLogger(__name__)


class HistoryLoadError(RuntimeError):
    def __init__(self):
        super().__init__("Realtime context history loading failed")


class HistoryStore:
    """Shared context backend and its optional lazy SQLite initialization."""

    def __init__(self, manager: ContextManager | None, transcription_timeout: float):
        self.manager = manager
        self.transcription_timeout = transcription_timeout
        self._initialization: asyncio.Task | None = None

    async def _initialize(self, db_path: str):
        self.manager = await asyncio.to_thread(SQLiteContextManager, db_path=db_path)

    async def ensure_manager(self, db_path: str) -> ContextManager:
        if self.manager is not None:
            return self.manager
        if self._initialization is None:
            self._initialization = asyncio.create_task(self._initialize(db_path))
        task = self._initialization
        try:
            # A canceled session must not cancel another session's backend
            # initialization or abandon an already-running SQLite thread.
            await asyncio.shield(task)
            return self.manager
        finally:
            if task.done() and self._initialization is task:
                self._initialization = None

    async def join_initialization(self):
        task = self._initialization
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.gather(task, return_exceptions=True)
                raise
            except Exception:
                logger.warning("Realtime context initialization failed")
            finally:
                if self._initialization is task:
                    self._initialization = None

    async def load_messages(self, context_id: str, limit: int) -> list[dict]:
        try:
            histories = await self.manager.get_histories(context_id, limit=limit)
            if not isinstance(histories, (list, tuple)):
                raise ValueError("Unexpected context history format")
        except Exception:
            raise HistoryLoadError() from None

        messages = []
        # Replay only portable conversation text, without stored provider IDs,
        # timestamps, instructions, or tool protocol fields.
        for message in histories[-limit:]:
            if not isinstance(message, dict) or message.get("type") not in (None, "message"):
                continue
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = message.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    part["text"] for part in content if isinstance(part, dict)
                    and part.get("type") in ("text", "input_text", "output_text")
                    and isinstance(part.get("text"), str)
                )
            else:
                continue
            if text.strip():
                messages.append({"role": role, "content": text})
        return messages


class HistoryWriter:
    """One session's ordered writes and their worker lifecycle."""

    def __init__(self, store: HistoryStore, context_id: str):
        self.store = store
        self.context_id = context_id
        self.turns = TurnHistoryTracker()
        self.queue: asyncio.Queue | None = None
        self.worker: asyncio.Task | None = None

    def start(self):
        if self.worker is None:
            self.queue = asyncio.Queue(maxsize=128)
            self.worker = asyncio.create_task(self.run())

    def submit(self, history: TurnHistory) -> bool:
        if self.queue is None:
            return False
        try:
            self.queue.put_nowait(history)
        except asyncio.QueueFull:
            logger.warning("Realtime context queue is full; turn history was not queued")
            return False
        self.turns.retain(history)
        return True

    async def run(self):
        manager = self.store.manager
        while True:
            history = await self.queue.get()
            try:
                if history is None:
                    return
                await history.response_ready.wait()
                if not history.input_ready.is_set():
                    remaining = history.remaining(
                        asyncio.get_running_loop().time(), self.store.transcription_timeout,
                    )
                    try:
                        await asyncio.wait_for(history.input_ready.wait(), remaining)
                    except asyncio.TimeoutError:
                        history.accept_transcript(failed=True)
                        logger.warning("Realtime input transcription timed out for context recording")
                messages = history.messages()
                if messages:
                    try:
                        await manager.add_histories(self.context_id, messages, "openai_realtime")
                    except Exception:
                        # Backend exceptions can contain credentials or history.
                        logger.warning("Realtime context recording failed")
            finally:
                if history is not None:
                    self.turns.discard(history)
                self.queue.task_done()

    async def close(self, timeout: float):
        if self.worker is None:
            return
        # A disconnected session cannot supply missing transcripts or output.
        self.turns.close()
        try:
            async with asyncio.timeout(timeout):
                await self.queue.put(None)
                await asyncio.shield(self.worker)
        except asyncio.TimeoutError:
            logger.warning("Realtime context recording did not finish before close_timeout")
        finally:
            if not self.worker.done():
                self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.turns.clear()
            while not self.queue.empty():
                self.queue.get_nowait()
                self.queue.task_done()
