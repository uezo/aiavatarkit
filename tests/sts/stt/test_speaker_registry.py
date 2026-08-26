import asyncio
import json
import threading

import numpy as np
import pytest

from aiavatar.sts.stt.speaker_registry import base as speaker_registry_module
from aiavatar.sts.stt.speaker_registry.base import InMemoryStore, SpeakerRegistry
from aiavatar.sts.stt.speaker_registry.postgres import PGVectorStore


class FakePGConnection:
    def __init__(self):
        self.speakers = {}

    async def execute(self, query, *args):
        if "CREATE TABLE" in query:
            return "CREATE TABLE"
        if "INSERT INTO" in query:
            speaker_id, embedding, metadata = args
            current_metadata = self.speakers.get(speaker_id, {}).get("metadata", {})
            self.speakers[speaker_id] = {
                "embedding": embedding,
                "metadata": {**current_metadata, **json.loads(metadata)},
            }
            return "INSERT 0 1"
        if "UPDATE" in query:
            metadata, speaker_id = args
            if speaker_id not in self.speakers:
                return "UPDATE 0"
            self.speakers[speaker_id]["metadata"].update(json.loads(metadata))
            return "UPDATE 1"
        raise AssertionError(f"Unexpected execute query: {query}")

    async def fetchrow(self, query, *args):
        if "COUNT(*)" in query:
            return (len(self.speakers),)
        if "SELECT metadata::text AS metadata" in query:
            (speaker_id,) = args
            speaker = self.speakers.get(speaker_id)
            if speaker is None:
                return None
            return {
                "metadata": json.dumps(speaker["metadata"]),
            }
        if "metadata::text AS metadata" in query:
            (speaker_id,) = args
            speaker = self.speakers.get(speaker_id)
            if speaker is None:
                return None
            return {
                "embedding": speaker["embedding"],
                "metadata": json.dumps(speaker["metadata"]),
            }
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def fetch(self, query, *args):
        if "ORDER BY embedding" in query:
            return [
                {"id": speaker_id, "embedding": speaker["embedding"]}
                for speaker_id, speaker in self.speakers.items()
            ]
        if "metadata::text AS metadata" in query:
            return [
                {
                    "id": speaker_id,
                    "embedding": speaker["embedding"],
                    "metadata": json.dumps(speaker["metadata"]),
                }
                for speaker_id, speaker in self.speakers.items()
            ]
        raise AssertionError(f"Unexpected fetch query: {query}")


class FakePGAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakePGPool:
    def __init__(self):
        self.connection = FakePGConnection()

    def acquire(self):
        return FakePGAcquire(self.connection)


@pytest.mark.asyncio
async def test_in_memory_registry_uses_async_store_contract():
    store = InMemoryStore()
    registry = SpeakerRegistry(match_threshold=0.8, store=store)
    embedding = np.array([1.0, 0.0], dtype=np.float32)

    results = await asyncio.gather(
        *(registry.match_topk_from_embedding(embedding) for _ in range(4))
    )

    assert sum(result.chosen.is_new for result in results) == 1
    assert len({result.chosen.speaker_id for result in results}) == 1
    assert await store.count() == 1

    speaker_id = results[0].chosen.speaker_id
    await registry.set_metadata(speaker_id, "name", "Alice")

    matched = await registry.match_topk_from_embedding(embedding)

    assert matched.chosen.speaker_id == speaker_id
    assert matched.chosen.metadata == {"name": "Alice"}
    assert await registry.get_metadata(speaker_id, "name") == "Alice"
    assert [item[0] async for item in store.all_items()] == [speaker_id]


@pytest.mark.asyncio
async def test_pgvector_registry_decodes_jsonb_metadata():
    pool = FakePGPool()

    async def get_pool():
        return pool

    store = PGVectorStore(get_pool=get_pool)
    registry = SpeakerRegistry(match_threshold=0.8, store=store)
    embedding = np.array([1.0, 0.0], dtype=np.float32)

    first = await registry.match_topk_from_embedding(embedding)
    matched_without_metadata = await registry.match_topk_from_embedding(embedding)
    await registry.set_metadata(first.chosen.speaker_id, "name", "Alice")
    matched = await registry.match_topk_from_embedding(embedding)
    name = await registry.get_metadata(first.chosen.speaker_id, "name")
    items = [item async for item in store.all_items()]

    assert first.chosen.is_new is True
    assert matched_without_metadata.chosen.metadata == {}
    assert matched.chosen.speaker_id == first.chosen.speaker_id
    assert matched.chosen.metadata == {"name": "Alice"}
    assert name == "Alice"
    assert len(items) == 1
    assert items[0][0] == first.chosen.speaker_id
    assert items[0][2] == {"name": "Alice"}


@pytest.mark.asyncio
async def test_pcm_embedding_runs_in_worker_thread(monkeypatch):
    worker_thread_ids = []

    class FakeVoiceEncoder:
        def __init__(self):
            worker_thread_ids.append(threading.get_ident())

        def embed_utterance(self, wav):
            worker_thread_ids.append(threading.get_ident())
            return np.array([1.0, 0.0], dtype=np.float32)

    def fake_preprocess_wav(wav, source_sr):
        worker_thread_ids.append(threading.get_ident())
        return wav

    monkeypatch.setattr(speaker_registry_module, "VoiceEncoder", FakeVoiceEncoder)
    monkeypatch.setattr(speaker_registry_module, "preprocess_wav", fake_preprocess_wav)
    registry = SpeakerRegistry(store=InMemoryStore())
    event_loop_thread_id = threading.get_ident()

    result = await registry.match_topk_from_pcm(
        np.zeros(16, dtype=np.int16).tobytes(),
        sample_rate=16000,
    )

    assert result.chosen.is_new is True
    assert worker_thread_ids
    assert all(thread_id != event_loop_thread_id for thread_id in worker_thread_ids)


@pytest.mark.asyncio
async def test_in_memory_heavy_work_runs_in_worker_thread(tmp_path, monkeypatch):
    store = InMemoryStore()
    store.data_path = str(tmp_path / "speakers")
    event_loop_thread_id = threading.get_ident()
    save_thread_ids = []
    similarity_thread_ids = []

    original_save_to_file = store._save_to_file
    original_topk_similarity = store._topk_similarity

    def observed_save_to_file():
        save_thread_ids.append(threading.get_ident())
        original_save_to_file()

    def observed_topk_similarity(q_norm, k):
        similarity_thread_ids.append(threading.get_ident())
        return original_topk_similarity(q_norm, k)

    monkeypatch.setattr(store, "_save_to_file", observed_save_to_file)
    monkeypatch.setattr(store, "_topk_similarity", observed_topk_similarity)

    embedding = np.array([1.0, 0.0], dtype=np.float32)
    await store.upsert("speaker", embedding)
    await store.topk_similarity(embedding, 1)

    assert save_thread_ids
    assert similarity_thread_ids
    assert all(thread_id != event_loop_thread_id for thread_id in save_thread_ids)
    assert all(thread_id != event_loop_thread_id for thread_id in similarity_thread_ids)
