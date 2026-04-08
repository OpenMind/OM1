from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.fuser.knowledge_base.base_retriever import Document
from src.fuser.memory_base.indexer import (
    MemoryIndex,
    _cosine_similarity,
    _hash_text,
    parse_daily_file,
)


def _make_embedding_client(dim: int = 64):
    """Mock embedding client that returns deterministic vectors."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    async def _embed(query: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(query)) % (2**31))
        v = rng.standard_normal(dim).astype("float32")
        return v / np.linalg.norm(v)

    async def _embed_batch(queries: list[str]) -> np.ndarray:
        vecs = [await _embed(q) for q in queries]
        return np.stack(vecs) if vecs else np.empty((0, dim), dtype="float32")

    client.embed = AsyncMock(side_effect=_embed)
    client.embed_batch = AsyncMock(side_effect=_embed_batch)
    return client


def _make_doc(text: str, source: str = "2026-04-01.md") -> Document:
    return Document(text=text, metadata={"source": source, "chunk_id": 0, "start_line": 0})


class TestHashText:
    def test_same_text_same_hash(self):
        assert _hash_text("hello") == _hash_text("hello")

    def test_different_text_different_hash(self):
        assert _hash_text("hello") != _hash_text("world")

    def test_returns_64_char_hex(self):
        h = _hash_text("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_string(self):
        h = _hash_text("")
        assert len(h) == 64


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype="float32")
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0], dtype="float32")
        b = np.array([0.0, 1.0], dtype="float32")
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors_score_minus_one(self):
        v = np.array([1.0, 0.0], dtype="float32")
        assert abs(_cosine_similarity(v, -v) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        z = np.zeros(3, dtype="float32")
        v = np.array([1.0, 0.0, 0.0], dtype="float32")
        assert _cosine_similarity(z, v) == 0.0
        assert _cosine_similarity(v, z) == 0.0


class TestMemoryIndexSize:
    def test_empty_on_init(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        assert index.size == 0

    @pytest.mark.asyncio
    async def test_size_after_load(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        docs = [_make_doc("chunk one"), _make_doc("chunk two")]
        await index.load_chunks_batch(docs)
        assert index.size == 2


class TestMemoryIndexLoadBatch:
    @pytest.mark.asyncio
    async def test_loads_chunks(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        docs = [_make_doc("alpha"), _make_doc("beta"), _make_doc("gamma")]
        loaded = await index.load_chunks_batch(docs)
        assert loaded == 3
        assert index.size == 3

    @pytest.mark.asyncio
    async def test_deduplicates_identical_text(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        loaded1 = await index.load_chunks_batch([_make_doc("same text")])
        assert loaded1 == 1
        loaded2 = await index.load_chunks_batch([_make_doc("same text")])
        assert loaded2 == 0
        assert index.size == 1

    @pytest.mark.asyncio
    async def test_empty_batch_returns_zero(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        loaded = await index.load_chunks_batch([])
        assert loaded == 0
        assert index.size == 0

    @pytest.mark.asyncio
    async def test_incremental_loads_deduplicate(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        await index.load_chunks_batch([_make_doc("a"), _make_doc("b")])
        loaded2 = await index.load_chunks_batch([_make_doc("a"), _make_doc("c")])
        assert loaded2 == 1  # only "c" is new
        assert index.size == 3

    @pytest.mark.asyncio
    async def test_embed_batch_error_returns_zero(self):
        client = _make_embedding_client()
        client.embed_batch = AsyncMock(side_effect=RuntimeError("embed failed"))
        index = MemoryIndex(client)
        loaded = await index.load_chunks_batch([_make_doc("fail")])
        assert loaded == 0
        assert index.size == 0


class TestMemoryIndexSearch:
    @pytest.mark.asyncio
    async def test_empty_index_returns_empty(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        results = await index.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_blank_query_returns_empty(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        await index.load_chunks_batch([_make_doc("content")])
        results = await index.search("   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_at_most_top_k(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        docs = [_make_doc(f"chunk {i}") for i in range(10)]
        await index.load_chunks_batch(docs)
        results = await index.search("some query", top_k=3, min_score=0.0)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_results_have_scores(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        await index.load_chunks_batch([_make_doc("hello world")])
        results = await index.search("hello", top_k=1, min_score=0.0)
        if results:
            assert results[0].score is not None

    @pytest.mark.asyncio
    async def test_min_score_filters_results(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        await index.load_chunks_batch([_make_doc("cats and dogs")])
        results = await index.search("unrelated query xyz", top_k=5, min_score=0.999)
        assert results == []

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_desc(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        docs = [_make_doc(f"doc {i}") for i in range(5)]
        await index.load_chunks_batch(docs)
        results = await index.search("query", top_k=5, min_score=0.0)
        scores = [r.score for r in results if r.score is not None]
        assert len(scores) == len(results), "all results should have a score"
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_returned_doc_is_copy(self):
        """Score should be set on a copy, not mutate the cached original."""
        client = _make_embedding_client()
        index = MemoryIndex(client)
        await index.load_chunks_batch([_make_doc("original")])
        results = await index.search("original", top_k=1, min_score=0.0)
        if results:
            cached_doc = list(index._cache.values())[0][1]
            assert cached_doc.score is None

    @pytest.mark.asyncio
    async def test_embed_error_returns_empty(self):
        client = _make_embedding_client()
        index = MemoryIndex(client)
        await index.load_chunks_batch([_make_doc("data")])
        client.embed = AsyncMock(side_effect=RuntimeError("embed failed"))
        results = await index.search("query", top_k=1, min_score=0.0)
        assert results == []


class TestParseDailyFile:
    def test_date_prefix_in_chunk_text(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text("## 10:00:00\n- **User**: hello\n- **Robot**: hi\n", encoding="utf-8")
        chunks = parse_daily_file(f)
        assert len(chunks) == 1
        assert chunks[0].text.startswith("[Date: 2026-04-06]")

    def test_chunk_contains_conversation(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text("## 10:00:00\n- **User**: hello\n- **Robot**: hi\n", encoding="utf-8")
        chunks = parse_daily_file(f)
        assert "hello" in chunks[0].text
        assert "hi" in chunks[0].text

    def test_multiple_sections_produce_multiple_chunks(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text(
            "## 10:00:00\n- **User**: first\n\n## 11:00:00\n- **User**: second\n",
            encoding="utf-8",
        )
        chunks = parse_daily_file(f)
        assert len(chunks) == 2
        assert all(c.text.startswith("[Date: 2026-04-06]") for c in chunks)

    def test_source_metadata_is_filename(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text("## 10:00:00\n- **User**: hello\n", encoding="utf-8")
        chunks = parse_daily_file(f)
        assert chunks[0].metadata["source"] == "2026-04-06.md"

    def test_empty_file_returns_no_chunks(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text("", encoding="utf-8")
        assert parse_daily_file(f) == []

    def test_missing_file_returns_no_chunks(self, tmp_path):
        f = tmp_path / "missing.md"
        assert parse_daily_file(f) == []

    def test_content_before_first_heading_is_chunk(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text("preamble text\n## 10:00:00\n- **User**: hello\n", encoding="utf-8")
        chunks = parse_daily_file(f)
        assert len(chunks) == 2
        assert "preamble" in chunks[0].text

    def test_chunk_ids_increment(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text(
            "## 10:00:00\n- first\n\n## 11:00:00\n- second\n\n## 12:00:00\n- third\n",
            encoding="utf-8",
        )
        chunks = parse_daily_file(f)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids == [0, 1, 2]

    def test_start_line_metadata(self, tmp_path):
        f = tmp_path / "2026-04-06.md"
        f.write_text("## 10:00:00\n- first\n## 11:00:00\n- second\n", encoding="utf-8")
        chunks = parse_daily_file(f)
        assert chunks[0].metadata["start_line"] == 1
        assert chunks[1].metadata["start_line"] == 3
