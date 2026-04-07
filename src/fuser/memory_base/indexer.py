import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from fuser.knowledge_base.base_embedding import BaseEmbeddingClient
from fuser.knowledge_base.base_retriever import Document

DEFAULT_MIN_SCORE = 0.3
DEFAULT_VALID_DURATION_DAYS = 14


def _hash_text(text: str) -> str:
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MemoryIndex:
    """In-memory embedding index for memory chunks.

    Parameters
    ----------
    embedding_client : BaseEmbeddingClient
        Embedding client from knowledge base
    """

    def __init__(self, embedding_client: BaseEmbeddingClient):
        self.embedding_client = embedding_client
        self._cache: dict[str, tuple[np.ndarray, Document]] = {}

    @property
    def size(self) -> int:
        """Number of cached chunks."""
        return len(self._cache)

    async def search(self, query: str, top_k: int = 1, min_score: float = DEFAULT_MIN_SCORE) -> list[Document]:
        """Search for most similar chunks using cosine similarity.

        Parameters
        ----------
        query : str
            The search query text.
        top_k : int
            Number of top results to return.
        min_score : float
            Minimum similarity score threshold.

        Returns
        -------
        list of Document
            Top-k matching documents with similarity scores.
        """
        if not self._cache or not query.strip():
            return []

        try:
            async with self.embedding_client:
                query_embedding = await self.embedding_client.embed(query)
        except Exception as e:
            logging.error(f"Memory index: failed to embed query: {e}")
            return []

        scored: list[tuple[float, Document]] = []
        for embedding, doc in self._cache.values():
            score = _cosine_similarity(query_embedding, embedding)
            if score >= min_score:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            results.append(Document(text=doc.text, metadata=doc.metadata.copy(), score=score))

        logging.info(f"Memory search: '{query[:50]}' → {len(results)} results " f"(from {self.size} chunks)")
        return results

    async def load_chunks_batch(self, chunks: list[Document]) -> int:
        """Load multiple chunks into the index in batch (used at startup).

        Parameters
        ----------
        chunks : list of Document
            Chunks to load.

        Returns
        -------
        int
            Number of new chunks actually embedded.
        """
        new_chunks = []
        new_hashes = []
        for chunk in chunks:
            text_hash = _hash_text(chunk.text)
            if text_hash not in self._cache:
                new_chunks.append(chunk)
                new_hashes.append(text_hash)

        if not new_chunks:
            return 0

        try:
            texts = [c.text for c in new_chunks]
            async with self.embedding_client:
                embeddings = await self.embedding_client.embed_batch(texts)

            for i, (text_hash, chunk) in enumerate(zip(new_hashes, new_chunks)):
                self._cache[text_hash] = (embeddings[i], chunk)

            logging.info(f"Memory index: loaded {len(new_chunks)} chunks (total: {self.size})")
            return len(new_chunks)
        except Exception as e:
            logging.error(f"Memory index: batch embed failed: {e}")
            return 0


def parse_daily_file(filepath: Path) -> list[Document]:
    """Parse a daily markdown file into document chunks.

    Each ## section becomes a separate chunk.

    Parameters
    ----------
    filepath : Path
        Path to a daily log markdown file.

    Returns
    -------
    list of Document
        Parsed document chunks with metadata.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Memory: failed to read {filepath}: {e}")
        return []

    date_prefix = f"[Date: {filepath.stem}]"

    chunks: list[Document] = []
    current_chunk = ""
    chunk_start_line = 1

    for i, line in enumerate(content.split("\n"), 1):
        if line.startswith("## ") and current_chunk.strip():
            chunks.append(
                Document(
                    text=f"{date_prefix}\n{current_chunk.strip()}",
                    metadata={
                        "source": filepath.name,
                        "chunk_id": len(chunks),
                        "start_line": chunk_start_line,
                    },
                )
            )
            current_chunk = line + "\n"
            chunk_start_line = i
        else:
            current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append(
            Document(
                text=f"{date_prefix}\n{current_chunk.strip()}",
                metadata={
                    "source": filepath.name,
                    "chunk_id": len(chunks),
                    "start_line": chunk_start_line,
                },
            )
        )

    return chunks


async def build_index(
    index: MemoryIndex,
    daily_dir: Path,
    valid_duration: int = DEFAULT_VALID_DURATION_DAYS,
) -> None:
    """Build an existing MemoryIndex from recent daily markdown files.

    Expired files (older than ``valid_duration`` days) are deleted.

    Parameters
    ----------
    index : MemoryIndex
        The index instance to populate (mutated in place).
    daily_dir : Path
        Directory containing daily markdown files.
    valid_duration : int
        Number of days to keep. Older daily logs are deleted.
    """
    if not daily_dir.exists():
        return

    cutoff = datetime.now() - timedelta(days=valid_duration)
    all_chunks: list[Document] = []

    for daily_file in sorted(daily_dir.glob("*.md")):
        try:
            file_date = datetime.strptime(daily_file.stem, "%Y-%m-%d")
        except ValueError:
            continue

        if file_date < cutoff:
            daily_file.unlink()
            logging.info(f"Memory: deleted expired daily log {daily_file.name}")
            continue

        chunks = parse_daily_file(daily_file)
        all_chunks.extend(chunks)

    if all_chunks:
        count = await index.load_chunks_batch(all_chunks)
        logging.info(
            f"Memory: populated index with {len(all_chunks)} chunks, {count} new "
            f"(valid_duration: {valid_duration} days)"
        )
