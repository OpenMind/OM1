"""In-memory embedding index for daily memory logs.

Maintains a hash-cached dictionary of chunk embeddings.
On search, does brute-force cosine similarity (no FAISS).
Follows the OpenClaw pattern: simple, no external index files.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np

from fuser.knowledge_base.base_embedding import BaseEmbeddingClient
from fuser.knowledge_base.base_retriever import Document


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

    Uses SHA-256 hash caching to avoid re-embedding unchanged chunks.
    Search is brute-force cosine similarity — fast enough for
    hundreds to low thousands of chunks.

    Parameters
    ----------
    embedding_client : BaseEmbeddingClient
        Shared embedding client for generating vectors.
    """

    def __init__(self, embedding_client: BaseEmbeddingClient):
        self.embedding_client = embedding_client
        # hash → (embedding vector, Document)
        self._cache: dict[str, tuple[np.ndarray, Document]] = {}

    @property
    def size(self) -> int:
        """Number of cached chunks."""
        return len(self._cache)

    async def add_chunk(self, text: str, metadata: dict) -> None:
        """Add a single chunk to the index (incremental).

        Skips if the chunk hash already exists in cache.

        Parameters
        ----------
        text : str
            The chunk text content.
        metadata : dict
            Chunk metadata (source file, line numbers, etc.).
        """
        text_hash = _hash_text(text)
        if text_hash in self._cache:
            return  # Already indexed, skip

        try:
            async with self.embedding_client:
                embedding = await self.embedding_client.embed(text)
            doc = Document(text=text, metadata=metadata)
            self._cache[text_hash] = (embedding, doc)
            logging.debug(f"Memory index: added chunk (total: {self.size})")
        except Exception as e:
            logging.error(f"Memory index: failed to embed chunk: {e}")

    async def add_chunks_batch(self, chunks: list[Document]) -> int:
        """Add multiple chunks to the index in batch.

        Parameters
        ----------
        chunks : list of Document
            Chunks to add.

        Returns
        -------
        int
            Number of new chunks actually embedded (excluding cached).
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

            logging.info(
                f"Memory index: added {len(new_chunks)} chunks (total: {self.size})"
            )
            return len(new_chunks)
        except Exception as e:
            logging.error(f"Memory index: batch embed failed: {e}")
            return 0

    async def search(
        self, query: str, top_k: int = 1, min_score: float = 0.3
    ) -> list[Document]:
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

        # Brute-force cosine similarity against all cached embeddings
        scored: list[tuple[float, Document]] = []
        for embedding, doc in self._cache.values():
            score = _cosine_similarity(query_embedding, embedding)
            if score >= min_score:
                scored.append((score, doc))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            results.append(
                Document(text=doc.text, metadata=doc.metadata.copy(), score=score)
            )

        logging.info(
            f"Memory search: '{query[:50]}' → {len(results)} results "
            f"(from {self.size} chunks)"
        )
        return results


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

    chunks: list[Document] = []
    current_chunk = ""
    chunk_start_line = 1

    for i, line in enumerate(content.split("\n"), 1):
        if line.startswith("## ") and current_chunk.strip():
            chunks.append(
                Document(
                    text=current_chunk.strip(),
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
                text=current_chunk.strip(),
                metadata={
                    "source": filepath.name,
                    "chunk_id": len(chunks),
                    "start_line": chunk_start_line,
                },
            )
        )

    return chunks


async def build_index_from_daily_dir(
    daily_dir: Path, embedding_client: BaseEmbeddingClient
) -> MemoryIndex:
    """Build a MemoryIndex from all daily markdown files.

    Parameters
    ----------
    daily_dir : Path
        Directory containing daily markdown files.
    embedding_client : BaseEmbeddingClient
        Client for generating embeddings.

    Returns
    -------
    MemoryIndex
        Populated index ready for search.
    """
    index = MemoryIndex(embedding_client)

    if not daily_dir.exists():
        return index

    all_chunks: list[Document] = []
    for daily_file in sorted(daily_dir.glob("*.md")):
        chunks = parse_daily_file(daily_file)
        all_chunks.extend(chunks)

    if all_chunks:
        count = await index.add_chunks_batch(all_chunks)
        logging.info(
            f"Memory: built index from {len(all_chunks)} total chunks, {count} new"
        )

    return index
