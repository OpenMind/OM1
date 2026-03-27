"""Memory reader for loading and searching long-term memory.

Reads MEMORY.md for persistent facts and searches daily logs
using in-memory cosine similarity (no FAISS), following the
OpenClaw pattern with hash-cached embeddings.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fuser.knowledge_base.base_embedding import BaseEmbeddingClient
from fuser.knowledge_base.base_retriever import Document
from fuser.memory.indexer import MemoryIndex, build_index_from_daily_dir


class MemoryReader:
    """Read and search long-term memory files.

    Provides two retrieval mechanisms:
    1. MEMORY.md — read in full (small, persistent facts)
    2. Daily logs — cosine similarity search for relevant context

    Parameters
    ----------
    embedding_client : BaseEmbeddingClient
        Shared embedding client (reused from KnowledgeBase or OpenAI).
    memory_root : str or Path, optional
        Root directory containing MEMORY.md and daily/ subdirectory.
        Defaults to ``<project_root>/memory``.
    min_score : float
        Minimum cosine similarity score for daily log search.
    """

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        memory_root: Optional[str | Path] = None,
        min_score: float = 0.3,
    ):
        if memory_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            memory_root = project_root / "memory"

        self.memory_root = Path(memory_root)
        self.memory_file = self.memory_root / "MEMORY.md"
        self.daily_dir = self.memory_root / "daily"
        self.embedding_client = embedding_client
        self.min_score = min_score
        self.index: Optional[MemoryIndex] = None
        self._index_initialized = False

    async def ensure_index(self) -> MemoryIndex:
        """Lazy-load the memory index on first use.

        Builds the index from all existing daily files. Subsequent
        calls return the cached index.

        Returns
        -------
        MemoryIndex
            The in-memory embedding index.
        """
        if not self._index_initialized:
            self.index = await build_index_from_daily_dir(
                self.daily_dir, self.embedding_client
            )
            self._index_initialized = True
            logging.info(f"Memory: index initialized with {self.index.size} chunks")
        return self.index

    def read_memory_md(self, max_chars: int = 500) -> str:
        """Read MEMORY.md contents, truncated to max_chars.

        Parameters
        ----------
        max_chars : int
            Maximum number of characters to return.

        Returns
        -------
        str
            Contents of MEMORY.md, or empty string if not found.
        """
        if not self.memory_file.exists():
            return ""

        try:
            content = self.memory_file.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            content_lines = [
                line
                for line in lines
                if not line.startswith("# ") and not line.startswith("<!--")
            ]
            result = "\n".join(content_lines).strip()
            if len(result) > max_chars:
                result = result[:max_chars] + "..."
            return result
        except Exception as e:
            logging.error(f"Memory: failed to read MEMORY.md: {e}")
            return ""

    def get_latest_daily(self, max_chars: int = 500) -> str:
        """Get the latest entries from today's or yesterday's daily log.

        Parameters
        ----------
        max_chars : int
            Maximum characters to return.

        Returns
        -------
        str
            Recent daily log content, or empty string.
        """
        if not self.daily_dir.exists():
            return ""

        for days_ago in range(2):
            date = datetime.now() - timedelta(days=days_ago)
            daily_path = self.daily_dir / f"{date.strftime('%Y-%m-%d')}.md"
            if daily_path.exists():
                try:
                    content = daily_path.read_text(encoding="utf-8")
                    if len(content) > max_chars:
                        content = "..." + content[-max_chars:]
                    return content.strip()
                except Exception as e:
                    logging.error(f"Memory: failed to read {daily_path}: {e}")

        return ""

    async def search_daily(
        self, query_text: str, top_k: int = 1, min_score: Optional[float] = None
    ) -> list[Document]:
        """Search daily logs using cosine similarity.

        Parameters
        ----------
        query_text : str
            The query text (typically from voice input).
        top_k : int
            Number of top results to return.
        min_score : float, optional
            Minimum similarity score.

        Returns
        -------
        list of Document
            Matching document chunks with similarity scores.
        """
        if not query_text or not query_text.strip():
            return []

        score_threshold = min_score if min_score is not None else self.min_score

        try:
            index = await self.ensure_index()
            return await index.search(
                query_text, top_k=top_k, min_score=score_threshold
            )
        except Exception as e:
            logging.error(f"Memory: search failed: {e}")
            return []

    def format_context(
        self,
        memory_md: str,
        search_results: list[Document],
        latest: str,
        max_chars: int = 1000,
    ) -> str:
        """Format memory into a prompt-ready context string.

        Parameters
        ----------
        memory_md : str
            Contents of MEMORY.md.
        search_results : list of Document
            Search results from daily logs.
        latest : str
            Latest daily log entries.
        max_chars : int
            Maximum total characters.

        Returns
        -------
        str
            Formatted memory context for prompt injection.
        """
        parts = []
        total_chars = 0

        if memory_md:
            section = f"[Facts]\n{memory_md}"
            parts.append(section)
            total_chars += len(section)

        for doc in search_results:
            if total_chars >= max_chars:
                break
            source = doc.metadata.get("source", "unknown")
            score = doc.score if doc.score is not None else 0.0
            section = f"[Past: {source}] (score: {score:.2f})\n{doc.text}"
            if total_chars + len(section) > max_chars:
                break
            parts.append(section)
            total_chars += len(section)

        if latest and total_chars < max_chars:
            remaining = max_chars - total_chars
            truncated = latest[:remaining]
            section = f"[Recent]\n{truncated}"
            parts.append(section)

        return "\n\n".join(parts)
