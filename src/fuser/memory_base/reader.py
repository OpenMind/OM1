import logging
from pathlib import Path
from typing import Optional

from fuser.knowledge_base.base_retriever import Document
from fuser.knowledge_base.faiss.embedding_client import EmbeddingClient
from fuser.memory_base.indexer import (
    DEFAULT_MIN_SCORE,
    MemoryIndex,
    build_index,
)

DEFAULT_MEMORY_MD_CHARS = 500
DEFAULT_CONTEXT_MAX_CHARS = 1000


class MemoryReader:
    """Read and search long-term memory files.

    1. MEMORY.md — read in full (include facts)
    2. Daily logs — top 3 relevant chunks

    Parameters
    ----------
    memory_root : str or Path, optional
        Root directory for memory storage.
    base_url : str
        Base URL for the embedding service.
    min_score : float
        Minimum similarity score threshold for search results.
    """

    def __init__(
        self,
        memory_root: Optional[str | Path] = None,
        base_url: str = "http://localhost:8100",
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        if memory_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            memory_root = project_root / "memory"

        self.memory_root = Path(memory_root)
        self.memory_file = self.memory_root / "MEMORY.md"
        self.daily_dir = self.memory_root / "daily"
        self.embedding_client = EmbeddingClient(base_url=base_url)
        self.min_score = min_score
        self.index = MemoryIndex(self.embedding_client)
        self._index_initialized = False

    async def ensure_index(self) -> MemoryIndex:
        """Initialize the memory index on first use.

        Build the index from all existing daily files for one time.

        Returns
        -------
        MemoryIndex
            The in-memory embedding index.
        """
        if not self._index_initialized:
            logging.info("Memory: building index...")
            await build_index(self.index, self.daily_dir)
            self._index_initialized = True
            logging.info(f"Memory: index initialized with {self.index.size} chunks")
        return self.index

    def read_memory_md(self, max_chars: int = DEFAULT_MEMORY_MD_CHARS) -> str:
        """Read MEMORY.md, truncated to max_chars.

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
            content_lines = [line for line in lines if not line.startswith("# ") and not line.startswith("<!--")]
            result = "\n".join(content_lines).strip()
            if len(result) > max_chars:
                result = result[:max_chars] + "..."
            return result
        except Exception as e:
            logging.error(f"Memory: failed to read MEMORY.md: {e}")
            return ""

    async def search_daily(self, query_text: str, top_k: int = 3, min_score: Optional[float] = None) -> list[Document]:
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
            return await self.index.search(query_text, top_k=top_k, min_score=score_threshold)
        except Exception as e:
            logging.error(f"Memory: search failed: {e}")
            return []

    def format_context(
        self,
        memory_md: str,
        search_results: list[Document],
        max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    ) -> str:
        """Format memory into a prompt-ready context string.

        Parameters
        ----------
        memory_md : str
            Contents of MEMORY.md.
        search_results : list of Document
            Search results from daily logs with date/time metadata.
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
            section = doc.text
            if total_chars + len(section) > max_chars:
                break
            parts.append(section)
            total_chars += len(section)

        return "\n\n".join(parts)
