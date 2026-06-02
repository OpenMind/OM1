import json
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
from providers.singleton import singleton

DEFAULT_CONTEXT_MAX_CHARS = 1000


@singleton
class MemoryReader:
    """Read and search long-term memory files.

    1. Per-user facts — from users/{id}/facts.json
    2. Daily logs — top relevant chunks via vector search

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
        base_url: str = "http://10.1.10.221:8100",
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        if memory_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            memory_root = project_root / "memory"

        self.memory_root = Path(memory_root)
        self.daily_dir = self.memory_root / "daily"
        self.users_dir = self.memory_root / "users"
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

    async def search_daily(
        self, query_text: str, top_k: int = 3, min_score: Optional[float] = None, user_id: Optional[str] = None
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
        user_id : str, optional
            If provided, filter results to this user's interactions.

        Returns
        -------
        list of Document
            Matching document chunks with similarity scores.
        """
        if not query_text or not query_text.strip():
            return []

        score_threshold = min_score if min_score is not None else self.min_score

        try:
            return await self.index.search(query_text, top_k=top_k, min_score=score_threshold, user_id=user_id)
        except Exception as e:
            logging.error(f"Memory: search failed: {e}")
            return []

    def format_context(
        self,
        search_results: list[Document],
        max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
        user_id: Optional[str] = None,
    ) -> str:
        """Format memory into a prompt-ready context string.

        Parameters
        ----------
        search_results : list of Document
            Search results from daily logs with date/time metadata.
        max_chars : int
            Maximum total characters.
        user_id : str, optional
            If provided, prepend user-specific facts.

        Returns
        -------
        str
            Formatted memory context for prompt injection.
        """
        parts = []
        total_chars = 0

        if user_id:
            user_facts = self.read_user_facts(user_id)
            if user_facts:
                section = f"[User: {user_id}]\n{user_facts}"
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

    def read_user_profile(self, user_id: str) -> Optional[dict]:
        """Read a user's profile.json."""
        profile_path = self.users_dir / user_id / "profile.json"
        if not profile_path.exists():
            return None
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Memory: failed to read profile for {user_id}: {e}")
            return None

    def read_user_facts(self, user_id: str) -> str:
        """Read a user's facts.json and format as a prompt string."""
        facts_path = self.users_dir / user_id / "facts.json"
        if not facts_path.exists():
            return ""
        try:
            data = json.loads(facts_path.read_text(encoding="utf-8"))
            facts = data.get("facts", [])
            if not facts:
                return ""
            lines = [f"- [{f.get('category', 'FACT')}] {f['fact']}" for f in facts if f.get("fact")]
            return "\n".join(lines)
        except Exception as e:
            logging.warning(f"Memory: failed to read facts for {user_id}: {e}")
            return ""
