"""Semantic memory provider using local embeddings and ChromaDB vector search."""

import logging
import os
import time
from typing import List, Optional

from providers.singleton import singleton

logger = logging.getLogger(__name__)

# Maximum characters per memory document
MAX_TEXT_LENGTH = 1000

# Default storage path relative to project root
DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "memory",
    "embeddings",
)


@singleton
class SemanticMemoryProvider:
    """Provides semantic memory storage and retrieval using embeddings.

    Uses sentence-transformers for local embedding generation and ChromaDB
    for persistent vector storage. Each robot mode gets its own collection.

    Parameters
    ----------
    None
        Configuration is done via the configure() method after instantiation.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.top_k: int = 3
        self.similarity_threshold: float = 0.3

        self._model = None
        self._chroma_client = None
        self._collections: dict = {}
        self._persist_dir: str = DEFAULT_PERSIST_DIR

    def configure(
        self,
        enabled: bool = False,
        top_k: int = 3,
        similarity_threshold: float = 0.3,
    ) -> None:
        """Configure the semantic memory provider.

        Parameters
        ----------
        enabled : bool
            Whether semantic memory is active.
        top_k : int
            Number of top results to return from retrieval.
        similarity_threshold : float
            Minimum cosine similarity score to include a result.
        """
        self.enabled = enabled
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        if self.enabled:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Lazy-load the embedding model and ChromaDB client."""
        if self._model is not None and self._chroma_client is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: all-MiniLM-L6-v2")
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.enabled = False
            return

        try:
            import chromadb

            os.makedirs(self._persist_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self._persist_dir)
            logger.info(f"ChromaDB initialized at {self._persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.enabled = False
            return

    def _get_collection(self, mode: str):
        """Get or create a ChromaDB collection for the given mode.

        Parameters
        ----------
        mode : str
            The robot operating mode name.

        Returns
        -------
        chromadb.Collection or None
            The ChromaDB collection, or None if not initialized.
        """
        if self._chroma_client is None:
            return None

        collection_name = f"om1_{mode}"
        if collection_name not in self._collections:
            self._collections[collection_name] = (
                self._chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            )
        return self._collections[collection_name]

    def store(
        self,
        sensory_input: str,
        action_response: str,
        mode: str,
        tick: int,
    ) -> None:
        """Store a sensory-action pair as a memory document.

        Parameters
        ----------
        sensory_input : str
            The formatted sensory input text.
        action_response : str
            The action response text from the LLM.
        mode : str
            The current robot operating mode.
        tick : int
            The current tick number.
        """
        if not self.enabled or self._model is None:
            return

        collection = self._get_collection(mode)
        if collection is None:
            return

        try:
            document = f"Input: {sensory_input[:MAX_TEXT_LENGTH]} | Response: {action_response[:MAX_TEXT_LENGTH]}"
            embedding = self._model.encode(document, normalize_embeddings=True).tolist()

            doc_id = f"tick_{tick}_{int(time.time() * 1000)}"

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[{"tick": tick, "timestamp": time.time(), "mode": mode}],
            )

            logger.debug(f"Stored memory: {doc_id} (mode={mode})")

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

    def retrieve(
        self,
        query: str,
        mode: str,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Retrieve relevant memories for a given query.

        Parameters
        ----------
        query : str
            The query text to find relevant memories for.
        mode : str
            The current robot operating mode.
        top_k : int, optional
            Override the default number of results. Defaults to self.top_k.

        Returns
        -------
        List[str]
            List of relevant memory document strings, ordered by similarity.
        """
        if not self.enabled or self._model is None:
            return []

        collection = self._get_collection(mode)
        if collection is None:
            return []

        try:
            if collection.count() == 0:
                return []

            k = top_k if top_k is not None else self.top_k
            k = min(k, collection.count())

            query_embedding = self._model.encode(
                query[:MAX_TEXT_LENGTH], normalize_embeddings=True
            ).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "distances"],
            )

            if not results or not results["documents"] or not results["documents"][0]:
                return []

            memories = []
            documents = results["documents"][0]
            # ChromaDB cosine distance = 1 - cosine_similarity
            distances = results["distances"][0] if results["distances"] else []

            for i, doc in enumerate(documents):
                if i < len(distances):
                    similarity = 1.0 - distances[i]
                    if similarity >= self.similarity_threshold:
                        memories.append(doc)
                else:
                    memories.append(doc)

            logger.debug(f"Retrieved {len(memories)} memories for mode={mode}")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def clear_mode(self, mode: str) -> None:
        """Clear all memories for a specific mode.

        Parameters
        ----------
        mode : str
            The robot operating mode to clear memories for.
        """
        if self._chroma_client is None:
            return

        collection_name = f"om1_{mode}"
        try:
            self._chroma_client.delete_collection(name=collection_name)
            self._collections.pop(collection_name, None)
            logger.info(f"Cleared memories for mode: {mode}")
        except Exception as e:
            logger.error(f"Failed to clear memories for mode {mode}: {e}")
