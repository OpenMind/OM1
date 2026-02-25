"""Base class for embedding client implementations."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingClient(ABC):
    """
    Abstract base class for embedding client implementations.

    Embedding clients are responsible for converting text into vector embeddings
    using various backends (local models, API services, etc.).
    """

    def __init__(self, **kwargs):
        """
        Initialize the embedding client.

        Parameters
        ----------
        **kwargs
            Backend-specific configuration parameters.
        """
        pass

    @abstractmethod
    async def embed(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Parameters
        ----------
        query : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, queries: list[str]) -> np.ndarray:
        """
        Embed multiple query strings in a single batch.

        Parameters
        ----------
        queries : list of str
            List of texts to embed.

        Returns
        -------
        np.ndarray
            Embedding matrix (shape: [len(queries), embedding_dim]).
        """
        pass

    async def __aenter__(self):
        """Context manager entry - can be overridden by subclasses if needed."""
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - can be overridden by subclasses if needed."""
        pass
