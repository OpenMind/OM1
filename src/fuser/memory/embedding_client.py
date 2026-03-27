"""OpenAI Embedding client for memory search.

Uses OpenAI's text-embedding-3-small model to generate embeddings.
Implements the same BaseEmbeddingClient interface as the existing
sidecar-based EmbeddingClient, so it can be used as a drop-in replacement.
"""

import logging
import os
from typing import Optional

import numpy as np

from fuser.knowledge_base.base_embedding import BaseEmbeddingClient


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """Embedding client using OpenAI's Embeddings API.

    Uses ``text-embedding-3-small`` by default (1536 dimensions).
    Reads ``OPENAI_API_KEY`` from the environment.

    Parameters
    ----------
    model : str
        OpenAI embedding model name.
    api_key : str, optional
        OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        """Lazy-init the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Parameters
        ----------
        query : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector (shape: [1536] for text-embedding-3-small).
        """
        client = self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=query,
        )
        embedding = response.data[0].embedding
        result = np.array(embedding, dtype="float32")
        logging.debug(
            f"OpenAI embed: len={len(query)}, dim={result.shape[0]}, "
            f"tokens={response.usage.total_tokens}"
        )
        return result

    async def embed_batch(self, queries: list[str]) -> np.ndarray:
        """Embed multiple query strings in a single batch.

        Parameters
        ----------
        queries : list of str
            List of texts to embed.

        Returns
        -------
        np.ndarray
            Embedding matrix (shape: [len(queries), 1536]).
        """
        if not queries:
            return np.array([], dtype="float32")

        client = self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=queries,
        )
        embeddings = [item.embedding for item in response.data]
        result = np.array(embeddings, dtype="float32")
        logging.debug(
            f"OpenAI embed_batch: {len(queries)} queries, dim={result.shape[1]}, "
            f"tokens={response.usage.total_tokens}"
        )
        return result

    async def __aenter__(self):
        """No-op context manager (OpenAI client manages its own sessions)."""
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """No-op."""
        pass
