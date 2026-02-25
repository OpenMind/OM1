import base64
import logging
from typing import Optional

import aiohttp
import numpy as np

from ..base_embedding import BaseEmbeddingClient

logger = logging.getLogger(__name__)


class EmbeddingClient(BaseEmbeddingClient):
    """
    Client for interacting with an embedding server.

    This implementation communicates with a remote embedding service
    via HTTP requests. Can be used with any embedding service that
    exposes a compatible API.
    """

    def __init__(
        self, host: str = "localhost", port: int = 8100, timeout: float = 30.0
    ):
        """
        Initialize the embedding client.

        Parameters
        ----------
        host : str
            Embedding server host (default: "localhost").
        port : int
            Embedding server port (default: 8100).
        timeout : float
            Request timeout in seconds (default: 30.0).
        """
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Create session when entering context manager."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Close session when exiting context manager."""
        if self._session:
            await self._session.close()
            self._session = None

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
            Embedding vector (shape: [384] for e5-small-v2).

        Raises
        ------
        aiohttp.ClientError
            If the request fails.
        """
        payload = {"query": query}

        if not self._session:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.base_url}/embed", json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        else:
            async with self._session.post(
                f"{self.base_url}/embed", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        emb_bytes = base64.b64decode(data["embedding_b64"])
        embedding = np.frombuffer(emb_bytes, dtype="float32")

        logger.debug(f"Embedded query (len={len(query)}) in {data['latency_ms']:.1f}ms")
        return embedding

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
            Embedding matrix (shape: [len(queries), 384]).

        Raises
        ------
        aiohttp.ClientError
            If the request fails.
        """
        payload = {"queries": queries}

        if not self._session:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/embed_batch", json=payload
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        else:
            async with self._session.post(
                f"{self.base_url}/embed_batch", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        embeddings = []
        for emb_b64 in data["embeddings_b64"]:
            emb_bytes = base64.b64decode(emb_b64)
            embedding = np.frombuffer(emb_bytes, dtype="float32")
            embeddings.append(embedding)

        embeddings_array = np.array(embeddings)
        logger.debug(f"Embedded {len(queries)} queries in {data['latency_ms']:.1f}ms")
        return embeddings_array
