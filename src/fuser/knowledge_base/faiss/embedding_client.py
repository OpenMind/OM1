import base64
import logging
from typing import Optional

import aiohttp
import numpy as np

from ..base_embedding import BaseEmbeddingClient


class EmbeddingClient(BaseEmbeddingClient):
    """
    Client for interacting with an embedding server.

    This implementation communicates with a remote embedding service
    via HTTP requests. Can be used with any embedding service that
    exposes a compatible API.
    """

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        """
        Initialize the embedding client.

        Parameters
        ----------
        base_url : str
            Base URL of the embedding server (default: "http://localhost:8100").
        timeout : float
            Request timeout in seconds (default: 30.0).
        """
        super().__init__()
        self.base_url = base_url
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

    async def _make_request(self, endpoint: str, payload: dict) -> dict:
        """
        Make an HTTP POST request to the embedding server.

        Parameters
        ----------
        endpoint : str
            API endpoint (e.g., "embed", "embed_batch").
        payload : dict
            JSON payload to send.

        Returns
        -------
        dict
            JSON response from server.
        """
        if self._session:
            async with self._session.post(
                f"{self.base_url}/{endpoint}", json=payload
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
        else:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/{endpoint}", json=payload
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()

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
        data = await self._make_request("embed", payload)

        emb_bytes = base64.b64decode(data["embedding_b64"])
        embedding = np.frombuffer(emb_bytes, dtype="float32")

        logging.debug(
            f"Embedded query (len={len(query)}) in {data['latency_ms']:.1f}ms"
        )
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
        data = await self._make_request("embed_batch", payload)

        embeddings = []
        for emb_b64 in data["embeddings_b64"]:
            emb_bytes = base64.b64decode(emb_b64)
            embedding = np.frombuffer(emb_bytes, dtype="float32")
            embeddings.append(embedding)

        embeddings_array = np.array(embeddings)
        logging.debug(f"Embedded {len(queries)} queries in {data['latency_ms']:.1f}ms")
        return embeddings_array
