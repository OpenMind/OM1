"""Tests for faiss/embedding_client module."""

import base64
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import numpy as np
import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import AioHTTPTestCase
from aiohttp.web import Application, json_response

from src.fuser.knowledge_base.faiss.embedding_client import EmbeddingClient


class MockEmbeddingServer(AioHTTPTestCase):
    """Mock embedding server for testing."""

    async def get_application(self):
        """Create the test application."""
        app = Application()
        app.router.add_post("/embed", self.handle_embed)
        app.router.add_post("/embed_batch", self.handle_embed_batch)
        return app

    async def handle_embed(self, request):
        """Handle single embedding request."""
        data = await request.json()
        query = data.get("query", "")

        embedding = np.random.randn(384).astype("float32")
        embedding_b64 = base64.b64encode(embedding.tobytes()).decode("utf-8")

        return json_response(
            {
                "embedding_b64": embedding_b64,
                "latency_ms": 10.5,
                "query_len": len(query),
            }
        )

    async def handle_embed_batch(self, request):
        """Handle batch embedding request."""
        data = await request.json()
        queries = data.get("queries", [])

        # Generate mock embeddings
        embeddings_b64 = []
        for _ in queries:
            embedding = np.random.randn(384).astype("float32")
            embedding_b64 = base64.b64encode(embedding.tobytes()).decode("utf-8")
            embeddings_b64.append(embedding_b64)

        return json_response(
            {
                "embeddings_b64": embeddings_b64,
                "latency_ms": 15.2,
                "num_queries": len(queries),
            }
        )

    async def test_embed_single_query(self):
        """Test embedding a single query."""
        host = self.server.host
        port = self.server.port
        client = EmbeddingClient(base_url=f"http://{host}:{port}")  # type: ignore

        async with client:
            embedding = await client.embed("test query")

            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32

    async def test_embed_batch_queries(self):
        """Test embedding multiple queries in batch."""
        host = self.server.host
        port = self.server.port
        client = EmbeddingClient(base_url=f"http://{host}:{port}")  # type: ignore

        queries = ["query 1", "query 2", "query 3"]

        async with client:
            embeddings = await client.embed_batch(queries)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (3, 384)
            assert embeddings.dtype == np.float32

    async def test_embed_without_context_manager(self):
        """Test embedding without using context manager."""
        host = self.server.host
        port = self.server.port
        client = EmbeddingClient(base_url=f"http://{host}:{port}")  # type: ignore

        embedding = await client.embed("test query")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    async def test_embed_batch_empty_list(self):
        """Test embedding empty batch."""
        host = self.server.host
        port = self.server.port
        client = EmbeddingClient(base_url=f"http://{host}:{port}")  # type: ignore

        async with client:
            embeddings = await client.embed_batch([])

            assert isinstance(embeddings, np.ndarray)
            # Empty batch results in (0,) shape due to np.array([]) behavior
            assert embeddings.shape[0] == 0


class TestEmbeddingClientUnit:
    """Unit tests for EmbeddingClient without running a server."""

    def test_initialization_default_params(self):
        """Test EmbeddingClient initialization with default parameters."""
        client = EmbeddingClient()

        assert client.base_url == "http://localhost:8100"
        assert client.timeout.total == 30.0
        assert client._session is None

    def test_initialization_custom_params(self):
        """Test EmbeddingClient initialization with custom parameters."""
        client = EmbeddingClient(base_url="http://192.168.1.1:9000", timeout=60.0)

        assert client.base_url == "http://192.168.1.1:9000"
        assert client.timeout.total == 60.0

    @pytest.mark.asyncio
    async def test_context_manager_creates_session(self):
        """Test that context manager creates and closes session."""
        client = EmbeddingClient()

        assert client._session is None

        async with client:
            assert client._session is not None
            assert isinstance(client._session, ClientSession)

    @pytest.mark.asyncio
    async def test_embed_with_server_error(self):
        """Test that embed raises exception on server error."""
        client = EmbeddingClient(
            base_url="http://localhost:9999"
        )  # Non-existent server

        with pytest.raises(aiohttp.ClientError):
            await client.embed("test query")

    @pytest.mark.asyncio
    async def test_embed_batch_with_server_error(self):
        """Test that embed_batch raises exception on server error."""
        client = EmbeddingClient(
            base_url="http://localhost:9999"
        )  # Non-existent server

        with pytest.raises(aiohttp.ClientError):
            await client.embed_batch(["query 1", "query 2"])

    @pytest.mark.asyncio
    async def test_multiple_embeds_with_same_session(self):
        """Test multiple embed calls with the same session."""
        client = EmbeddingClient()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_embedding = np.random.randn(384).astype("float32")
        mock_embedding_b64 = base64.b64encode(mock_embedding.tobytes()).decode("utf-8")
        mock_response.json = AsyncMock(
            return_value={"embedding_b64": mock_embedding_b64, "latency_ms": 10.0}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_session.post.return_value.__aexit__ = AsyncMock()
        mock_session.close = AsyncMock()  # Add async close method

        async with client:
            client._session = mock_session

            # Make multiple calls
            await client.embed("query 1")
            await client.embed("query 2")

            # Session.post should be called twice
            assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_decodes_base64_correctly(self):
        """Test that embed correctly decodes base64 encoded embeddings."""
        client = EmbeddingClient()

        # Create a known embedding
        expected_embedding = np.array([1.0, 2.0, 3.0], dtype="float32")
        embedding_b64 = base64.b64encode(expected_embedding.tobytes()).decode("utf-8")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={"embedding_b64": embedding_b64, "latency_ms": 10.0}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_session.post.return_value.__aexit__ = AsyncMock()
        mock_session.close = AsyncMock()  # Add async close method

        async with client:
            client._session = mock_session
            result = await client.embed("test")

            np.testing.assert_array_almost_equal(result, expected_embedding)

    @pytest.mark.asyncio
    async def test_embed_batch_decodes_multiple_embeddings(self):
        """Test that embed_batch correctly decodes multiple embeddings."""
        client = EmbeddingClient()

        # Create known embeddings
        expected_embeddings = [
            np.array([1.0, 2.0, 3.0], dtype="float32"),
            np.array([4.0, 5.0, 6.0], dtype="float32"),
        ]
        embeddings_b64 = [
            base64.b64encode(emb.tobytes()).decode("utf-8")
            for emb in expected_embeddings
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={"embeddings_b64": embeddings_b64, "latency_ms": 15.0}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_session.post.return_value.__aexit__ = AsyncMock()
        mock_session.close = AsyncMock()  # Add async close method

        async with client:
            client._session = mock_session
            result = await client.embed_batch(["query1", "query2"])

            assert result.shape == (2, 3)
            np.testing.assert_array_almost_equal(result[0], expected_embeddings[0])
            np.testing.assert_array_almost_equal(result[1], expected_embeddings[1])

    @pytest.mark.asyncio
    async def test_timeout_configuration(self):
        """Test that timeout is properly configured."""
        client = EmbeddingClient(timeout=5.0)

        assert client.timeout.total == 5.0

        # Verify timeout is used when creating session
        async with client:
            # The session should have the timeout set
            # Note: We can't directly check the session's timeout as it's encapsulated
            pass
