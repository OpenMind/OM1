import numpy as np
import pytest

from src.fuser.knowledge_base.base_embedding import BaseEmbeddingClient


class ConcreteEmbeddingClient(BaseEmbeddingClient):
    """Concrete implementation for testing."""

    def __init__(self, embedding_dim: int = 384, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.embed_calls = []
        self.embed_batch_calls = []

    async def embed(self, query: str) -> np.ndarray:
        """Return a mock embedding vector."""
        self.embed_calls.append(query)
        return np.random.randn(self.embedding_dim).astype("float32")

    async def embed_batch(self, queries: list[str]) -> np.ndarray:
        """Return mock embedding vectors for batch."""
        self.embed_batch_calls.append(queries)
        return np.random.randn(len(queries), self.embedding_dim).astype("float32")


class TestBaseEmbeddingClient:
    """Test suite for BaseEmbeddingClient abstract class."""

    @pytest.mark.asyncio
    async def test_cannot_instantiate_abstract_class(self):
        """Test that BaseEmbeddingClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbeddingClient()  # type: ignore

    @pytest.mark.asyncio
    async def test_concrete_implementation_embed(self):
        """Test that concrete implementation can embed a query."""
        client = ConcreteEmbeddingClient(embedding_dim=384)
        query = "test query"

        embedding = await client.embed(query)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
        assert query in client.embed_calls

    @pytest.mark.asyncio
    async def test_concrete_implementation_embed_batch(self):
        """Test that concrete implementation can embed multiple queries."""
        client = ConcreteEmbeddingClient(embedding_dim=384)
        queries = ["query 1", "query 2", "query 3"]

        embeddings = await client.embed_batch(queries)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
        assert queries in client.embed_batch_calls

    @pytest.mark.asyncio
    async def test_context_manager_enter_exit(self):
        """Test that context manager methods work properly."""
        client = ConcreteEmbeddingClient()

        async with client as ctx_client:
            assert ctx_client is client
            embedding = await ctx_client.embed("test")
            assert embedding.shape == (384,)

    @pytest.mark.asyncio
    async def test_different_embedding_dimensions(self):
        """Test that clients can use different embedding dimensions."""
        dim = 768
        client = ConcreteEmbeddingClient(embedding_dim=dim)

        embedding = await client.embed("test")

        assert embedding.shape == (dim,)

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test that empty batch returns empty array."""
        client = ConcreteEmbeddingClient()

        embeddings = await client.embed_batch([])

        assert embeddings.shape == (0, 384)

    @pytest.mark.asyncio
    async def test_single_item_batch(self):
        """Test that single item batch works correctly."""
        client = ConcreteEmbeddingClient()

        embeddings = await client.embed_batch(["single query"])

        assert embeddings.shape == (1, 384)
