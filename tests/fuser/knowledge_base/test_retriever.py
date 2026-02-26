from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.fuser.knowledge_base.base_retriever import Document
from src.fuser.knowledge_base.retriever import KnowledgeBase


class TestKnowledgeBase:
    """Test suite for KnowledgeBase class."""

    @pytest.fixture
    def mock_kb_structure(self, tmp_path):
        """Create a mock knowledge base directory structure."""
        kb_root = tmp_path / "knowledge_base"
        kb_root.mkdir()

        demo_dir = kb_root / "demo"
        demo_dir.mkdir()

        # Create mock index and metadata files
        (demo_dir / "demo.faiss").touch()
        (demo_dir / "demo.pkl").touch()

        return kb_root

    def test_initialization_default_params(self, mock_kb_structure):
        """Test KnowledgeBase initialization with default parameters."""
        with (
            patch(
                "src.fuser.knowledge_base.retriever.EmbeddingClient"
            ) as mock_embedding,
            patch(
                "src.fuser.knowledge_base.retriever.FAISSRetriever"
            ) as mock_retriever,
        ):
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.num_documents = 10
            mock_retriever_instance.dimension = 384
            mock_retriever.return_value = mock_retriever_instance

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            assert kb.kb_dir == mock_kb_structure / "demo"
            mock_embedding.assert_called_once_with(host="localhost", port=8100)
            mock_retriever.assert_called_once()

    def test_initialization_custom_params(self, mock_kb_structure):
        """Test KnowledgeBase initialization with custom parameters."""
        with (
            patch(
                "src.fuser.knowledge_base.retriever.EmbeddingClient"
            ) as mock_embedding,
            patch(
                "src.fuser.knowledge_base.retriever.FAISSRetriever"
            ) as mock_retriever,
        ):
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.num_documents = 10
            mock_retriever_instance.dimension = 384
            mock_retriever.return_value = mock_retriever_instance

            kb = KnowledgeBase(
                knowledge_base_name="demo",
                knowledge_base_root=mock_kb_structure,
                embedding_host="192.168.1.1",
                embedding_port=9000,
            )

            assert kb.kb_dir == mock_kb_structure / "demo"
            mock_embedding.assert_called_once_with(host="192.168.1.1", port=9000)

    def test_initialization_kb_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when knowledge base doesn't exist."""
        kb_root = tmp_path / "knowledge_base"
        kb_root.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            KnowledgeBase(
                knowledge_base_name="nonexistent", knowledge_base_root=kb_root
            )

        assert "Knowledge base not found" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_initialization_invalid_retriever_type(self, mock_kb_structure):
        """Test that ValueError is raised for invalid retriever type."""
        with pytest.raises(ValueError) as exc_info:
            KnowledgeBase(
                knowledge_base_name="demo",
                knowledge_base_root=mock_kb_structure,
                retriever_type="invalid",
            )

        assert "Unknown retriever type" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_initialization_default_kb_root(self):
        """Test that default knowledge_base_root is set correctly."""
        with (
            patch(
                "src.fuser.knowledge_base.retriever.EmbeddingClient"
            ) as mock_embedding,
            patch(
                "src.fuser.knowledge_base.retriever.FAISSRetriever"
            ) as mock_retriever,
            patch("src.fuser.knowledge_base.retriever.Path") as mock_path,
        ):
            mock_parent = MagicMock()
            mock_parent.parent.parent.parent.parent = MagicMock()
            mock_kb_root = MagicMock()
            mock_parent.parent.parent.parent.parent.__truediv__.return_value = (
                mock_kb_root
            )

            mock_kb_dir = MagicMock()
            mock_kb_dir.exists.return_value = True
            mock_kb_root.__truediv__.return_value = mock_kb_dir

            mock_path.return_value = mock_parent
            mock_path.__file__ = "/fake/path/retriever.py"

            mock_retriever_instance = MagicMock()
            mock_retriever_instance.num_documents = 10
            mock_retriever_instance.dimension = 384
            mock_retriever.return_value = mock_retriever_instance

            KnowledgeBase(knowledge_base_name="demo")
            mock_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_single(self, mock_kb_structure):
        """Test querying knowledge base with a single query."""
        with (
            patch(
                "src.fuser.knowledge_base.retriever.EmbeddingClient"
            ) as mock_embedding_cls,
            patch(
                "src.fuser.knowledge_base.retriever.FAISSRetriever"
            ) as mock_retriever_cls,
        ):
            # Setup mocks
            mock_embedding = MagicMock()
            mock_embedding.embed = AsyncMock(
                return_value=np.random.randn(384).astype("float32")
            )
            mock_embedding.__aenter__ = AsyncMock(return_value=mock_embedding)
            mock_embedding.__aexit__ = AsyncMock()
            mock_embedding_cls.return_value = mock_embedding

            expected_docs = [
                Document(text=f"doc {i}", metadata={"id": i}, score=0.9 - i * 0.1)
                for i in range(3)
            ]
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = expected_docs
            mock_retriever.num_documents = 10
            mock_retriever.dimension = 384
            mock_retriever_cls.return_value = mock_retriever

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            # Execute query
            results = await kb.query("test query", top_k=3)

            # Verify
            assert len(results) == 3
            assert results == expected_docs
            mock_embedding.embed.assert_called_once_with("test query")
            mock_retriever.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_batch(self, mock_kb_structure):
        """Test querying knowledge base with multiple queries."""
        with (
            patch(
                "src.fuser.knowledge_base.retriever.EmbeddingClient"
            ) as mock_embedding_cls,
            patch(
                "src.fuser.knowledge_base.retriever.FAISSRetriever"
            ) as mock_retriever_cls,
        ):
            # Setup mocks
            mock_embedding = MagicMock()
            mock_embedding.embed_batch = AsyncMock(
                return_value=np.random.randn(2, 384).astype("float32")
            )
            mock_embedding.__aenter__ = AsyncMock(return_value=mock_embedding)
            mock_embedding.__aexit__ = AsyncMock()
            mock_embedding_cls.return_value = mock_embedding

            expected_results = [
                [
                    Document(text=f"doc {i}", metadata={"id": i}, score=0.9)
                    for i in range(2)
                ],
                [
                    Document(text=f"doc {i}", metadata={"id": i}, score=0.8)
                    for i in range(2)
                ],
            ]
            mock_retriever = MagicMock()
            mock_retriever.batch_search.return_value = expected_results
            mock_retriever.num_documents = 10
            mock_retriever.dimension = 384
            mock_retriever_cls.return_value = mock_retriever

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            # Execute batch query
            queries = ["query 1", "query 2"]
            results = await kb.query_batch(queries, top_k=2)

            # Verify
            assert len(results) == 2
            assert results == expected_results
            mock_embedding.embed_batch.assert_called_once_with(queries)
            mock_retriever.batch_search.assert_called_once()

    def test_format_context_empty_results(self, mock_kb_structure):
        """Test formatting context with empty results."""
        with (
            patch("src.fuser.knowledge_base.retriever.EmbeddingClient"),
            patch("src.fuser.knowledge_base.retriever.FAISSRetriever") as mock_ret,
        ):
            mock_retriever = MagicMock()
            mock_retriever.num_documents = 10
            mock_retriever.dimension = 384
            mock_ret.return_value = mock_retriever

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            context = kb.format_context([])

            assert context == ""

    def test_format_context_with_results(self, mock_kb_structure):
        """Test formatting context with results."""
        with (
            patch("src.fuser.knowledge_base.retriever.EmbeddingClient"),
            patch("src.fuser.knowledge_base.retriever.FAISSRetriever") as mock_ret,
        ):
            mock_retriever = MagicMock()
            mock_retriever.num_documents = 10
            mock_retriever.dimension = 384
            mock_ret.return_value = mock_retriever

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            docs = [
                Document(
                    text="Content 1",
                    metadata={"source": "doc1.txt", "chunk_id": 0},
                    score=0.95,
                ),
                Document(
                    text="Content 2",
                    metadata={"source": "doc2.txt", "chunk_id": 1},
                    score=0.85,
                ),
            ]

            context = kb.format_context(docs)

            assert "Content 1" in context
            assert "Content 2" in context
            assert "doc1.txt" in context
            assert "doc2.txt" in context
            assert "0.950" in context
            assert "0.850" in context

    def test_format_context_max_chars_limit(self, mock_kb_structure):
        """Test that format_context respects max_chars limit."""
        with (
            patch("src.fuser.knowledge_base.retriever.EmbeddingClient"),
            patch("src.fuser.knowledge_base.retriever.FAISSRetriever") as mock_ret,
        ):
            mock_retriever = MagicMock()
            mock_retriever.num_documents = 10
            mock_retriever.dimension = 384
            mock_ret.return_value = mock_retriever

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            docs = [
                Document(
                    text="X" * 100,
                    metadata={"source": f"doc{i}.txt", "chunk_id": i},
                    score=0.9,
                )
                for i in range(10)
            ]

            context = kb.format_context(docs, max_chars=300)

            assert len(context) <= 300

    def test_format_context_missing_metadata(self, mock_kb_structure):
        """Test formatting context with missing metadata fields."""
        with (
            patch("src.fuser.knowledge_base.retriever.EmbeddingClient"),
            patch("src.fuser.knowledge_base.retriever.FAISSRetriever") as mock_ret,
        ):
            mock_retriever = MagicMock()
            mock_retriever.num_documents = 10
            mock_retriever.dimension = 384
            mock_ret.return_value = mock_retriever

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            docs = [Document(text="Content with no metadata", metadata={}, score=0.95)]

            context = kb.format_context(docs)

            assert "Content with no metadata" in context
            assert "unknown" in context  # Default source
            assert "?" in context  # Default chunk_id

    @pytest.mark.asyncio
    async def test_query_with_different_top_k(self, mock_kb_structure):
        """Test that top_k parameter is properly passed through."""
        with (
            patch(
                "src.fuser.knowledge_base.retriever.EmbeddingClient"
            ) as mock_embedding_cls,
            patch(
                "src.fuser.knowledge_base.retriever.FAISSRetriever"
            ) as mock_retriever_cls,
        ):
            mock_embedding = MagicMock()
            mock_embedding.embed = AsyncMock(
                return_value=np.random.randn(384).astype("float32")
            )
            mock_embedding.__aenter__ = AsyncMock(return_value=mock_embedding)
            mock_embedding.__aexit__ = AsyncMock()
            mock_embedding_cls.return_value = mock_embedding

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            mock_retriever.num_documents = 10
            mock_retriever.dimension = 384
            mock_retriever_cls.return_value = mock_retriever

            kb = KnowledgeBase(
                knowledge_base_name="demo", knowledge_base_root=mock_kb_structure
            )

            await kb.query("test", top_k=10)

            call_args = mock_retriever.search.call_args
            assert call_args[1]["top_k"] == 10
