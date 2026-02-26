from pathlib import Path

import numpy as np
import pytest

from src.fuser.knowledge_base.base_retriever import BaseRetriever, Document


class ConcreteRetriever(BaseRetriever):
    """Concrete implementation for testing."""

    def __init__(self, index_path, metadata_path, test_docs=None, test_dim=384):
        super().__init__(index_path, metadata_path)
        self.documents = test_docs or []
        self.dimension = test_dim
        self.load_called = False

    def _load(self):
        """Mock load implementation."""
        self.load_called = True

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[Document]:
        """Return top-k documents based on mock similarity."""
        results = []
        for i, doc in enumerate(self.documents[:top_k]):
            score = 1.0 - (i * 0.1)  # Decreasing scores
            results.append(
                Document(text=doc.text, metadata=doc.metadata.copy(), score=score)
            )
        return results

    def batch_search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> list[list[Document]]:
        """Return top-k documents for each query."""
        return [self.search(emb, top_k) for emb in query_embeddings]


class TestDocument:
    """Test suite for Document dataclass."""

    def test_document_creation(self):
        """Test creating a Document with required fields."""
        doc = Document(text="test content", metadata={"source": "test.txt"})

        assert doc.text == "test content"
        assert doc.metadata == {"source": "test.txt"}
        assert doc.score is None

    def test_document_with_score(self):
        """Test creating a Document with a score."""
        doc = Document(text="test content", metadata={"source": "test.txt"}, score=0.95)

        assert doc.text == "test content"
        assert doc.metadata == {"source": "test.txt"}
        assert doc.score == 0.95

    def test_document_metadata_independent(self):
        """Test that document metadata is independent."""
        metadata = {"source": "test.txt", "chunk_id": 0}
        doc1 = Document(text="content1", metadata=metadata)
        doc2 = Document(text="content2", metadata=metadata)

        doc1.metadata["new_key"] = "value"

        assert "new_key" in doc1.metadata
        assert "new_key" in doc2.metadata  # They share the same dict reference

    def test_document_equality(self):
        """Test document equality based on dataclass fields."""
        doc1 = Document(text="test", metadata={"id": 1}, score=0.9)
        doc2 = Document(text="test", metadata={"id": 1}, score=0.9)

        assert doc1 == doc2

    def test_document_inequality(self):
        """Test document inequality."""
        doc1 = Document(text="test1", metadata={"id": 1})
        doc2 = Document(text="test2", metadata={"id": 1})

        assert doc1 != doc2


class TestBaseRetriever:
    """Test suite for BaseRetriever abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseRetriever cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRetriever(  # type: ignore
                index_path="test.faiss", metadata_path="test.pkl"
            )

    def test_concrete_implementation_initialization(self, tmp_path):
        """Test that concrete implementation can be initialized."""
        index_path = tmp_path / "test.faiss"
        metadata_path = tmp_path / "test.pkl"

        retriever = ConcreteRetriever(index_path, metadata_path)

        assert retriever.index_path == index_path
        assert retriever.metadata_path == metadata_path
        assert isinstance(retriever.documents, list)
        assert retriever.dimension == 384

    def test_search_returns_documents(self, tmp_path):
        """Test that search returns a list of Documents."""
        test_docs = [Document(text=f"doc {i}", metadata={"id": i}) for i in range(5)]
        retriever = ConcreteRetriever(
            tmp_path / "test.faiss", tmp_path / "test.pkl", test_docs=test_docs
        )

        query_embedding = np.random.randn(384).astype("float32")
        results = retriever.search(query_embedding, top_k=3)

        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
        assert all(doc.score is not None for doc in results)

    def test_search_respects_top_k(self, tmp_path):
        """Test that search returns at most top_k results."""
        test_docs = [Document(text=f"doc {i}", metadata={"id": i}) for i in range(10)]
        retriever = ConcreteRetriever(
            tmp_path / "test.faiss", tmp_path / "test.pkl", test_docs=test_docs
        )

        query_embedding = np.random.randn(384).astype("float32")
        results = retriever.search(query_embedding, top_k=5)

        assert len(results) == 5

    def test_search_scores_decreasing(self, tmp_path):
        """Test that search results have decreasing scores."""
        test_docs = [Document(text=f"doc {i}", metadata={"id": i}) for i in range(5)]
        retriever = ConcreteRetriever(
            tmp_path / "test.faiss", tmp_path / "test.pkl", test_docs=test_docs
        )

        query_embedding = np.random.randn(384).astype("float32")
        results = retriever.search(query_embedding, top_k=5)

        scores = [doc.score for doc in results if doc.score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_batch_search(self, tmp_path):
        """Test batch search with multiple query embeddings."""
        test_docs = [Document(text=f"doc {i}", metadata={"id": i}) for i in range(5)]
        retriever = ConcreteRetriever(
            tmp_path / "test.faiss", tmp_path / "test.pkl", test_docs=test_docs
        )

        query_embeddings = np.random.randn(3, 384).astype("float32")
        all_results = retriever.batch_search(query_embeddings, top_k=2)

        assert len(all_results) == 3
        assert all(len(results) == 2 for results in all_results)
        assert all(
            isinstance(doc, Document) for results in all_results for doc in results
        )

    def test_num_documents_property(self, tmp_path):
        """Test that num_documents returns the correct count."""
        test_docs = [Document(text=f"doc {i}", metadata={"id": i}) for i in range(7)]
        retriever = ConcreteRetriever(
            tmp_path / "test.faiss", tmp_path / "test.pkl", test_docs=test_docs
        )

        assert retriever.num_documents == 7

    def test_empty_retriever(self, tmp_path):
        """Test retriever with no documents."""
        retriever = ConcreteRetriever(
            tmp_path / "test.faiss", tmp_path / "test.pkl", test_docs=[]
        )

        assert retriever.num_documents == 0

        query_embedding = np.random.randn(384).astype("float32")
        results = retriever.search(query_embedding, top_k=5)

        assert len(results) == 0

    def test_path_conversion(self, tmp_path):
        """Test that paths are converted to Path objects."""
        index_path_str = str(tmp_path / "test.faiss")
        metadata_path_str = str(tmp_path / "test.pkl")

        retriever = ConcreteRetriever(index_path_str, metadata_path_str)

        assert isinstance(retriever.index_path, Path)
        assert isinstance(retriever.metadata_path, Path)
