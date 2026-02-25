"""Tests for faiss/faiss_retriever module."""

import pickle
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from src.fuser.knowledge_base.base_retriever import Document
from src.fuser.knowledge_base.faiss.faiss_retriever import FAISSRetriever


class TestFAISSRetriever:
    """Test suite for FAISSRetriever class."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "text": f"Document {i} content",
                "metadata": {"source": f"doc{i}.txt", "chunk_id": i},
            }
            for i in range(10)
        ]

    @pytest.fixture
    def mock_faiss_index(self, tmp_path, sample_documents):
        """Create a mock FAISS index for testing."""
        dimension = 384
        num_docs = len(sample_documents)

        index = faiss.IndexFlatL2(dimension)
        embeddings = np.random.randn(num_docs, dimension).astype("float32")
        index.add(x=embeddings)  # type: ignore

        index_path = tmp_path / "test.faiss"
        faiss.write_index(index, str(index_path))

        metadata_path = tmp_path / "test.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(sample_documents, f)

        return index_path, metadata_path, dimension, num_docs

    def test_initialization_loads_index(self, mock_faiss_index):
        """Test that FAISSRetriever loads index and metadata."""
        index_path, metadata_path, dim, num_docs = mock_faiss_index

        retriever = FAISSRetriever(index_path, metadata_path)

        assert retriever.index is not None
        assert retriever.dimension == dim
        assert len(retriever.documents) == num_docs
        assert retriever.num_documents == num_docs

    def test_initialization_index_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when index doesn't exist."""
        index_path = tmp_path / "nonexistent.faiss"
        metadata_path = tmp_path / "test.pkl"
        metadata_path.touch()

        with pytest.raises(FileNotFoundError) as exc_info:
            FAISSRetriever(index_path, metadata_path)

        assert "Index not found" in str(exc_info.value)

    def test_initialization_metadata_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when metadata doesn't exist."""
        # Create a valid index
        index = faiss.IndexFlatL2(384)
        index_path = tmp_path / "test.faiss"
        faiss.write_index(index, str(index_path))

        metadata_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError) as exc_info:
            FAISSRetriever(index_path, metadata_path)

        assert "Metadata not found" in str(exc_info.value)

    def test_initialization_mismatch_warning(self, tmp_path, sample_documents, caplog):
        """Test warning when document count doesn't match vector count."""
        dimension = 384

        index = faiss.IndexFlatL2(dimension)
        embeddings = np.random.randn(5, dimension).astype("float32")
        index.add(x=embeddings)  # type: ignore
        index_path = tmp_path / "test.faiss"
        faiss.write_index(index, str(index_path))

        metadata_path = tmp_path / "test.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(sample_documents, f)

        FAISSRetriever(index_path, metadata_path)

        assert any("Mismatch" in record.message for record in caplog.records)

    def test_search_returns_top_k_documents(self, mock_faiss_index):
        """Test that search returns top-k most similar documents."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        query_embedding = np.random.randn(dim).astype("float32")
        results = retriever.search(query_embedding, top_k=3)

        assert len(results) == 3
        assert all(isinstance(doc, Document) for doc in results)
        assert all(doc.score is not None for doc in results)

    def test_search_scores_are_positive(self, mock_faiss_index):
        """Test that search scores are positive (similarity scores)."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        query_embedding = np.random.randn(dim).astype("float32")
        results = retriever.search(query_embedding, top_k=5)

        for doc in results:
            assert doc.score is not None
            assert doc.score > 0

    def test_search_wrong_dimension_raises_error(self, mock_faiss_index):
        """Test that search raises ValueError for wrong embedding dimension."""
        index_path, metadata_path, _, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        wrong_dim_embedding = np.random.randn(256).astype("float32")

        with pytest.raises(ValueError) as exc_info:
            retriever.search(wrong_dim_embedding, top_k=3)

        assert "Query dim" in str(exc_info.value)
        assert "index dim" in str(exc_info.value)

    def test_search_top_k_larger_than_index(self, mock_faiss_index):
        """Test search when top_k is larger than number of documents."""
        index_path, metadata_path, dim, num_docs = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        query_embedding = np.random.randn(dim).astype("float32")
        results = retriever.search(query_embedding, top_k=100)

        # Should return at most num_docs results
        assert len(results) <= num_docs

    def test_batch_search_multiple_queries(self, mock_faiss_index):
        """Test batch search with multiple query embeddings."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        num_queries = 3
        query_embeddings = np.random.randn(num_queries, dim).astype("float32")
        all_results = retriever.batch_search(query_embeddings, top_k=2)

        assert len(all_results) == num_queries
        assert all(len(results) == 2 for results in all_results)
        assert all(
            isinstance(doc, Document) for results in all_results for doc in results
        )

    def test_batch_search_wrong_dimension_raises_error(self, mock_faiss_index):
        """Test that batch_search raises ValueError for wrong embedding dimension."""
        index_path, metadata_path, _, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        wrong_dim_embeddings = np.random.randn(3, 256).astype("float32")

        with pytest.raises(ValueError) as exc_info:
            retriever.batch_search(wrong_dim_embeddings, top_k=3)

        assert "Query dim" in str(exc_info.value)

    def test_batch_search_empty_batch(self, mock_faiss_index):
        """Test batch search with empty batch."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        query_embeddings = np.empty((0, dim), dtype="float32")
        all_results = retriever.batch_search(query_embeddings, top_k=5)

        assert len(all_results) == 0

    def test_search_returns_document_copies(self, mock_faiss_index):
        """Test that search returns independent document copies."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        query_embedding = np.random.randn(dim).astype("float32")
        results = retriever.search(query_embedding, top_k=2)

        # Modify first result's metadata
        results[0].metadata["modified"] = True

        # Original document should not be modified
        assert "modified" not in retriever.documents[0].metadata

    def test_documents_are_document_instances(self, mock_faiss_index):
        """Test that loaded documents are Document instances."""
        index_path, metadata_path, _, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        assert all(isinstance(doc, Document) for doc in retriever.documents)

    def test_search_with_negative_indices(self, tmp_path):
        """Test that search handles negative indices from FAISS gracefully."""
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        embeddings = np.random.randn(2, dimension).astype("float32")
        index.add(x=embeddings)  # type: ignore

        index_path = tmp_path / "test.faiss"
        faiss.write_index(index, str(index_path))

        metadata = [{"text": f"Doc {i}", "metadata": {"id": i}} for i in range(2)]
        metadata_path = tmp_path / "test.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        retriever = FAISSRetriever(index_path, metadata_path)

        with patch.object(retriever.index, "search") as mock_search:
            mock_search.return_value = (
                np.array([[0.1, 0.2]], dtype="float32"),
                np.array([[0, -1]], dtype="int64"),
            )

            query_embedding = np.random.randn(dimension).astype("float32")
            results = retriever.search(query_embedding, top_k=2)

            assert len(results) == 1
            assert results[0].text == "Doc 0"

    def test_search_with_out_of_bounds_indices(self, mock_faiss_index):
        """Test that search handles out-of-bounds indices gracefully."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        with patch.object(retriever.index, "search") as mock_search:
            mock_search.return_value = (
                np.array([[0.1, 0.2]], dtype="float32"),
                np.array([[0, 999]], dtype="int64"),
            )

            query_embedding = np.random.randn(dim).astype("float32")
            results = retriever.search(query_embedding, top_k=2)

            assert len(results) == 1

    def test_index_is_loaded_on_init(self, mock_faiss_index):
        """Test that _load is called during initialization."""
        index_path, metadata_path, _, _ = mock_faiss_index

        with patch.object(FAISSRetriever, "_load") as mock_load:

            mock_load.return_value = None
            FAISSRetriever(index_path, metadata_path)
            mock_load.assert_called_once()

    def test_score_calculation(self, mock_faiss_index):
        """Test that score is calculated correctly from distance."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        with patch.object(retriever.index, "search") as mock_search:
            # For IndexFlatIP, dist is already cosine similarity
            mock_search.return_value = (
                np.array([[0.95, 0.85, 0.70]], dtype="float32"),
                np.array([[0, 1, 2]], dtype="int64"),
            )

            query_embedding = np.random.randn(dim).astype("float32")
            results = retriever.search(query_embedding, top_k=3)

            assert results[0].score == pytest.approx(0.95, rel=1e-5)
            assert results[1].score == pytest.approx(0.85, rel=1e-5)
            assert results[2].score == pytest.approx(0.70, rel=1e-5)

    def test_batch_search_returns_independent_results(self, mock_faiss_index):
        """Test that batch search returns independent result sets."""
        index_path, metadata_path, dim, _ = mock_faiss_index
        retriever = FAISSRetriever(index_path, metadata_path)

        query_embeddings = np.random.randn(2, dim).astype("float32")
        all_results = retriever.batch_search(query_embeddings, top_k=3)
        all_results[0][0].metadata["modified"] = True

        assert "modified" not in all_results[1][0].metadata
