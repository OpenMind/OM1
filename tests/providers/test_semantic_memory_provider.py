import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the SemanticMemoryProvider singleton before each test."""
    from providers.semantic_memory_provider import SemanticMemoryProvider

    SemanticMemoryProvider.reset()  # type: ignore
    yield
    SemanticMemoryProvider.reset()  # type: ignore


@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for ChromaDB persistence."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_model():
    """Create a mock SentenceTransformer model."""
    model = MagicMock()
    model.encode.return_value = np.random.randn(384).astype(np.float32)
    return model


class TestSemanticMemoryProviderInit:
    def test_default_state_is_disabled(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        assert provider.enabled is False
        assert provider.top_k == 3
        assert provider.similarity_threshold == 0.3

    def test_configure_without_enable_stays_disabled(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.configure(enabled=False, top_k=5, similarity_threshold=0.5)
        assert provider.enabled is False
        assert provider.top_k == 5
        assert provider.similarity_threshold == 0.5

    def test_singleton_returns_same_instance(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        p1 = SemanticMemoryProvider()
        p2 = SemanticMemoryProvider()
        assert p1 is p2


class TestSemanticMemoryProviderConfigure:
    def test_configure_enabled_calls_initialize(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        with patch.object(provider, "_ensure_initialized") as mock_init:
            provider.configure(enabled=True, top_k=5, similarity_threshold=0.4)
            assert provider.enabled is True
            assert provider.top_k == 5
            assert provider.similarity_threshold == 0.4
            mock_init.assert_called_once()

    def test_configure_disabled_skips_initialize(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.configure(enabled=False)
        assert provider._model is None
        assert provider._chroma_client is None


class TestStoreAndRetrieve:
    def test_store_when_disabled_is_noop(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = False
        provider.store("input", "response", "default", 1)

    def test_retrieve_when_disabled_returns_empty(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = False
        result = provider.retrieve("query", "default")
        assert result == []

    def test_store_when_model_is_none_is_noop(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = None
        provider.store("input", "response", "default", 1)

    def test_retrieve_when_model_is_none_returns_empty(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = None
        result = provider.retrieve("query", "default")
        assert result == []

    def test_store_and_retrieve_with_mock_model(self, mock_model, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)
        provider._persist_dir = temp_persist_dir

        vec1 = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec1

        provider.store("I see a person waving", "I waved back", "test_mode", 1)

        collection = provider._get_collection("test_mode")
        assert collection is not None
        assert collection.count() == 1

        results = provider.retrieve("person waving", "test_mode")
        assert len(results) == 1
        assert "I see a person waving" in results[0]
        assert "I waved back" in results[0]

    def test_retrieve_filters_by_threshold(self, mock_model, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)
        provider._persist_dir = temp_persist_dir
        provider.similarity_threshold = 0.99

        vec1 = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec1
        provider.store("memory text", "action text", "test_mode", 1)

        # Use a very different vector for query
        vec2 = np.zeros(384, dtype=np.float32)
        vec2[0] = 1.0
        mock_model.encode.return_value = vec2

        results = provider.retrieve("unrelated query", "test_mode")
        assert isinstance(results, list)

    def test_store_truncates_long_text(self, mock_model, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import (
            MAX_TEXT_LENGTH,
            SemanticMemoryProvider,
        )

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)

        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec

        long_text = "x" * 5000
        provider.store(long_text, long_text, "test_mode", 1)

        collection = provider._get_collection("test_mode")
        assert collection.count() == 1
        doc = collection.get()["documents"][0]
        # Each side truncated to MAX_TEXT_LENGTH, total < 2 * MAX_TEXT_LENGTH + overhead
        assert len(doc) <= 2 * MAX_TEXT_LENGTH + 30

    def test_retrieve_empty_collection(self, mock_model, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)

        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec

        results = provider.retrieve("query", "empty_mode")
        assert results == []


class TestClearMode:
    def test_clear_mode_removes_collection(self, mock_model, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)

        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec

        provider.store("data", "action", "clearable", 1)
        assert provider._get_collection("clearable").count() == 1

        provider.clear_mode("clearable")
        assert "om1_clearable" not in provider._collections

    def test_clear_mode_when_no_client_is_noop(self):
        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider._chroma_client = None
        provider.clear_mode("some_mode")


class TestModeIsolation:
    def test_different_modes_have_separate_collections(
        self, mock_model, temp_persist_dir
    ):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)

        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec

        provider.store("mode_a data", "mode_a action", "mode_a", 1)
        provider.store("mode_b data", "mode_b action", "mode_b", 2)

        assert provider._get_collection("mode_a").count() == 1
        assert provider._get_collection("mode_b").count() == 1

        results_a = provider.retrieve("data", "mode_a")
        results_b = provider.retrieve("data", "mode_b")

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert "mode_a" in results_a[0]
        assert "mode_b" in results_b[0]


class TestMultipleStores:
    def test_multiple_stores_accumulate(self, mock_model, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)

        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec

        for i in range(5):
            provider.store(f"input_{i}", f"action_{i}", "test_mode", i)

        collection = provider._get_collection("test_mode")
        assert collection.count() == 5

    def test_top_k_limits_results(self, mock_model, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = mock_model
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)
        provider.top_k = 2
        provider.similarity_threshold = 0.0

        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec

        for i in range(5):
            provider.store(f"input_{i}", f"action_{i}", "test_mode", i)

        results = provider.retrieve("query", "test_mode")
        assert len(results) <= 2


class TestErrorHandling:
    def test_store_handles_encoding_error(self, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = MagicMock()
        provider._model.encode.side_effect = RuntimeError("encoding failed")
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)

        provider.store("input", "response", "test_mode", 1)

    def test_retrieve_handles_query_error(self, temp_persist_dir):
        import chromadb

        from providers.semantic_memory_provider import SemanticMemoryProvider

        provider = SemanticMemoryProvider()
        provider.enabled = True
        provider._model = MagicMock()
        provider._model.encode.side_effect = RuntimeError("encoding failed")
        provider._chroma_client = chromadb.PersistentClient(path=temp_persist_dir)

        result = provider.retrieve("query", "test_mode")
        assert result == []
