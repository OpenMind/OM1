"""Unit tests for EpisodicMemoryProvider."""

import json
import os
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from providers.episodic_memory_provider import EpisodicMemoryProvider


@pytest.fixture
def reset_singleton():
    from typing import Any
    provider_factory: Any = EpisodicMemoryProvider
    provider_factory.reset()
    provider = provider_factory()
    yield provider
    provider_factory.reset()


class TestInit:

    def test_default_state(self, reset_singleton):
        """Provider should initialize with empty episodes and default mode."""
        p = reset_singleton
        assert p._episodes == []
        assert p._pending_flush == []
        assert p._openai_client is None
        assert p._current_mode == "default"
        assert p._loaded_modes == set()


class TestGetStoragePath:

    def test_returns_hidden_json_path(self, reset_singleton, tmp_path):
        """Storage path should be a hidden .json file under storage dir."""
        reset_singleton._storage_dir = str(tmp_path)
        path = reset_singleton._get_storage_path("slam")
        assert path.endswith(".slam.episodes.json")
        assert os.path.isdir(tmp_path)


class TestGetOpenAIClient:

    def test_creates_client_when_none(self, reset_singleton):
        """Should create new OpenAI client if none exists."""
        provider = reset_singleton
        provider._openai_client = None
        with patch("providers.episodic_memory_provider.AsyncOpenAI") as mock_cls:
            mock_instance = Mock()
            mock_cls.return_value = mock_instance
            client = provider._get_openai_client()
        assert client is mock_instance
        assert provider._openai_client is mock_instance
        mock_cls.assert_called_once()

    def test_reuses_existing_client(self, reset_singleton):
        """Should reuse existing OpenAI client without creating a new one."""
        provider = reset_singleton
        mock_client = Mock()
        provider._openai_client = mock_client
        with patch("providers.episodic_memory_provider.AsyncOpenAI") as mock_cls:
            client = provider._get_openai_client()
        assert client is mock_client
        mock_cls.assert_not_called()


class TestEmbed:

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, reset_singleton):
        """_embed should return None for empty string without calling API."""
        provider = reset_singleton
        provider._openai_client = None
        with patch.object(provider, "_get_openai_client") as mock_get:
            result = await provider._embed("")
        assert result is None
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_whitespace_only(self, reset_singleton):
        """_embed should return None for whitespace-only input."""
        provider = reset_singleton
        with patch.object(provider, "_get_openai_client"):
            result = await provider._embed("   ")
        assert result is None

    @pytest.mark.asyncio
    async def test_embed_api_failure(self, reset_singleton):
        """_embed should return None if OpenAI API raises an exception."""
        provider = reset_singleton
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(side_effect=Exception("API error"))
        provider._openai_client = mock_client
        with patch.object(provider, "_get_openai_client", return_value=mock_client):
            result = await provider._embed("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self, reset_singleton):
        """_embed should return embedding vector on success."""
        provider = reset_singleton
        fake_embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data[0].embedding = fake_embedding
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        with patch.object(provider, "_get_openai_client", return_value=mock_client):
            result = await provider._embed("some text")
        assert result == fake_embedding


class TestCosineSimilarity:

    def test_identical_vectors(self, reset_singleton):
        """Cosine similarity of identical vectors should be 1.0."""
        v = [1.0, 0.0, 0.0]
        assert reset_singleton._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, reset_singleton):
        """Cosine similarity of orthogonal vectors should be 0.0."""
        assert reset_singleton._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self, reset_singleton):
        """Cosine similarity with zero vector should return 0.0."""
        assert reset_singleton._cosine_similarity([0, 0], [1, 2]) == pytest.approx(0.0)

    def test_opposite_vectors(self, reset_singleton):
        """Cosine similarity of opposite vectors should be -1.0."""
        assert reset_singleton._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


class TestLoadMode:

    def test_load_nonexistent_file(self, reset_singleton, tmp_path):
        """load_mode should succeed even if file does not exist."""
        reset_singleton._storage_dir = str(tmp_path)
        reset_singleton.load_mode("missing")
        assert "missing" in reset_singleton._loaded_modes

    def test_load_existing_file(self, reset_singleton, tmp_path):
        """load_mode should load episodes from existing JSON file."""
        reset_singleton._storage_dir = str(tmp_path)
        ep = {"episode_id": "abc", "mode": "slam", "voice_input": "hi", "actions": []}
        path = tmp_path / ".slam.episodes.json"
        path.write_text(json.dumps([ep]))
        reset_singleton.load_mode("slam")
        assert any(e["episode_id"] == "abc" for e in reset_singleton._episodes)
        assert "slam" in reset_singleton._loaded_modes

    def test_load_is_idempotent(self, reset_singleton, tmp_path):
        """load_mode called twice should not duplicate episodes."""
        reset_singleton._storage_dir = str(tmp_path)
        ep = {"episode_id": "xyz", "mode": "nav", "voice_input": "go", "actions": []}
        path = tmp_path / ".nav.episodes.json"
        path.write_text(json.dumps([ep]))
        reset_singleton.load_mode("nav")
        reset_singleton.load_mode("nav")
        count = sum(1 for e in reset_singleton._episodes if e["episode_id"] == "xyz")
        assert count == 1

    def test_load_corrupt_file_does_not_raise(self, reset_singleton, tmp_path):
        """load_mode should not raise on corrupt JSON file."""
        reset_singleton._storage_dir = str(tmp_path)
        path = tmp_path / ".bad.episodes.json"
        path.write_text("not json")
        reset_singleton.load_mode("bad")
        assert "bad" in reset_singleton._loaded_modes


class TestSaveToDisk:

    def test_save_empty_pending_is_noop(self, reset_singleton, tmp_path):
        """save_to_disk should not write anything if pending list is empty."""
        reset_singleton._storage_dir = str(tmp_path)
        reset_singleton._pending_flush = []
        reset_singleton.save_to_disk()
        assert list(tmp_path.iterdir()) == []

    def test_save_writes_file(self, reset_singleton, tmp_path):
        """save_to_disk should write pending episodes to correct file."""
        reset_singleton._storage_dir = str(tmp_path)
        ep = {
            "episode_id": str(uuid.uuid4()),
            "mode": "slam",
            "voice_input": "test",
            "actions": [],
            "timestamp": time.time(),
        }
        reset_singleton._pending_flush = [ep]
        reset_singleton.save_to_disk()
        path = tmp_path / ".slam.episodes.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert any(e["episode_id"] == ep["episode_id"] for e in data)
        assert reset_singleton._pending_flush == []

    def test_save_caps_at_1000(self, reset_singleton, tmp_path):
        """save_to_disk should cap stored episodes at 1000 total."""
        reset_singleton._storage_dir = str(tmp_path)
        existing = [
            {"episode_id": str(i), "mode": "m", "voice_input": "x", "actions": []}
            for i in range(999)
        ]
        path = tmp_path / ".m.episodes.json"
        path.write_text(json.dumps(existing))
        new_ep = {"episode_id": "new1", "mode": "m", "voice_input": "y", "actions": []}
        new_ep2 = {"episode_id": "new2", "mode": "m", "voice_input": "z", "actions": []}
        reset_singleton._pending_flush = [new_ep, new_ep2]
        reset_singleton.save_to_disk()
        data = json.loads(path.read_text())
        assert len(data) == 1000

    def test_save_corrupt_existing_file_is_overwritten(self, reset_singleton, tmp_path):
        """save_to_disk should overwrite corrupt existing file."""
        reset_singleton._storage_dir = str(tmp_path)
        path = tmp_path / ".z.episodes.json"
        path.write_text("not valid json")
        ep = {"episode_id": "ok", "mode": "z", "voice_input": "hi", "actions": []}
        reset_singleton._pending_flush = [ep]
        reset_singleton.save_to_disk()
        data = json.loads(path.read_text())
        assert any(e["episode_id"] == "ok" for e in data)

    def test_save_handles_write_error(self, reset_singleton, tmp_path):
        """save_to_disk should not raise if file write fails."""
        reset_singleton._storage_dir = str(tmp_path)
        reset_singleton._pending_flush = [
            {"episode_id": "err", "mode": "x", "voice_input": "v", "actions": []}
        ]
        with patch("builtins.open", side_effect=OSError("no space")):
            reset_singleton.save_to_disk()


class TestWriteEpisode:

    @pytest.mark.asyncio
    async def test_write_episode_success(self, reset_singleton):
        fake_embedding = [0.5] * 10
        with patch.object(reset_singleton, "_embed", AsyncMock(return_value=fake_embedding)):
            await reset_singleton.write_episode(
                mode="slam",
                voice_input="go to kitchen",
                actions=[{"type": "speak", "value": "ok"}],
            )
        assert len(reset_singleton._episodes) == 1
        assert len(reset_singleton._pending_flush) == 1
        ep = reset_singleton._episodes[0]
        assert ep["mode"] == "slam"
        assert ep["voice_input"] == "go to kitchen"
        assert ep["embedding"] == fake_embedding

    @pytest.mark.asyncio
    async def test_write_episode_empty_input_skipped(self, reset_singleton):
        with patch.object(reset_singleton, "_embed", AsyncMock()) as mock_embed:
            await reset_singleton.write_episode(mode="slam", voice_input="", actions=[])
        mock_embed.assert_not_called()
        assert reset_singleton._episodes == []

    @pytest.mark.asyncio
    async def test_write_episode_no_embedding_skipped(self, reset_singleton):
        with patch.object(reset_singleton, "_embed", AsyncMock(return_value=None)):
            await reset_singleton.write_episode(mode="slam", voice_input="hello", actions=[])
        assert reset_singleton._episodes == []

    @pytest.mark.asyncio
    async def test_write_episode_with_battery(self, reset_singleton):
        with patch.object(reset_singleton, "_embed", AsyncMock(return_value=[0.1])):
            await reset_singleton.write_episode(
                mode="nav", voice_input="charge", actions=[], battery="80%"
            )
        assert reset_singleton._episodes[0]["battery"] == "80%"


class TestRecall:

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, reset_singleton):
        result = await reset_singleton.recall("")
        assert result == []

    @pytest.mark.asyncio
    async def test_recall_no_episodes(self, reset_singleton):
        with patch.object(reset_singleton, "_embed", AsyncMock(return_value=[0.1, 0.2])):
            result = await reset_singleton.recall("anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_recall_returns_top_k(self, reset_singleton):
        embeddings = {
            "query": [1.0, 0.0],
            "ep1": [1.0, 0.0],
            "ep2": [0.0, 1.0],
            "ep3": [0.9, 0.1],
        }
        reset_singleton._episodes = [
            {"episode_id": "1", "mode": "m", "voice_input": "a", "actions": [], "embedding": embeddings["ep1"]},
            {"episode_id": "2", "mode": "m", "voice_input": "b", "actions": [], "embedding": embeddings["ep2"]},
            {"episode_id": "3", "mode": "m", "voice_input": "c", "actions": [], "embedding": embeddings["ep3"]},
        ]
        with patch.object(reset_singleton, "_embed", AsyncMock(return_value=embeddings["query"])):
            result = await reset_singleton.recall("query", top_k=2)
        assert len(result) == 2
        for ep in result:
            assert "embedding" not in ep

    @pytest.mark.asyncio
    async def test_recall_skips_episodes_without_embedding(self, reset_singleton):
        reset_singleton._episodes = [
            {"episode_id": "no-emb", "mode": "m", "voice_input": "x", "actions": []},
        ]
        with patch.object(reset_singleton, "_embed", AsyncMock(return_value=[1.0, 0.0])):
            result = await reset_singleton.recall("query")
        assert result == []

    @pytest.mark.asyncio
    async def test_recall_no_query_embedding(self, reset_singleton):
        reset_singleton._episodes = [
            {"episode_id": "1", "mode": "m", "voice_input": "x", "actions": [], "embedding": [1.0]},
        ]
        with patch.object(reset_singleton, "_embed", AsyncMock(return_value=None)):
            result = await reset_singleton.recall("query")
        assert result == []
