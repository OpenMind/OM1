"""Unit tests for EpisodicMemoryInput plugin."""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from inputs.plugins.episodic_memory_input import (
    EpisodicMemoryInput,
    EpisodicMemoryInputConfig,
)


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.episodic_memory_input.IOProvider") as mock_cls:
        provider = Mock()
        provider.tick_counter = 1
        mock_cls.return_value = provider
        yield provider


@pytest.fixture
def mock_episodic_provider():
    with patch(
        "inputs.plugins.episodic_memory_input.EpisodicMemoryProvider"
    ) as mock_cls:
        provider = Mock()
        provider.recall = AsyncMock(return_value=[])
        mock_cls.return_value = provider
        yield provider


@pytest.fixture
def input_plugin(mock_io_provider, mock_episodic_provider):
    config = EpisodicMemoryInputConfig()
    plugin = EpisodicMemoryInput(config=config)
    yield plugin


class TestEpisodicMemoryInputConfig:

    def test_default_config(self):
        config = EpisodicMemoryInputConfig()
        assert config is not None


class TestEpisodicMemoryInput:

    def test_init(self, input_plugin):
        assert input_plugin.descriptor_for_LLM == "EpisodicMemory"
        assert input_plugin.messages == []

    @pytest.mark.asyncio
    async def test_poll_returns_voice_input(self, input_plugin, mock_io_provider):
        voice_mock = Mock()
        voice_mock.input = "go to kitchen"
        voice_mock.tick = 1
        mock_io_provider.tick_counter = 1
        mock_io_provider.get_input.return_value = voice_mock
        result = await input_plugin._poll()
        assert result == "go to kitchen"

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_no_voice(
        self, input_plugin, mock_io_provider
    ):
        mock_io_provider.get_input.return_value = None
        result = await input_plugin._poll()
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_returns_none_stale_tick(self, input_plugin, mock_io_provider):
        voice_mock = Mock()
        voice_mock.input = "hello"
        voice_mock.tick = 0
        mock_io_provider.tick_counter = 5
        mock_io_provider.get_input.return_value = voice_mock
        result = await input_plugin._poll()
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_empty_string_returns_none(self, input_plugin):
        """_raw_to_text with empty/None input should return None without calling recall."""
        result = await input_plugin._raw_to_text("")
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_no_episodes_returns_none(self, input_plugin):
        """_raw_to_text should return None when recall returns empty list."""
        input_plugin.provider.recall = AsyncMock(return_value=[])
        result = await input_plugin._raw_to_text("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_appends_message(
        self, input_plugin, mock_episodic_provider
    ):
        mock_episodic_provider.recall = AsyncMock(
            return_value=[
                {
                    "timestamp": time.time() - 60,
                    "mode": "slam",
                    "voice_input": "go to kitchen",
                    "actions": [{"type": "speak", "value": "heading to kitchen"}],
                }
            ]
        )
        await input_plugin.raw_to_text("go to kitchen")
        assert len(input_plugin.messages) == 1

    @pytest.mark.asyncio
    async def test_raw_to_text_no_episodes(self, input_plugin, mock_episodic_provider):
        mock_episodic_provider.recall = AsyncMock(return_value=[])
        await input_plugin.raw_to_text("unknown query")
        assert len(input_plugin.messages) == 0

    @pytest.mark.asyncio
    async def test_raw_to_text_empty_input(self, input_plugin, mock_episodic_provider):
        await input_plugin.raw_to_text(None)
        mock_episodic_provider.recall.assert_not_called()

    @pytest.mark.asyncio
    async def test_raw_to_text_empty_formatted(
        self, input_plugin, mock_episodic_provider
    ):
        mock_episodic_provider.recall = AsyncMock(
            return_value=[
                {"timestamp": None, "mode": "slam", "voice_input": "", "actions": []}
            ]
        )
        input_plugin._format_episodes = Mock(return_value=None)
        await input_plugin.raw_to_text("test")
        assert len(input_plugin.messages) == 0

    def test_formatted_latest_buffer_empty(self, input_plugin):
        assert input_plugin.formatted_latest_buffer() is None

    def test_formatted_latest_buffer_returns_and_clears(self, input_plugin):
        from inputs.base import Message

        input_plugin.messages = [
            Message(timestamp=time.time(), message="past interaction")
        ]
        result = input_plugin.formatted_latest_buffer()
        assert result is not None
        assert "EpisodicMemory" in result
        assert input_plugin.messages == []

    def test_format_episodes_empty_list_returns_none(self, input_plugin):
        result = input_plugin._format_episodes([])
        assert result is None

    def test_format_episodes_time_labels(self, input_plugin):
        now = time.time()
        episodes = [
            {
                "timestamp": now - 30,
                "mode": "slam",
                "voice_input": "test1",
                "actions": [],
            },
            {
                "timestamp": now - 120,
                "mode": "slam",
                "voice_input": "test2",
                "actions": [],
            },
            {
                "timestamp": now - 7200,
                "mode": "slam",
                "voice_input": "test3",
                "actions": [],
            },
            {
                "timestamp": now - 90000,
                "mode": "slam",
                "voice_input": "test4",
                "actions": [],
            },
        ]
        result = input_plugin._format_episodes(episodes)
        assert "just now" in result
        assert "m ago" in result
        assert "h ago" in result
        assert "d ago" in result
