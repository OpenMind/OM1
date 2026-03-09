"""Unit tests for EpisodicMemoryWriter background."""

from unittest.mock import Mock, patch

import pytest

from backgrounds.plugins.episodic_memory_writer import (
    EpisodicMemoryWriter,
    EpisodicMemoryWriterConfig,
)


@pytest.fixture
def writer():
    """Create EpisodicMemoryWriter instance with mocked provider."""
    with patch('backgrounds.plugins.episodic_memory_writer.EpisodicMemoryProvider') as mock_cls:
        mock_provider = Mock()
        mock_cls.return_value = mock_provider
        config = EpisodicMemoryWriterConfig()
        w = EpisodicMemoryWriter(config=config)
        w.provider = mock_provider
        yield w, mock_provider


class TestEpisodicMemoryWriterConfig:

    def test_default_config(self):
        """Default config should instantiate without error."""
        config = EpisodicMemoryWriterConfig()
        assert config is not None


class TestEpisodicMemoryWriter:

    def test_init(self, writer):
        """Writer should initialize with provider and correct flush interval."""
        w, _ = writer
        assert w.provider is not None
        assert w.FLUSH_INTERVAL_SECONDS == 30

    def test_run_success_calls_sleep(self, writer):
        """run() should flush to disk and sleep once per cycle."""
        w, mock_provider = writer
        sleep_called = []
        w.sleep = lambda d: sleep_called.append(d)
        w.run()
        mock_provider.save_to_disk.assert_called_once()
        assert len(sleep_called) == 1
        assert sleep_called[0] == w.FLUSH_INTERVAL_SECONDS

    def test_run_handles_flush_exception(self, writer):
        """run() should not crash if save_to_disk raises."""
        w, mock_provider = writer
        call_count = 0

        def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                w.should_stop = Mock(return_value=True)
                return False
            return True

        mock_provider.save_to_disk.side_effect = Exception("disk error")
        w.sleep = mock_sleep
        w.run()

    def test_stop_flushes(self, writer):
        """stop() should trigger final flush to disk."""
        w, mock_provider = writer
        w.stop()
        mock_provider.save_to_disk.assert_called_once()

    def test_stop_handles_exception(self, writer):
        """stop() should not crash if save_to_disk raises."""
        w, mock_provider = writer
        mock_provider.save_to_disk.side_effect = Exception("disk full")
        w.stop()
