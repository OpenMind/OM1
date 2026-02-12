from unittest.mock import MagicMock, patch

from backgrounds.plugins.unitree_go2_frontier_exploration import (
    UnitreeGo2FrontierExploration,
    UnitreeGo2FrontierExplorationConfig,
)


class TestUnitreeGo2FrontierExplorationConfig:
    """Test cases for UnitreeGo2FrontierExplorationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeGo2FrontierExplorationConfig()
        assert config.topic == "explore/status"
        assert config.context_aware_text == '{"exploration_done": true}'

    def test_custom_topic(self):
        """Test custom topic configuration."""
        config = UnitreeGo2FrontierExplorationConfig(topic="custom/topic")
        assert config.topic == "custom/topic"

    def test_custom_context_aware_text(self):
        """Test custom context_aware_text configuration."""
        config = UnitreeGo2FrontierExplorationConfig(
            context_aware_text='{"key": "value"}'
        )
        assert config.context_aware_text == '{"key": "value"}'


class TestUnitreeGo2FrontierExploration:
    """Test cases for UnitreeGo2FrontierExploration background plugin."""

    @patch(
        "backgrounds.plugins.unitree_go2_frontier_exploration.UnitreeGo2FrontierExplorationProvider"
    )
    def test_initialization(self, mock_provider_class):
        """Test background initialization with valid JSON config."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2FrontierExplorationConfig()
        background = UnitreeGo2FrontierExploration(config)

        assert background.config is config
        assert background.unitree_go2_frontier_exploration_provider == mock_provider
        mock_provider_class.assert_called_once_with(
            topic="explore/status",
            context_aware_text={"exploration_done": True},
        )
        mock_provider.start.assert_called_once()

    @patch(
        "backgrounds.plugins.unitree_go2_frontier_exploration.UnitreeGo2FrontierExplorationProvider"
    )
    def test_initialization_with_custom_config(self, mock_provider_class):
        """Test initialization with custom topic and context."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2FrontierExplorationConfig(
            topic="nav/explore",
            context_aware_text='{"done": false, "progress": 50}',
        )
        UnitreeGo2FrontierExploration(config)

        mock_provider_class.assert_called_once_with(
            topic="nav/explore",
            context_aware_text={"done": False, "progress": 50},
        )

    @patch(
        "backgrounds.plugins.unitree_go2_frontier_exploration.UnitreeGo2FrontierExplorationProvider"
    )
    def test_initialization_with_invalid_json_fallback(self, mock_provider_class):
        """Test that invalid JSON falls back to default dict."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2FrontierExplorationConfig(
            context_aware_text="not valid json{{{",
        )
        UnitreeGo2FrontierExploration(config)

        mock_provider_class.assert_called_once_with(
            topic="explore/status",
            context_aware_text={"exploration_done": True},
        )

    @patch(
        "backgrounds.plugins.unitree_go2_frontier_exploration.UnitreeGo2FrontierExplorationProvider"
    )
    def test_initialization_with_invalid_json_logs_error(
        self, mock_provider_class, caplog
    ):
        """Test that invalid JSON logs an error message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2FrontierExplorationConfig(
            context_aware_text="not valid json",
        )
        with caplog.at_level("ERROR"):
            UnitreeGo2FrontierExploration(config)

        assert "Error decoding context_aware_text JSON" in caplog.text

    @patch(
        "backgrounds.plugins.unitree_go2_frontier_exploration.UnitreeGo2FrontierExplorationProvider"
    )
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2FrontierExplorationConfig()
        with caplog.at_level("INFO"):
            UnitreeGo2FrontierExploration(config)

        assert (
            "Unitree Go2 Frontier Exploration Provider initialized in background"
            in caplog.text
        )

    @patch(
        "backgrounds.plugins.unitree_go2_frontier_exploration.UnitreeGo2FrontierExplorationProvider"
    )
    def test_provider_start_called(self, mock_provider_class):
        """Test that provider.start() is called during initialization."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2FrontierExplorationConfig()
        UnitreeGo2FrontierExploration(config)

        mock_provider.start.assert_called_once()

    @patch(
        "backgrounds.plugins.unitree_go2_frontier_exploration.UnitreeGo2FrontierExplorationProvider"
    )
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2FrontierExplorationConfig(topic="test/topic")
        background = UnitreeGo2FrontierExploration(config)

        assert background.config is config
        assert background.config.topic == "test/topic"
