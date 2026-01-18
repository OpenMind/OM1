import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from runtime.multi_mode.config import (
    ModeConfig,
    ModeSystemConfig,
    TransitionRule,
    TransitionType,
    _load_mode_components,
    load_mode_config,
)
from runtime.single_mode.config import RuntimeConfig


@pytest.fixture
def mock_sensor():
    """Mock sensor for testing."""
    mock = Mock()
    mock.config = Mock()
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.config = Mock()
    return mock


@pytest.fixture
def mock_simulator():
    """Mock simulator for testing."""
    mock = Mock()
    mock.config = Mock()
    return mock


@pytest.fixture
def mock_action():
    """Mock action for testing."""
    mock = Mock()
    mock.config = Mock()
    return mock


@pytest.fixture
def mock_background():
    """Mock background for testing."""
    mock = Mock()
    mock.config = Mock()
    return mock


@pytest.fixture
def sample_mode_config():
    """Sample mode configuration for testing."""
    return ModeConfig(
        version="v1.0.0",
        name="test_mode",
        display_name="Test Mode",
        description="A test mode for unit testing",
        system_prompt_base="You are a test assistant.",
        hertz=2.0,
        timeout_seconds=300.0,
        remember_locations=True,
        save_interactions=True,
    )


@pytest.fixture
def sample_system_config():
    """Sample system configuration for testing."""
    return ModeSystemConfig(
        name="test_system",
        default_mode="default",
        config_name="test_config",
        allow_manual_switching=True,
        mode_memory_enabled=True,
        api_key="test_api_key",
        robot_ip="192.168.1.100",
        URID="test_urid",
        unitree_ethernet="eth0",
        system_governance="Test governance",
        system_prompt_examples="Test examples",
    )


@pytest.fixture
def sample_transition_rule():
    """Sample transition rule for testing."""
    return TransitionRule(
        from_mode="mode1",
        to_mode="mode2",
        transition_type=TransitionType.INPUT_TRIGGERED,
        trigger_keywords=["switch", "change mode"],
        priority=5,
        cooldown_seconds=10.0,
    )


class TestTransitionRule:
    """Test cases for TransitionRule class."""

    def test_transition_rule_creation(self, sample_transition_rule):
        """Test basic transition rule creation."""
        rule = sample_transition_rule
        assert rule.from_mode == "mode1"
        assert rule.to_mode == "mode2"
        assert rule.transition_type == TransitionType.INPUT_TRIGGERED
        assert rule.trigger_keywords == ["switch", "change mode"]
        assert rule.priority == 5
        assert rule.cooldown_seconds == 10.0

    def test_transition_rule_defaults(self):
        """Test transition rule with default values."""
        rule = TransitionRule(
            from_mode="default_from",
            to_mode="default_to",
            transition_type=TransitionType.MANUAL,
        )
        assert rule.trigger_keywords == []
        assert rule.priority == 1
        assert rule.cooldown_seconds == 0.0
        assert rule.timeout_seconds is None
        assert rule.context_conditions == {}

    def test_transition_type_enum(self):
        """Test TransitionType enum values."""
        assert TransitionType.INPUT_TRIGGERED.value == "input_triggered"
        assert TransitionType.TIME_BASED.value == "time_based"
        assert TransitionType.CONTEXT_AWARE.value == "context_aware"
        assert TransitionType.MANUAL.value == "manual"


class TestModeConfig:
    """Test cases for ModeConfig class."""

    def test_mode_config_creation(self, sample_mode_config):
        """Test basic mode config creation."""
        config = sample_mode_config
        assert config.name == "test_mode"
        assert config.display_name == "Test Mode"
        assert config.description == "A test mode for unit testing"
        assert config.system_prompt_base == "You are a test assistant."
        assert config.hertz == 2.0
        assert config.timeout_seconds == 300.0
        assert config.remember_locations is True
        assert config.save_interactions is True

    def test_mode_config_defaults(self):
        """Test mode config with default values."""
        config = ModeConfig(
            version="v1.0.0",
            name="minimal_mode",
            display_name="Minimal Mode",
            description="Minimal test mode",
            system_prompt_base="Basic prompt",
        )
        assert config.hertz == 1.0
        assert config.timeout_seconds is None
        assert config.remember_locations is False
        assert config.save_interactions is False
        assert len(config.agent_inputs) == 0
        assert config.cortex_llm is None
        assert len(config.simulators) == 0
        assert len(config.agent_actions) == 0
        assert len(config.backgrounds) == 0

    def test_to_runtime_config_success(
        self, sample_mode_config, sample_system_config, mock_llm
    ):
        """Test successful conversion to RuntimeConfig."""
        sample_mode_config.cortex_llm = mock_llm
        sample_system_config.modes = {"test_mode": sample_mode_config}

        runtime_config = sample_mode_config.to_runtime_config(sample_system_config)

        assert isinstance(runtime_config, RuntimeConfig)
        assert runtime_config.hertz == 2.0
        assert runtime_config.name == "test_system_test_mode"
        assert runtime_config.system_prompt_base == "You are a test assistant."
        assert runtime_config.system_governance == "Test governance"
        assert runtime_config.system_prompt_examples == "Test examples"
        assert runtime_config.cortex_llm == mock_llm
        assert runtime_config.robot_ip == "192.168.1.100"
        assert runtime_config.api_key == "test_api_key"
        assert runtime_config.URID == "test_urid"
        assert runtime_config.unitree_ethernet == "eth0"

    def test_to_runtime_config_no_llm(self, sample_mode_config, sample_system_config):
        """Test conversion to RuntimeConfig fails when no LLM is configured."""
        sample_system_config.modes = {"test_mode": sample_mode_config}

        with pytest.raises(ValueError, match="No LLM configured for mode test_mode"):
            sample_mode_config.to_runtime_config(sample_system_config)

    def test_is_loaded_false(self, sample_mode_config):
        """Test is_loaded() returns False when components are not loaded."""
        assert sample_mode_config.is_loaded() is False

    def test_is_loaded_true_with_inputs(self, sample_mode_config, mock_sensor):
        """Test is_loaded() returns True when inputs are loaded."""
        sample_mode_config.agent_inputs = [mock_sensor]
        assert sample_mode_config.is_loaded() is True

    def test_is_loaded_true_with_llm(self, sample_mode_config, mock_llm):
        """Test is_loaded() returns True when LLM is loaded."""
        sample_mode_config.cortex_llm = mock_llm
        assert sample_mode_config.is_loaded() is True

    def test_is_loaded_true_with_actions(self, sample_mode_config, mock_action):
        """Test is_loaded() returns True when actions are loaded."""
        sample_mode_config.agent_actions = [mock_action]
        assert sample_mode_config.is_loaded() is True

    @patch("runtime.multi_mode.config._load_mode_components")
    def test_load_components(
        self, mock_load_components, sample_mode_config, sample_system_config
    ):
        """Test load_components calls _load_mode_components."""
        sample_mode_config.load_components(sample_system_config)
        mock_load_components.assert_called_once_with(
            sample_mode_config, sample_system_config
        )


class TestModeSystemConfig:
    """Test cases for ModeSystemConfig class."""

    def test_system_config_creation(self, sample_system_config):
        """Test basic system config creation."""
        config = sample_system_config
        assert config.name == "test_system"
        assert config.default_mode == "default"
        assert config.config_name == "test_config"
        assert config.allow_manual_switching is True
        assert config.mode_memory_enabled is True
        assert config.api_key == "test_api_key"
        assert config.robot_ip == "192.168.1.100"
        assert config.URID == "test_urid"
        assert config.unitree_ethernet == "eth0"
        assert config.system_governance == "Test governance"
        assert config.system_prompt_examples == "Test examples"

    def test_system_config_defaults(self):
        """Test system config with default values."""
        config = ModeSystemConfig(
            name="minimal_system",
            default_mode="default",
        )
        assert config.config_name == ""
        assert config.allow_manual_switching is True
        assert config.mode_memory_enabled is True
        assert config.api_key is None
        assert config.robot_ip is None
        assert config.URID is None
        assert config.unitree_ethernet is None
        assert config.system_governance == ""
        assert config.system_prompt_examples == ""
        assert config.global_cortex_llm is None
        assert len(config.modes) == 0
        assert len(config.transition_rules) == 0


class TestLoadModeComponents:
    """Test cases for _load_mode_components function."""

    @patch("runtime.multi_mode.config.load_input")
    @patch("runtime.multi_mode.config.load_simulator")
    @patch("runtime.multi_mode.config.load_action")
    @patch("runtime.multi_mode.config.load_background")
    @patch("runtime.multi_mode.config.load_llm")
    def test_load_mode_components_complete(
        self,
        mock_load_llm,
        mock_load_background,
        mock_load_action,
        mock_load_simulator,
        mock_load_input,
        sample_mode_config,
        sample_system_config,
        mock_sensor,
        mock_simulator,
        mock_action,
        mock_background,
        mock_llm,
    ):
        """Test loading all component types."""
        mock_load_input.return_value = mock_sensor
        mock_load_simulator.return_value = mock_simulator
        mock_load_action.return_value = mock_action
        mock_load_background.return_value = mock_background
        mock_load_llm.return_value = mock_llm

        sample_mode_config._raw_inputs = [{"type": "test_input", "config": {}}]
        sample_mode_config._raw_simulators = [{"type": "test_simulator", "config": {}}]
        sample_mode_config._raw_actions = [{"type": "test_action", "config": {}}]
        sample_mode_config._raw_backgrounds = [
            {"type": "test_background", "config": {}}
        ]
        sample_mode_config._raw_llm = {"type": "test_llm", "config": {}}

        _load_mode_components(sample_mode_config, sample_system_config)

        assert len(sample_mode_config.agent_inputs) == 1
        assert sample_mode_config.agent_inputs[0] == mock_sensor
        assert len(sample_mode_config.simulators) == 1
        assert sample_mode_config.simulators[0] == mock_simulator
        assert len(sample_mode_config.agent_actions) == 1
        assert sample_mode_config.agent_actions[0] == mock_action
        assert len(sample_mode_config.backgrounds) == 1
        assert sample_mode_config.backgrounds[0] == mock_background
        assert sample_mode_config.cortex_llm == mock_llm

    @patch("runtime.multi_mode.config.load_llm")
    def test_load_mode_components_with_global_llm(
        self,
        mock_load_llm,
        sample_mode_config,
        sample_system_config,
        mock_llm,
    ):
        """Test loading components with global LLM configuration."""
        mock_load_llm.return_value = mock_llm

        sample_mode_config._raw_llm = None
        sample_system_config.global_cortex_llm = {"type": "global_llm", "config": {}}

        _load_mode_components(sample_mode_config, sample_system_config)

        assert sample_mode_config.cortex_llm == mock_llm
        mock_load_llm.assert_called_once()

    def test_load_mode_components_no_llm_raises_error(
        self,
        sample_mode_config,
        sample_system_config,
    ):
        """Test that missing LLM configuration raises ValueError."""
        sample_mode_config._raw_llm = None
        sample_system_config.global_cortex_llm = None

        with pytest.raises(
            ValueError, match="No LLM configuration found for mode test_mode"
        ):
            _load_mode_components(sample_mode_config, sample_system_config)


class TestLoadModeConfig:
    """Test cases for load_mode_config function."""

    def test_load_mode_config_file_not_found(self):
        """Test load_mode_config with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_mode_config("non_existent_config")

    @patch.dict(
        os.environ,
        {"ROBOT_IP": "env_robot_ip", "OM_API_KEY": "env_api_key", "URID": "env_urid"},
    )
    def test_load_mode_config_env_fallback(self):
        """Test that environment variables are used as fallback."""
        config_data = {
            "version": "v1.0.1",
            "default_mode": "default",
            "api_key": "openmind_free",
            "system_governance": "Env governance",
            "modes": {
                "default": {
                    "hertz": 1.0,
                    "display_name": "Default",
                    "description": "Default mode",
                    "system_prompt_base": "Test prompt",
                    "agent_inputs": [],
                    "agent_actions": [],
                }
            },
            "cortex_llm": {"type": "test_llm"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
            import json5

            json5.dump(config_data, f)
            temp_file = f.name

        try:
            with patch("runtime.multi_mode.config.os.path.join") as mock_join:
                mock_join.return_value = temp_file

                config = load_mode_config("env_test")

                assert config.robot_ip == "env_robot_ip"
                assert config.api_key == "env_api_key"
                assert config.URID == "env_urid"

        finally:
            os.unlink(temp_file)

    @patch("runtime.multi_mode.config.load_unitree")
    def test_load_mode_config_with_unitree_ethernet(self, mock_load_unitree):
        """Test that unitree_ethernet triggers load_unitree call."""
        config_data = {
            "version": "v1.0.1",
            "unitree_ethernet": "eth0",
            "default_mode": "default",
            "api_key": "openmind_free",
            "system_governance": "Env governance",
            "modes": {
                "default": {
                    "hertz": 1.0,
                    "display_name": "Default",
                    "description": "Default mode",
                    "system_prompt_base": "Test prompt",
                    "agent_inputs": [],
                    "agent_actions": [],
                }
            },
            "cortex_llm": {"type": "test_llm"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
            import json5

            json5.dump(config_data, f)
            temp_file = f.name

        try:
            with patch("runtime.multi_mode.config.os.path.join") as mock_join:
                mock_join.return_value = temp_file

                config = load_mode_config("unitree_test")

                assert config.unitree_ethernet == "eth0"
                mock_load_unitree.assert_called_once_with("eth0")

        finally:
            os.unlink(temp_file)

    def test_load_mode_config_invalid_version(self):
        """Test load_mode_config with invalid version format."""
        config_data = {
            "version": "invalid_version",
            "name": "invalid_version_test",
            "default_mode": "default",
            "modes": {
                "default": {
                    "display_name": "Default",
                    "description": "Default mode",
                    "system_prompt_base": "Test prompt",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json5", delete=False) as f:
            import json5

            json5.dump(config_data, f)
            temp_file = f.name

        try:
            with patch("runtime.multi_mode.config.os.path.join") as mock_join:
                mock_join.return_value = temp_file

                with pytest.raises(ValueError, match="Invalid version format"):
                    load_mode_config("invalid_version_test")

        finally:
            os.unlink(temp_file)
