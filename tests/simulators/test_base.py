from simulators.base import Simulator, SimulatorConfig


def test_simulator_init():
    """Test simulator initialization with config."""
    config = SimulatorConfig(name="test_sim")
    simulator = Simulator(config)
    assert simulator.name == "test_sim"
    assert simulator.config == config


def test_simulator_init_default_name():
    """Test simulator initialization with default name."""
    config = SimulatorConfig()
    simulator = Simulator(config)
    assert simulator.name == "Simulator"
    assert simulator.config == config


def test_simulator_config_kwargs():
    """Test simulator config with additional kwargs."""
    config = SimulatorConfig(name="test_sim", host="localhost", port=8000)
    name = getattr(config, "name", None)
    host = getattr(config, "host", None)
    port = getattr(config, "port", None)
    assert name == "test_sim"
    assert host == "localhost"
    assert port == 8000
