from backgrounds.base import Background, BackgroundConfig


def test_background_init():
    """Test background initialization with config."""
    config = BackgroundConfig()
    background = Background(config)
    assert background.name == "Background"
    assert background.config == config


def test_background_init_default_name():
    """Test background initialization with default name."""
    config = BackgroundConfig()
    background = Background(config)
    assert background.name == "Background"
    assert background.config == config
