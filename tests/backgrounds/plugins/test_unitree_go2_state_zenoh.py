from unittest.mock import MagicMock, patch

from backgrounds.plugins.unitree_go2_state_zenoh import (
    UnitreeGo2StateZenoh,
    UnitreeGo2StateZenohConfig,
)


class TestUnitreeGo2StateZenoh:
    def test_initialization_default(self):
        with patch("backgrounds.plugins.unitree_go2_state_zenoh.UnitreeGo2StateZenohProvider") as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2StateZenohConfig()
            background = UnitreeGo2StateZenoh(config)

            assert background.config is config
            assert background.unitree_go2_state_provider is mock_provider
            mock_provider_class.assert_called_once_with(None, False)

    def test_initialization_custom_topic(self):
        with patch("backgrounds.plugins.unitree_go2_state_zenoh.UnitreeGo2StateZenohProvider") as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            config = UnitreeGo2StateZenohConfig(api_key="test_key", use_sim=True)
            background = UnitreeGo2StateZenoh(config)

            mock_provider_class.assert_called_once_with("test_key", True)
            assert background.unitree_go2_state_provider is mock_provider

    def test_initialization_logging(self, caplog):
        with patch("backgrounds.plugins.unitree_go2_state_zenoh.UnitreeGo2StateZenohProvider") as mock_provider_class:
            mock_provider_class.return_value = MagicMock()

            config = UnitreeGo2StateZenohConfig()
            with caplog.at_level("INFO"):
                UnitreeGo2StateZenoh(config)

            assert "Unitree Go2 State Zenoh Provider initialized" in caplog.text
