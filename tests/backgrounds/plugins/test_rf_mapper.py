from unittest.mock import MagicMock, patch

from src.backgrounds.plugins.rf_mapper import RFmapper, RFmapperConfig


class TestRFmapperConfig:
    def test_config_defaults(self):
        config = RFmapperConfig()
        assert config.name == "RFmapper"
        assert config.api_key is None
        assert config.URID is None
        assert config.unitree_ethernet is None

    def test_config_custom_values(self):
        config = RFmapperConfig(
            name="CustomMapper",
            api_key="my_api_key",
            URID="12345",
            unitree_ethernet="eth0",
        )
        assert config.name == "CustomMapper"
        assert config.api_key == "my_api_key"
        assert config.URID == "12345"
        assert config.unitree_ethernet == "eth0"


class TestRFmapper:
    @patch("src.backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("src.backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("src.backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("src.backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("src.backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    @patch("src.backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("src.backgrounds.plugins.rf_mapper.logging")
    def test_initialization_sets_attributes_and_starts_thread(
        self,
        mock_logging,
        mock_thread_class,
        mock_new_loop,
        mock_gps_provider_class,
        mock_rtk_provider_class,
        mock_odom_provider_class,
        mock_fabric_submitter_class,
    ):
        mock_loop_instance = MagicMock()
        mock_new_loop.return_value = mock_loop_instance

        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        mock_gps_instance = MagicMock()
        mock_gps_instance.running = True
        mock_gps_provider_class.return_value = mock_gps_instance

        mock_rtk_instance = MagicMock()
        mock_rtk_instance.running = True
        mock_rtk_provider_class.return_value = mock_rtk_instance

        mock_odom_instance = MagicMock()
        mock_odom_provider_class.return_value = mock_odom_instance

        mock_fabric_submitter_instance = MagicMock()
        mock_fabric_submitter_class.return_value = mock_fabric_submitter_instance

        config = RFmapperConfig(
            name="TestMapper",
            api_key="test_key",
            URID="test_urid",
            unitree_ethernet="eth1",
        )

        rf_mapper = RFmapper(config)

        assert rf_mapper.name == "TestMapper"
        assert rf_mapper.api_key == "test_key"
        assert rf_mapper.URID == "test_urid"
        assert rf_mapper.unitree_ethernet == "eth1"

        assert rf_mapper.loop is mock_loop_instance
        assert rf_mapper.thread is mock_thread_instance
        assert rf_mapper.running is False
        assert rf_mapper.scan_results == []
        assert rf_mapper.scan_idx == 0
        assert rf_mapper.scan_last_sent == 0
        assert rf_mapper.payload_idx == 0

        mock_gps_provider_class.assert_called_once()
        mock_rtk_provider_class.assert_called_once()
        mock_odom_provider_class.assert_called_once()

        assert rf_mapper.gps_on is True
        assert rf_mapper.rtk_on is True

        mock_fabric_submitter_class.assert_called_once_with(
            api_key="test_key", write_to_local_file=True
        )

        mock_thread_instance.start.assert_called_once()

        mock_logging.info.assert_any_call(f"Mapper config: {config}")

        assert rf_mapper.seen_devices == {}
        assert rf_mapper.seen_names == []
