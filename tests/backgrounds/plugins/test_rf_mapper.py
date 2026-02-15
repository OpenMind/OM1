from unittest.mock import MagicMock, patch

from backgrounds.plugins.rf_mapper import RFmapper, RFmapperConfig


class TestRFmapperConfig:
    """Test cases for RFmapperConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RFmapperConfig()
        assert config.name == "RFmapper"
        assert config.api_key is None
        assert config.URID is None
        assert config.unitree_ethernet is None

    def test_custom_name(self):
        """Test custom name configuration."""
        config = RFmapperConfig(name="CustomMapper")
        assert config.name == "CustomMapper"

    def test_custom_api_key(self):
        """Test custom api_key configuration."""
        config = RFmapperConfig(api_key="test-api-key")
        assert config.api_key == "test-api-key"

    def test_custom_urid(self):
        """Test custom URID configuration."""
        config = RFmapperConfig(URID="robot-001")
        assert config.URID == "robot-001"

    def test_custom_unitree_ethernet(self):
        """Test custom unitree_ethernet configuration."""
        config = RFmapperConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"

    def test_all_custom_values(self):
        """Test configuration with all custom values."""
        config = RFmapperConfig(
            name="Mapper1",
            api_key="key-123",
            URID="robot-002",
            unitree_ethernet="enp2s0",
        )
        assert config.name == "Mapper1"
        assert config.api_key == "key-123"
        assert config.URID == "robot-002"
        assert config.unitree_ethernet == "enp2s0"


class TestRFmapper:
    """Test cases for RFmapper background plugin."""

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_initialization(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test background initialization creates all providers."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps

        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk

        mock_odom = MagicMock()
        mock_odom_class.return_value = mock_odom

        mock_fds = MagicMock()
        mock_fds_class.return_value = mock_fds

        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        config = RFmapperConfig(api_key="test-key", URID="robot-001")
        background = RFmapper(config)

        assert background.name == "RFmapper"
        assert background.api_key == "test-key"
        assert background.URID == "robot-001"
        assert background.gps == mock_gps
        assert background.rtk == mock_rtk
        assert background.odom == mock_odom
        assert background.fds == mock_fds

        mock_gps_class.assert_called_once()
        mock_rtk_class.assert_called_once()
        mock_odom_class.assert_called_once()
        mock_fds_class.assert_called_once_with(
            api_key="test-key", write_to_local_file=True
        )

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_initialization_logging(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
        caplog,
    ):
        """Test that initialization logs config and provider info."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk
        mock_odom_class.return_value = MagicMock()
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig()
        with caplog.at_level("INFO"):
            RFmapper(config)

        assert "Mapper config:" in caplog.text
        assert "Mapper Gps Provider:" in caplog.text
        assert "Mapper Rtk Provider:" in caplog.text
        assert "Mapper Odom Provider:" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_thread_started_on_init(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that thread.start() is called during __init__ via self.start()."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk
        mock_odom_class.return_value = MagicMock()
        mock_fds_class.return_value = MagicMock()

        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        config = RFmapperConfig()
        RFmapper(config)

        mock_thread.start.assert_called_once()

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_initial_state_values(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that initial state values are set to defaults."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk
        mock_odom_class.return_value = MagicMock()
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig()
        background = RFmapper(config)

        assert background.scan_results == []
        assert background.scan_idx == 0
        assert background.scan_last_sent == 0
        assert background.payload_idx == 0
        assert background.odom_x == 0.0
        assert background.odom_y == 0.0
        assert background.gps_lat == 0.0
        assert background.gps_lon == 0.0
        assert background.rtk_lat == 0.0
        assert background.rtk_lon == 0.0

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_gps_on_reflects_provider_running(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that gps_on reflects the GPS provider's running state."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk
        mock_odom_class.return_value = MagicMock()
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig()
        background = RFmapper(config)

        assert background.gps_on is True
        assert background.rtk_on is False

    @patch("backgrounds.plugins.rf_mapper.time.sleep")
    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_stop(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
        mock_sleep,
    ):
        """Test stop method sets running to False and joins thread."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk
        mock_odom_class.return_value = MagicMock()
        mock_fds_class.return_value = MagicMock()

        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        config = RFmapperConfig()
        background = RFmapper(config)

        background.stop()

        assert background.running is False
        mock_sleep.assert_called_with(1)
        mock_thread.join.assert_called_once()

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_parses_gps_data(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that run() parses GPS data from provider."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = {
            "gps_unix_ts": 1234567890.0,
            "gps_lat": 40.7128,
            "gps_lon": -74.0060,
            "gps_alt": 10.5,
            "yaw_mag_0_360": 180.0,
            "gps_qua": 4,
            "ble_scan": None,
        }
        mock_gps_class.return_value = mock_gps

        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk.data = None
        mock_rtk_class.return_value = mock_rtk

        mock_odom = MagicMock()
        mock_odom.position = None
        mock_odom_class.return_value = mock_odom

        mock_fds = MagicMock()
        mock_fds_class.return_value = mock_fds

        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig(URID="robot-001")
        background = RFmapper(config)
        # Enable running so the while loop enters, then stop after one iteration
        background.running = True

        def stop_after_one_iteration(duration):
            background.running = False

        background.sleep = stop_after_one_iteration  # type: ignore[assignment]

        background.run()

        assert background.gps_lat == 40.7128
        assert background.gps_lon == -74.0060
        assert background.gps_alt == 10.5

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_parses_odom_data(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that run() parses odometry data from provider."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = None
        mock_gps_class.return_value = mock_gps

        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk.data = None
        mock_rtk_class.return_value = mock_rtk

        mock_odom = MagicMock()
        mock_odom.position = {
            "odom_x": 1.5,
            "odom_y": 2.5,
            "odom_rockchip_ts": 100.0,
            "odom_subscriber_ts": 101.0,
            "odom_yaw_0_360": 90.0,
            "odom_yaw_m180_p180": 90.0,
        }
        mock_odom_class.return_value = mock_odom

        mock_fds = MagicMock()
        mock_fds_class.return_value = mock_fds

        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig()
        background = RFmapper(config)
        background.running = True

        def stop_after_one_iteration(duration):
            background.running = False

        background.sleep = stop_after_one_iteration  # type: ignore[assignment]

        background.run()

        assert background.odom_x == 1.5
        assert background.odom_y == 2.5

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_submits_fabric_data(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that run() submits data to FabricDataSubmitter."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = None
        mock_gps_class.return_value = mock_gps

        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk.data = None
        mock_rtk_class.return_value = mock_rtk

        mock_odom = MagicMock()
        mock_odom.position = None
        mock_odom_class.return_value = mock_odom

        mock_fds = MagicMock()
        mock_fds_class.return_value = mock_fds

        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig(URID="robot-001")
        background = RFmapper(config)
        background.running = True

        def stop_after_one_iteration(duration):
            background.running = False

        background.sleep = stop_after_one_iteration  # type: ignore[assignment]

        background.run()

        mock_fds.share_data.assert_called_once()
        call_args = mock_fds.share_data.call_args[0][0]
        assert call_args.machine_id == "robot-001"
        assert call_args.payload_idx == 0

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_with_no_urid_uses_unknown(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that run() uses 'Unknown' when URID is None."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = None
        mock_gps_class.return_value = mock_gps

        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk.data = None
        mock_rtk_class.return_value = mock_rtk

        mock_odom = MagicMock()
        mock_odom.position = None
        mock_odom_class.return_value = mock_odom

        mock_fds = MagicMock()
        mock_fds_class.return_value = mock_fds

        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig()
        background = RFmapper(config)
        background.running = True

        def stop_after_one_iteration(duration):
            background.running = False

        background.sleep = stop_after_one_iteration  # type: ignore[assignment]

        background.run()

        call_args = mock_fds.share_data.call_args[0][0]
        assert call_args.machine_id == "Unknown"

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_config_stored(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test that config is stored correctly."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk
        mock_odom_class.return_value = MagicMock()
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig(name="TestMapper", api_key="key-123")
        background = RFmapper(config)

        assert background.config is config
        assert background.name == "TestMapper"
        assert background.api_key == "key-123"
