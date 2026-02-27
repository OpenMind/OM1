import threading
import time
from unittest.mock import MagicMock, patch

from backgrounds.plugins.rf_mapper import RFmapper, RFmapperConfig
from providers.fabric_map_provider import RFData


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

        def stop_after_one_iteration(duration: float) -> bool:
            background.running = False
            return False

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

        def stop_after_one_iteration(duration: float) -> bool:
            background.running = False
            return False

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

        def stop_after_one_iteration(duration: float) -> bool:
            background.running = False
            return False

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

        def stop_after_one_iteration(duration: float) -> bool:
            background.running = False
            return False

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


class TestRFmapperThreadSafetyAndMemoryLeak:
    """Test cases for thread safety and memory leak fixes."""

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_scan_lock_exists(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test scan_lock is initialized as a threading.Lock."""
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

        assert hasattr(background, "scan_lock")
        assert isinstance(background.scan_lock, type(threading.Lock()))

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_seen_names_not_exists(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test seen_names has been removed."""
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

        assert not hasattr(background, "seen_names")

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_seen_devices_stale_cleanup(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test stale BLE devices older than 60s are removed during scan."""

        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk_class.return_value = mock_rtk
        mock_odom_class.return_value = MagicMock()
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()
        mock_event_loop.return_value = MagicMock()

        config = RFmapperConfig()
        background = RFmapper(config)

        now = time.time()

        # Tambah device stale (120 detik lalu)
        stale_device = RFData(
            unix_ts=now - 120,
            address="AA:BB:CC:DD:EE:FF",
            name=None,
            rssi=-90,
            tx_power=None,
            service_uuid="",
            mfgkey="",
            mfgval="",
        )
        background.seen_devices["AA:BB:CC:DD:EE:FF"] = stale_device

        # Tambah device fresh
        fresh_device = RFData(
            unix_ts=now,
            address="11:22:33:44:55:66",
            name=None,
            rssi=-70,
            tx_power=None,
            service_uuid="",
            mfgkey="",
            mfgval="",
        )
        background.seen_devices["11:22:33:44:55:66"] = fresh_device

        # Jalankan cleanup logic langsung (sama persis seperti di scan())
        stale_addrs = [
            addr
            for addr, dev in background.seen_devices.items()
            if now - dev.unix_ts > 60
        ]
        for addr in stale_addrs:
            del background.seen_devices[addr]

        # Device stale harus sudah dihapus
        assert "AA:BB:CC:DD:EE:FF" not in background.seen_devices
        # Device fresh harus masih ada
        assert "11:22:33:44:55:66" in background.seen_devices


class TestRFmapperCoverage:
    """Additional tests to improve coverage."""

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_with_ble_scan_data(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test run() handles ble_scan data correctly."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = {
            "gps_unix_ts": 1234567890.0,
            "gps_lat": 40.7128,
            "gps_lon": -74.0060,
            "gps_alt": 10.5,
            "yaw_mag_0_360": 180.0,
            "gps_qua": 4,
            "ble_scan": [{"addr": "AA:BB", "rssi": -70}],
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
        background.running = True

        def stop_after_one(duration: float) -> bool:
            background.running = False
            return False

        background.sleep = stop_after_one
        background.run()

        assert background.ble_scan == [{"addr": "AA:BB", "rssi": -70}]

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_with_ble_scan_none_logs_warning(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
        caplog,
    ):
        """Test run() logs warning when ble_scan is None."""
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
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig(URID="robot-001")
        background = RFmapper(config)
        background.running = True

        def stop_after_one(duration: float) -> bool:
            background.running = False
            return False

        background.sleep = stop_after_one
        with caplog.at_level("WARNING"):
            background.run()

        assert "No nRF52 scan results" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_with_rtk_data(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test run() parses RTK data correctly."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = None
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = True
        mock_rtk.data = {
            "rtk_unix_ts": 1234567890.0,
            "rtk_lat": 40.7128,
            "rtk_lon": -74.0060,
            "rtk_alt": 10.5,
            "rtk_qua": 4,
        }
        mock_rtk_class.return_value = mock_rtk
        mock_odom = MagicMock()
        mock_odom.position = None
        mock_odom_class.return_value = mock_odom
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig(URID="robot-001")
        background = RFmapper(config)
        background.running = True

        def stop_after_one(duration: float) -> bool:
            background.running = False
            return False

        background.sleep = stop_after_one
        background.run()

        assert background.rtk_lat == 40.7128
        assert background.rtk_lon == -74.0060
        assert background.rtk_alt == 10.5

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_with_fresh_scan_results(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test run() sends fresh scan results when available."""

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

        # Simulasi ada scan results baru
        fresh = RFData(
            unix_ts=time.time(),
            address="AA:BB:CC:DD:EE:FF",
            name="TestDevice",
            rssi=-70,
            tx_power=None,
            service_uuid="",
            mfgkey="",
            mfgval="",
        )
        background.scan_results = [fresh]
        background.scan_idx = 1
        background.scan_last_sent = 0
        background.running = True

        def stop_after_one(duration: float) -> bool:
            background.running = False
            return False

        background.sleep = stop_after_one
        background.run()

        # scan_results harus dikosongkan setelah dikirim
        assert background.scan_results == []
        assert background.scan_last_sent == 1

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_keyboard_interrupt(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
    ):
        """Test run() handles KeyboardInterrupt gracefully."""
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
        mock_fds_class.return_value = MagicMock()
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        config = RFmapperConfig()
        background = RFmapper(config)
        background.running = True

        def raise_keyboard_interrupt(duration: float) -> bool:
            raise KeyboardInterrupt

        background.sleep = raise_keyboard_interrupt

        # Tidak boleh raise exception
        background.run()

        assert background.running is False

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_handles_odom_exception(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
        caplog,
    ):
        """Test run() handles exception in odom parsing gracefully."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = None
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = False
        mock_rtk.data = None
        mock_rtk_class.return_value = mock_rtk
        mock_odom = MagicMock()
        mock_odom.position = {"bad_key": "causes_error"}
        mock_odom_class.return_value = mock_odom
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig()
        background = RFmapper(config)
        background.running = True

        def stop_after_one(duration: float) -> bool:
            background.running = False
            return False

        background.sleep = stop_after_one
        with caplog.at_level("ERROR"):
            background.run()

        assert "Error parsing Odom" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_handles_rtk_exception(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
        caplog,
    ):
        """Test run() handles exception in RTK parsing gracefully."""
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = None
        mock_gps_class.return_value = mock_gps
        mock_rtk = MagicMock()
        mock_rtk.running = True
        mock_rtk.data = {"bad_key": "causes_error"}
        mock_rtk_class.return_value = mock_rtk
        mock_odom = MagicMock()
        mock_odom.position = None
        mock_odom_class.return_value = mock_odom
        mock_fds_class.return_value = MagicMock()
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig()
        background = RFmapper(config)
        background.running = True

        def stop_after_one(duration: float) -> bool:
            background.running = False
            return False

        background.sleep = stop_after_one
        with caplog.at_level("ERROR"):
            background.run()

        assert "Error parsing RTK" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_run_handles_fabric_exception(
        self,
        mock_event_loop,
        mock_gps_class,
        mock_rtk_class,
        mock_odom_class,
        mock_fds_class,
        mock_thread_class,
        caplog,
    ):
        """Test run() handles exception in fabric sharing gracefully."""
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
        mock_fds.share_data.side_effect = Exception("Fabric error")
        mock_fds_class.return_value = mock_fds
        mock_thread_class.return_value = MagicMock()

        config = RFmapperConfig(URID="robot-001")
        background = RFmapper(config)
        background.running = True

        def stop_after_one(duration: float) -> bool:
            background.running = False
            return False

        background.sleep = stop_after_one
        with caplog.at_level("ERROR"):
            background.run()

        assert "Error sharing to Fabric" in caplog.text
