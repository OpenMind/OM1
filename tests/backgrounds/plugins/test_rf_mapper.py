import logging
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backgrounds.plugins.rf_mapper import RFmapper, RFmapperConfig
from providers.fabric_map_provider import RFData


def make_mapper(
    gps_data=None,
    rtk_data=None,
    odom_position=None,
    urid=None,
    rtk_running=False,
    fds_mock=None,
):
    """Create an RFmapper with all external deps mocked out."""
    with (
        patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop"),
        patch("backgrounds.plugins.rf_mapper.GpsProvider") as MockGps,
        patch("backgrounds.plugins.rf_mapper.RtkProvider") as MockRtk,
        patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider") as MockOdom,
        patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter") as MockFds,
        patch("backgrounds.plugins.rf_mapper.threading.Thread"),
    ):
        mock_gps = MagicMock()
        mock_gps.running = True
        mock_gps.data = gps_data
        MockGps.return_value = mock_gps

        mock_rtk = MagicMock()
        mock_rtk.running = rtk_running
        mock_rtk.data = rtk_data
        MockRtk.return_value = mock_rtk

        mock_odom = MagicMock()
        mock_odom.position = odom_position
        MockOdom.return_value = mock_odom

        _fds = fds_mock or MagicMock()
        MockFds.return_value = _fds

        mapper = RFmapper(RFmapperConfig(URID=urid))
        return mapper


def _stop_after_one(mapper):
    """Return a sleep stub that stops the run() loop after one iteration."""

    def _stop(duration: float) -> bool:
        mapper.running = False
        return False

    return _stop


class TestRFmapperConfig:
    """Test cases for RFmapperConfig."""

    def test_default_config(self):
        config = RFmapperConfig()
        assert config.name == "RFmapper"
        assert config.api_key is None
        assert config.URID is None
        assert config.unitree_ethernet is None

    def test_custom_name(self):
        config = RFmapperConfig(name="CustomMapper")
        assert config.name == "CustomMapper"

    def test_custom_api_key(self):
        config = RFmapperConfig(api_key="test-api-key")
        assert config.api_key == "test-api-key"

    def test_custom_urid(self):
        config = RFmapperConfig(URID="robot-001")
        assert config.URID == "robot-001"

    def test_custom_unitree_ethernet(self):
        config = RFmapperConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"

    def test_all_custom_values(self):
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


class TestRFmapperInit:
    """Test cases for RFmapper initialisation."""

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_initialization(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps = MagicMock(running=True)
        mock_gps_cls.return_value = mock_gps
        mock_rtk = MagicMock(running=False)
        mock_rtk_cls.return_value = mock_rtk
        mock_odom = MagicMock()
        mock_odom_cls.return_value = mock_odom
        mock_fds = MagicMock()
        mock_fds_cls.return_value = mock_fds
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(api_key="test-key", URID="robot-001"))

        assert bg.name == "RFmapper"
        assert bg.api_key == "test-key"
        assert bg.URID == "robot-001"
        assert bg.gps is mock_gps
        assert bg.rtk is mock_rtk
        assert bg.odom is mock_odom
        assert bg.fds is mock_fds
        mock_fds_cls.assert_called_once_with(
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
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
        caplog,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        with caplog.at_level("INFO"):
            RFmapper(RFmapperConfig())

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
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        RFmapper(RFmapperConfig())

        mock_thread.start.assert_called_once()

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_initial_state_values(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())

        assert bg.scan_results == []
        assert bg.scan_idx == 0
        assert bg.scan_last_sent == 0
        assert bg.payload_idx == 0
        assert bg.odom_x == 0.0
        assert bg.odom_y == 0.0
        assert bg.gps_lat == 0.0
        assert bg.gps_lon == 0.0
        assert bg.rtk_lat == 0.0
        assert bg.rtk_lon == 0.0

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_gps_on_reflects_provider_running(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())

        assert bg.gps_on is True
        assert bg.rtk_on is False

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_config_stored(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        config = RFmapperConfig(name="TestMapper", api_key="key-123")
        bg = RFmapper(config)

        assert bg.config is config
        assert bg.name == "TestMapper"
        assert bg.api_key == "key-123"


class TestRFmapperStop:

    @patch("backgrounds.plugins.rf_mapper.time.sleep")
    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_stop(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
        mock_sleep,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        bg = RFmapper(RFmapperConfig())
        bg.stop()

        assert bg.running is False
        mock_sleep.assert_called_with(1)
        mock_thread.join.assert_called_once()


class TestRFmapperThreadSafetyAndMemoryLeak:

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_scan_lock_exists(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())

        assert hasattr(bg, "scan_lock")
        assert isinstance(bg.scan_lock, type(threading.Lock()))

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_seen_names_not_exists(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())

        assert not hasattr(bg, "seen_names")

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_seen_devices_stale_cleanup(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True)
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock()
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        now = time.time()

        bg.seen_devices["AA:BB:CC:DD:EE:FF"] = RFData(
            unix_ts=now - 120,
            address="AA:BB:CC:DD:EE:FF",
            name=None,
            rssi=-90,
            tx_power=None,
            service_uuid="",
            mfgkey="",
            mfgval="",
        )
        bg.seen_devices["11:22:33:44:55:66"] = RFData(
            unix_ts=now,
            address="11:22:33:44:55:66",
            name=None,
            rssi=-70,
            tx_power=None,
            service_uuid="",
            mfgkey="",
            mfgval="",
        )

        stale_addrs = [
            addr for addr, dev in bg.seen_devices.items() if now - dev.unix_ts > 60
        ]
        for addr in stale_addrs:
            del bg.seen_devices[addr]

        assert "AA:BB:CC:DD:EE:FF" not in bg.seen_devices
        assert "11:22:33:44:55:66" in bg.seen_devices


class TestRFmapperRun:
    """Test run() data parsing and submission."""

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_parses_gps_data(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(
            running=True,
            data={
                "gps_unix_ts": 1234567890.0,
                "gps_lat": 40.7128,
                "gps_lon": -74.006,
                "gps_alt": 10.5,
                "yaw_mag_0_360": 180.0,
                "gps_qua": 4,
                "ble_scan": None,
            },
        )
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(URID="robot-001"))
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

        assert bg.gps_lat == 40.7128
        assert bg.gps_lon == -74.006
        assert bg.gps_alt == 10.5

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_parses_gps_yaw_mag(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(
            running=True,
            data={
                "gps_unix_ts": 1234567890.0,
                "gps_lat": 1.0,
                "gps_lon": 2.0,
                "gps_alt": 3.0,
                "yaw_mag_0_360": 270.0,
                "gps_qua": 4,
                "ble_scan": None,
            },
        )
        mock_rtk_cls.return_value = MagicMock(running=False)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

        assert bg.yaw_mag_0_360 == 270.0

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_parses_ble_scan_data(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(
            running=True,
            data={
                "gps_unix_ts": 1234567890.0,
                "gps_lat": 40.7128,
                "gps_lon": -74.006,
                "gps_alt": 10.5,
                "yaw_mag_0_360": 180.0,
                "gps_qua": 4,
                "ble_scan": [{"addr": "AA:BB", "rssi": -70}],
            },
        )
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(URID="robot-001"))
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

        assert bg.ble_scan == [{"addr": "AA:BB", "rssi": -70}]

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_ble_scan_none_logs_warning(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
        caplog,
    ):
        mock_gps_cls.return_value = MagicMock(
            running=True,
            data={
                "gps_unix_ts": 1234567890.0,
                "gps_lat": 40.7128,
                "gps_lon": -74.006,
                "gps_alt": 10.5,
                "yaw_mag_0_360": 180.0,
                "gps_qua": 4,
                "ble_scan": None,
            },
        )
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(URID="robot-001"))
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        with caplog.at_level("WARNING"):
            bg.run()

        assert "No nRF52 scan results" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_parses_odom_data(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(
            position={
                "odom_x": 1.5,
                "odom_y": 2.5,
                "odom_rockchip_ts": 100.0,
                "odom_subscriber_ts": 101.0,
                "odom_yaw_0_360": 90.0,
                "odom_yaw_m180_p180": 90.0,
            }
        )
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

        assert bg.odom_x == 1.5
        assert bg.odom_y == 2.5

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_parses_rtk_data(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(
            running=True,
            data={
                "rtk_unix_ts": 1234567890.0,
                "rtk_lat": 40.7128,
                "rtk_lon": -74.006,
                "rtk_alt": 10.5,
                "rtk_qua": 4,
            },
        )
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(URID="robot-001"))
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

        assert bg.rtk_lat == 40.7128
        assert bg.rtk_lon == -74.006
        assert bg.rtk_alt == 10.5

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_submits_fabric_data(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds = MagicMock()
        mock_fds_cls.return_value = mock_fds
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(URID="robot-001"))
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

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
    def test_no_urid_uses_unknown(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds = MagicMock()
        mock_fds_cls.return_value = mock_fds
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

        call_args = mock_fds.share_data.call_args[0][0]
        assert call_args.machine_id == "Unknown"

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_sends_fresh_scan_results(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(URID="robot-001"))
        bg.scan_results = [
            RFData(
                unix_ts=time.time(),
                address="AA:BB:CC:DD:EE:FF",
                name="TestDevice",
                rssi=-70,
                tx_power=None,
                service_uuid="",
                mfgkey="",
                mfgval="",
            )
        ]
        bg.scan_idx = 1
        bg.scan_last_sent = 0
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        bg.run()

        assert bg.scan_results == []
        assert bg.scan_last_sent == 1


class TestRFmapperRunExceptions:

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_keyboard_interrupt(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        bg.running = True
        bg.sleep = lambda duration: (_ for _ in ()).throw(KeyboardInterrupt)  # type: ignore[assignment]
        bg.run()

        assert bg.running is False

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_gps_exception_logged(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
        caplog,
    ):
        mock_gps = MagicMock(running=True)
        type(mock_gps).data = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("gps boom"))
        )
        mock_gps_cls.return_value = mock_gps
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR):
            bg.run()

        assert "Error parsing GPS" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_odom_exception_logged(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
        caplog,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position={"bad_key": "causes_error"})
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR):
            bg.run()

        assert "Error parsing Odom" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_rtk_exception_logged(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
        caplog,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(
            running=True, data={"bad_key": "causes_error"}
        )
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig())
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR):
            bg.run()

        assert "Error parsing RTK" in caplog.text

    @patch("backgrounds.plugins.rf_mapper.threading.Thread")
    @patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter")
    @patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider")
    @patch("backgrounds.plugins.rf_mapper.RtkProvider")
    @patch("backgrounds.plugins.rf_mapper.GpsProvider")
    @patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop")
    def test_fabric_exception_logged(
        self,
        mock_loop,
        mock_gps_cls,
        mock_rtk_cls,
        mock_odom_cls,
        mock_fds_cls,
        mock_thread_cls,
        caplog,
    ):
        mock_gps_cls.return_value = MagicMock(running=True, data=None)
        mock_rtk_cls.return_value = MagicMock(running=False, data=None)
        mock_odom_cls.return_value = MagicMock(position=None)
        mock_fds = MagicMock()
        mock_fds.share_data.side_effect = Exception("Fabric error")
        mock_fds_cls.return_value = mock_fds
        mock_thread_cls.return_value = MagicMock()

        bg = RFmapper(RFmapperConfig(URID="robot-001"))
        bg.running = True
        bg.sleep = _stop_after_one(bg)  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR):
            bg.run()

        assert "Error sharing to Fabric" in caplog.text


class TestDetectionCallbackBranches:
    """Cover the update-existing-device branches inside detection_callback."""

    @pytest.mark.asyncio
    async def test_updates_existing_device_tx_power_name_mfgval(self):
        with (
            patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop"),
            patch("backgrounds.plugins.rf_mapper.GpsProvider") as MockGps,
            patch("backgrounds.plugins.rf_mapper.RtkProvider") as MockRtk,
            patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider"),
            patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter"),
            patch("backgrounds.plugins.rf_mapper.threading.Thread"),
            patch("backgrounds.plugins.rf_mapper.BleakScanner") as MockScanner,
            patch(
                "backgrounds.plugins.rf_mapper.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            MockGps.return_value = MagicMock(running=True)
            MockRtk.return_value = MagicMock(running=False)

            captured = {}

            def fake_scanner(cb):
                captured["cb"] = cb
                s = MagicMock()
                s.start = AsyncMock()
                s.stop = AsyncMock()
                return s

            MockScanner.side_effect = fake_scanner

            mapper = RFmapper(RFmapperConfig())
            addr = "AA:BB:CC:DD:EE:FF"
            mapper.seen_devices[addr] = RFData(
                unix_ts=time.time(),
                address=addr,
                name=None,
                rssi=-80,
                tx_power=None,
                service_uuid="",
                mfgkey="",
                mfgval="AB",
            )
            await mapper.scan()

            fake_device = MagicMock()
            fake_device.address = addr
            fake_device.name = None
            fake_adv = MagicMock()
            fake_adv.local_name = "NewName"
            fake_adv.rssi = -75
            fake_adv.tx_power = -10
            fake_adv.manufacturer_data = {0x004C: bytes.fromhex("AABBCCDD")}
            fake_adv.service_uuids = []
            captured["cb"](fake_device, fake_adv)

            assert mapper.seen_devices[addr].tx_power == -10
            assert mapper.seen_devices[addr].name == "NewName"
            assert mapper.seen_devices[addr].mfgval == "AABBCCDD"

    @pytest.mark.asyncio
    async def test_new_device_with_device_name_and_service_uuid(self):
        with (
            patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop"),
            patch("backgrounds.plugins.rf_mapper.GpsProvider") as MockGps,
            patch("backgrounds.plugins.rf_mapper.RtkProvider") as MockRtk,
            patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider"),
            patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter"),
            patch("backgrounds.plugins.rf_mapper.threading.Thread"),
            patch("backgrounds.plugins.rf_mapper.BleakScanner") as MockScanner,
            patch(
                "backgrounds.plugins.rf_mapper.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            MockGps.return_value = MagicMock(running=True)
            MockRtk.return_value = MagicMock(running=False)

            captured = {}

            def fake_scanner(cb):
                captured["cb"] = cb
                s = MagicMock()
                s.start = AsyncMock()
                s.stop = AsyncMock()
                return s

            MockScanner.side_effect = fake_scanner

            mapper = RFmapper(RFmapperConfig())
            await mapper.scan()

            fake_device = MagicMock()
            fake_device.address = "11:22:33:44:55:66"
            fake_device.name = "DeviceName"
            fake_adv = MagicMock()
            fake_adv.local_name = None
            fake_adv.rssi = -60
            fake_adv.tx_power = None
            fake_adv.manufacturer_data = {}
            fake_adv.service_uuids = ["0000180d-0000-1000-8000-00805f9b34fb"]
            captured["cb"](fake_device, fake_adv)

            entry = mapper.seen_devices["11:22:33:44:55:66"]
            assert entry.name == "DeviceName"
            assert entry.service_uuid == "0000180d-0000-1000-8000-00805f9b34fb"

    @pytest.mark.asyncio
    async def test_scan_removes_stale_devices(self):
        """Cover: del self.seen_devices[addr] inside scan() for stale devices."""
        with (
            patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop"),
            patch("backgrounds.plugins.rf_mapper.GpsProvider") as MockGps,
            patch("backgrounds.plugins.rf_mapper.RtkProvider") as MockRtk,
            patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider"),
            patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter"),
            patch("backgrounds.plugins.rf_mapper.threading.Thread"),
            patch("backgrounds.plugins.rf_mapper.BleakScanner") as MockScanner,
            patch(
                "backgrounds.plugins.rf_mapper.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            MockGps.return_value = MagicMock(running=True)
            MockRtk.return_value = MagicMock(running=False)

            def fake_scanner(cb):
                s = MagicMock()
                s.start = AsyncMock()
                s.stop = AsyncMock()
                return s

            MockScanner.side_effect = fake_scanner

            mapper = RFmapper(RFmapperConfig())

            now = time.time()
            # Device stale > 60 detik
            mapper.seen_devices["AA:BB:CC:DD:EE:FF"] = RFData(
                unix_ts=now - 120,
                address="AA:BB:CC:DD:EE:FF",
                name=None,
                rssi=-90,
                tx_power=None,
                service_uuid="",
                mfgkey="",
                mfgval="",
            )
            # Device fresh
            mapper.seen_devices["11:22:33:44:55:66"] = RFData(
                unix_ts=now,
                address="11:22:33:44:55:66",
                name=None,
                rssi=-70,
                tx_power=None,
                service_uuid="",
                mfgkey="",
                mfgval="",
            )

            await mapper.scan()

            assert "AA:BB:CC:DD:EE:FF" not in mapper.seen_devices
            assert "11:22:33:44:55:66" in mapper.seen_devices

    @pytest.mark.asyncio
    async def test_scan_includes_named_devices_beyond_top20(self):
        with (
            patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop"),
            patch("backgrounds.plugins.rf_mapper.GpsProvider") as MockGps,
            patch("backgrounds.plugins.rf_mapper.RtkProvider") as MockRtk,
            patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider"),
            patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter"),
            patch("backgrounds.plugins.rf_mapper.threading.Thread"),
            patch("backgrounds.plugins.rf_mapper.BleakScanner") as MockScanner,
            patch(
                "backgrounds.plugins.rf_mapper.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            MockGps.return_value = MagicMock(running=True)
            MockRtk.return_value = MagicMock(running=False)

            def fake_scanner(cb):
                s = MagicMock()
                s.start = AsyncMock()
                s.stop = AsyncMock()
                return s

            MockScanner.side_effect = fake_scanner

            mapper = RFmapper(RFmapperConfig())
            now = time.time()
            for i in range(25):
                addr = f"AA:BB:CC:DD:{i:02X}:00"
                mapper.seen_devices[addr] = RFData(
                    unix_ts=now,
                    address=addr,
                    name=None,
                    rssi=-(50 + i),
                    tx_power=None,
                    service_uuid="",
                    mfgkey="",
                    mfgval="",
                )
            mapper.seen_devices["FF:FF:FF:FF:FF:FF"] = RFData(
                unix_ts=now,
                address="FF:FF:FF:FF:FF:FF",
                name="WeakButNamed",
                rssi=-99,
                tx_power=None,
                service_uuid="",
                mfgkey="",
                mfgval="",
            )

            result = await mapper.scan()

            assert "WeakButNamed" in [d.name for d in result]
            assert len(result) == 21


class TestScanTaskCoverage:
    """Cover _scan_task method body."""

    def test_scan_task_sets_running_and_updates_scan_results(self):
        with (
            patch("backgrounds.plugins.rf_mapper.asyncio.new_event_loop"),
            patch("backgrounds.plugins.rf_mapper.asyncio.set_event_loop"),
            patch("backgrounds.plugins.rf_mapper.GpsProvider") as MockGps,
            patch("backgrounds.plugins.rf_mapper.RtkProvider") as MockRtk,
            patch("backgrounds.plugins.rf_mapper.UnitreeGo2OdomProvider"),
            patch("backgrounds.plugins.rf_mapper.FabricDataSubmitter"),
            patch("backgrounds.plugins.rf_mapper.threading.Thread"),
        ):
            MockGps.return_value = MagicMock(running=True)
            MockRtk.return_value = MagicMock(running=False)

            mapper = RFmapper(RFmapperConfig())
            fake_rf = RFData(
                unix_ts=time.time(),
                address="CC:DD:EE:FF:00:11",
                name="TestDev",
                rssi=-65,
                tx_power=None,
                service_uuid="",
                mfgkey="",
                mfgval="",
            )
            call_count = {"n": 0}

            def fake_run_until_complete(coro):
                try:
                    coro.close()
                except Exception:
                    pass
                call_count["n"] += 1
                mapper.scan_idx += 1
                if call_count["n"] >= 2:
                    mapper.running = False
                return [fake_rf]

            mock_loop = MagicMock()
            mock_loop.run_until_complete.side_effect = fake_run_until_complete
            mapper.loop = mock_loop

            with patch("backgrounds.plugins.rf_mapper.time.sleep"):
                mapper._scan_task()

            assert call_count["n"] >= 1
            assert mapper.scan_idx >= 1
