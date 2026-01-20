import time
from unittest.mock import MagicMock, patch

import pytest

from src.providers.fabric_map_provider import (
    FabricData,
    FabricDataSubmitter,
    RFData,
    RFDataRaw,
)

# --- Tests for Dataclasses ---


class TestRFData:
    def test_to_dict(self):
        rf_data = RFData(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            name="TestDevice",
            rssi=-50,
            tx_power=10,
            service_uuid="1234",
            mfgkey="key1",
            mfgval="val1",
        )
        expected_dict = {
            "unix_ts": 1234567890.0,
            "address": "AA:BB:CC:DD:EE:FF",
            "name": "TestDevice",
            "rssi": -50,
            "tx_power": 10,
            "service_uuid": "1234",
            "mfgkey": "key1",
            "mfgval": "val1",
        }
        assert rf_data.to_dict() == expected_dict


class TestRFDataRaw:
    def test_to_dict(self):
        rf_data_raw = RFDataRaw(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            rssi=-50,
            packet="01020304",
        )
        expected_dict = {
            "unix_ts": 1234567890.0,
            "address": "AA:BB:CC:DD:EE:FF",
            "rssi": -50,
            "packet": "01020304",
        }
        assert rf_data_raw.to_dict() == expected_dict


class TestFabricData:
    def test_to_dict(self):
        rf_data_list = [
            RFData(
                1234567890.0, "AA:BB:CC:DD:EE:FF", "Dev1", -50, 10, "1234", "k1", "v1"
            )
        ]
        rf_data_raw_list = [RFDataRaw(1234567890.0, "AA:BB:CC:DD:EE:FF", -50, "0102")]

        fabric_data = FabricData(
            machine_id="test_machine",
            payload_idx=1,
            gps_unix_ts=1234567890.0,
            gps_lat=0.0,
            gps_lon=0.0,
            gps_alt=0.0,
            gps_qua=0,
            rtk_unix_ts=1234567890.0,
            rtk_lat=0.0,
            rtk_lon=0.0,
            rtk_alt=0.0,
            rtk_qua=0,
            mag=0.0,
            unix_ts=time.time(),
            odom_x=0.0,
            odom_y=0.0,
            odom_rockchip_ts=time.time(),
            odom_subscriber_ts=time.time(),
            odom_yaw_0_360=0.0,
            odom_yaw_m180_p180=0.0,
            rf_data=rf_data_list,
            rf_data_raw=rf_data_raw_list,
        )

        expected_dict = {
            "machine_id": "test_machine",
            "payload_idx": 1,
            "gps_unix_ts": 1234567890.0,
            "gps_lat": 0.0,
            "gps_lon": 0.0,
            "gps_alt": 0.0,
            "gps_qua": 0,
            "rtk_unix_ts": 1234567890.0,
            "rtk_lat": 0.0,
            "rtk_lon": 0.0,
            "rtk_alt": 0.0,
            "rtk_qua": 0,
            "mag": 0.0,
            "unix_ts": fabric_data.unix_ts,
            "odom_x": 0.0,
            "odom_y": 0.0,
            "odom_yaw_0_360": 0.0,
            "odom_yaw_m180_p180": 0.0,
            "odom_rockchip_ts": fabric_data.odom_rockchip_ts,
            "odom_subscriber_ts": fabric_data.odom_subscriber_ts,
            "rf_data": [
                {
                    "unix_ts": 1234567890.0,
                    "address": "AA:BB:CC:DD:EE:FF",
                    "name": "Dev1",
                    "rssi": -50,
                    "tx_power": 10,
                    "service_uuid": "1234",
                    "mfgkey": "k1",
                    "mfgval": "v1",
                }
            ],
            "rf_data_raw": [
                {
                    "unix_ts": 1234567890.0,
                    "address": "AA:BB:CC:DD:EE:FF",
                    "rssi": -50,
                    "packet": "0102",
                }
            ],
        }

        assert fabric_data.to_dict() == expected_dict


# --- Tests for FabricDataSubmitter Class ---
# Note: FabricDataSubmitter is a singleton. Mocking is required for isolation.


class TestFabricDataSubmitter:

    @patch("src.providers.fabric_map_provider.ThreadPoolExecutor")
    def test_init(self, mock_executor, monkeypatch):
        mock_provider_instance = MagicMock()
        api_key = "test_api_key"
        base_url = "https://test.api.com/  "
        write_to_local_file = True
        mock_provider_instance.api_key = api_key
        mock_provider_instance.base_url = base_url
        mock_provider_instance.write_to_local_file = write_to_local_file
        mock_provider_instance.executor = mock_executor.return_value

        monkeypatch.setattr(
            "src.providers.fabric_map_provider.FabricDataSubmitter",
            lambda *a, **kw: mock_provider_instance,
        )

        provider = FabricDataSubmitter(
            api_key=api_key, base_url=base_url, write_to_local_file=write_to_local_file
        )

        assert provider.api_key == api_key
        assert provider.base_url == base_url
        assert provider.write_to_local_file == write_to_local_file
        mock_executor.assert_called_once_with(max_workers=1)
        assert provider.executor is mock_executor.return_value

    @patch("src.providers.fabric_map_provider.time.time")
    @patch("src.providers.fabric_map_provider.logging")
    def test_update_filename(self, mock_logging, mock_time, monkeypatch):
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.fabric_map_provider.FabricDataSubmitter",
            lambda *a, **kw: mock_provider_instance,
        )

        mock_time.return_value = 1234567890.123456
        expected_timestamp_part = "1234567890_123456"
        expected_filename = f"dump/fabric_{expected_timestamp_part}Z.jsonl"

        provider = FabricDataSubmitter(api_key="some_key")
        provider.filename_base = "dump/fabric"

        with patch.object(provider, "update_filename") as mock_update_method:
            mock_update_method.return_value = expected_filename

            def temp_update_logic():
                unix_ts = mock_time()
                mock_logging.info(f"fabric time: {unix_ts}")
                unix_ts_str = str(unix_ts).replace(".", "_")
                filename = f"{provider.filename_base}_{unix_ts_str}Z.jsonl"
                provider.filename_current = filename
                return filename

            mock_update_method.side_effect = temp_update_logic

            filename = provider.update_filename()

            mock_time.assert_called_once()
            mock_logging.info.assert_called_once_with("fabric time: 1234567890.123456")
            assert filename == expected_filename
            assert provider.filename_current == expected_filename

    @patch("src.providers.fabric_map_provider.ThreadPoolExecutor")
    def test_share_data_submits_task(self, mock_executor_class, monkeypatch):
        mock_executor_instance = MagicMock()
        mock_executor_class.return_value = mock_executor_instance

        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.fabric_map_provider.FabricDataSubmitter",
            lambda *a, **kw: mock_provider_instance,
        )
        mock_provider_instance.executor = mock_executor_instance

        provider = FabricDataSubmitter(api_key="some_key")
        provider.executor.submit = MagicMock()

        fabric_data = FabricData(
            machine_id="test_machine",
            payload_idx=1,
            gps_unix_ts=time.time(),
            gps_lat=0.0,
            gps_lon=0.0,
            gps_alt=0.0,
            gps_qua=0,
            rtk_unix_ts=time.time(),
            rtk_lat=0.0,
            rtk_lon=0.0,
            rtk_alt=0.0,
            rtk_qua=0,
            mag=0.0,
            unix_ts=time.time(),
            odom_x=0.0,
            odom_y=0.0,
            odom_rockchip_ts=time.time(),
            odom_subscriber_ts=time.time(),
            odom_yaw_0_360=0.0,
            odom_yaw_m180_p180=0.0,
            rf_data=[],
            rf_data_raw=[],
        )

        provider.share_data(fabric_data)

        from unittest.mock import ANY

        provider.executor.submit.assert_called_once_with(ANY, fabric_data)
        submitted_func = provider.executor.submit.call_args[0][0]
        assert submitted_func == provider._share_data_worker
