import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from providers.fabric_map_provider import (
    FabricData,
    FabricDataSubmitter,
    RFData,
    RFDataRaw,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    FabricDataSubmitter.reset()  # type: ignore
    yield
    FabricDataSubmitter.reset()  # type: ignore


@pytest.fixture
def mock_requests():
    with patch("providers.fabric_map_provider.requests") as mock_req:
        yield mock_req


def test_rf_data_to_dict():
    rf_data = RFData(
        unix_ts=1234567890.0,
        address="AA:BB:CC:DD:EE:FF",
        name="TestDevice",
        rssi=-50,
        tx_power=-20,
        service_uuid="1234",
        mfgkey="key",
        mfgval="val",
    )

    data_dict = rf_data.to_dict()

    assert data_dict["unix_ts"] == 1234567890.0
    assert data_dict["address"] == "AA:BB:CC:DD:EE:FF"
    assert data_dict["name"] == "TestDevice"
    assert data_dict["rssi"] == -50
    assert data_dict["tx_power"] == -20


def test_rf_data_raw_to_dict():
    rf_data_raw = RFDataRaw(
        unix_ts=1234567890.0,
        address="AA:BB:CC:DD:EE:FF",
        rssi=-50,
        packet="test_packet",
    )

    data_dict = rf_data_raw.to_dict()

    assert data_dict["unix_ts"] == 1234567890.0
    assert data_dict["address"] == "AA:BB:CC:DD:EE:FF"
    assert data_dict["rssi"] == -50
    assert data_dict["packet"] == "test_packet"


def test_initialization():
    provider = FabricDataSubmitter()

    assert provider is not None


def test_singleton_pattern():
    """Test that FabricDataSubmitter follows the singleton pattern."""
    provider1 = FabricDataSubmitter()
    provider2 = FabricDataSubmitter()
    assert provider1 is provider2


def test_fabric_url():
    """Test that the fabric URL is set correctly."""
    provider = FabricDataSubmitter(base_url="http://test.endpoint")
    assert "test.endpoint" in provider.base_url


def test_share_data(mock_requests):
    """Test sharing fabric data with an API key."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_requests.post.return_value = mock_response

    provider = FabricDataSubmitter(api_key="test_key")

    fabric_data = FabricData(
        machine_id="test_machine",
        payload_idx=1,
        gps_unix_ts=1234567890.0,
        gps_lat=37.7749,
        gps_lon=-122.4194,
        gps_alt=10.0,
        gps_qua=2,
        rtk_unix_ts=0.0,
        rtk_lat=0.0,
        rtk_lon=0.0,
        rtk_alt=0.0,
        rtk_qua=0,
        mag=0.0,
        unix_ts=1234567890.0,
        odom_x=0.0,
        odom_y=0.0,
        odom_rockchip_ts=0.0,
        odom_subscriber_ts=0.0,
        odom_yaw_0_360=0.0,
        odom_yaw_m180_p180=0.0,
        rf_data=[],
        rf_data_raw=[],
    )

    provider.share_data(fabric_data)
    time.sleep(0.1)

    mock_requests.post.assert_called_once()


def test_share_data_no_api_key(mock_requests):
    """Test sharing fabric data without an API key."""
    provider = FabricDataSubmitter()

    fabric_data = FabricData(
        machine_id="test_machine",
        payload_idx=1,
        gps_unix_ts=1234567890.0,
        gps_lat=37.7749,
        gps_lon=-122.4194,
        gps_alt=10.0,
        gps_qua=2,
        rtk_unix_ts=0.0,
        rtk_lat=0.0,
        rtk_lon=0.0,
        rtk_alt=0.0,
        rtk_qua=0,
        mag=0.0,
        unix_ts=1234567890.0,
        odom_x=0.0,
        odom_y=0.0,
        odom_rockchip_ts=0.0,
        odom_subscriber_ts=0.0,
        odom_yaw_0_360=0.0,
        odom_yaw_m180_p180=0.0,
        rf_data=[],
        rf_data_raw=[],
    )

    provider.share_data(fabric_data)
    time.sleep(0.1)
    mock_requests.post.assert_not_called()


def test_share_data_with_rf_data(mock_requests):
    """Test sharing fabric data including RF data."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_requests.post.return_value = mock_response

    provider = FabricDataSubmitter(api_key="test_key")

    ble_data = [
        RFDataRaw(
            unix_ts=1234567890.0,
            address="AA:BB:CC:DD:EE:FF",
            rssi=-50,
            packet="packet1",
        )
    ]

    fabric_data = FabricData(
        machine_id="test_machine",
        payload_idx=1,
        gps_unix_ts=1234567890.0,
        gps_lat=37.7749,
        gps_lon=-122.4194,
        gps_alt=10.0,
        gps_qua=2,
        rtk_unix_ts=0.0,
        rtk_lat=0.0,
        rtk_lon=0.0,
        rtk_alt=0.0,
        rtk_qua=0,
        mag=0.0,
        unix_ts=1234567890.0,
        odom_x=0.0,
        odom_y=0.0,
        odom_rockchip_ts=0.0,
        odom_subscriber_ts=0.0,
        odom_yaw_0_360=0.0,
        odom_yaw_m180_p180=0.0,
        rf_data=[],
        rf_data_raw=ble_data,
    )

    provider.share_data(fabric_data)
    time.sleep(0.1)

    mock_requests.post.assert_called_once()


def test_write_to_local_file(mock_requests):
    """Test writing fabric data to a local file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = FabricDataSubmitter(api_key="test_key", write_to_local_file=True)

        provider.filename_base = os.path.join(tmpdir, "fabric")
        provider.filename_current = provider.update_filename()

        fabric_data = FabricData(
            machine_id="test_machine",
            payload_idx=1,
            gps_unix_ts=1234567890.0,
            gps_lat=37.7749,
            gps_lon=-122.4194,
            gps_alt=10.0,
            gps_qua=2,
            rtk_unix_ts=0.0,
            rtk_lat=0.0,
            rtk_lon=0.0,
            rtk_alt=0.0,
            rtk_qua=0,
            mag=0.0,
            unix_ts=1234567890.0,
            odom_x=0.0,
            odom_y=0.0,
            odom_rockchip_ts=0.0,
            odom_subscriber_ts=0.0,
            odom_yaw_0_360=0.0,
            odom_yaw_m180_p180=0.0,
            rf_data=[],
            rf_data_raw=[],
        )

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_requests.post.return_value = mock_response

        provider.share_data(fabric_data)
        time.sleep(0.1)

        assert os.path.exists(provider.filename_current)
