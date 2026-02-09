import math
from unittest.mock import Mock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_gps_odom_reader import (
    UnitreeGo2GPSOdomReader,
    UnitreeGo2GPSOdomReaderConfig,
)


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.unitree_go2_gps_odom_reader.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_odom_provider():
    with patch("inputs.plugins.unitree_go2_gps_odom_reader.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_initialization_sets_defaults_and_raises_on_missing_config():
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=1.0, origin_lon=2.0, origin_yaw_deg=90.0, unitree_ethernet="eth0"
    )
    with (
        patch("inputs.plugins.unitree_go2_gps_odom_reader.IOProvider") as _,
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider"
        ) as mock_odom,
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    assert instance.lat0 == 1.0
    assert instance.lon0 == 2.0
    assert instance._yaw_offset == math.radians(90.0)
    assert instance.pose_x == 0.0
    assert instance.pose_y == 0.0
    assert instance.pose_yaw == 0.0
    assert instance.io_provider is not None
    assert instance.buf == []
    assert instance.descriptor_for_LLM == "Latitude, Longitude, and Yaw"
    mock_odom.assert_called_once_with(channel="eth0")


def test_initialization_raises_on_missing_origin_coordinates():
    config = UnitreeGo2GPSOdomReaderConfig(origin_lat=1.0)  # Missing lon and yaw
    with (
        patch("inputs.plugins.unitree_go2_gps_odom_reader.IOProvider"),
        patch("inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider"),
    ):
        with pytest.raises(
            ValueError, match="Missing origin coordinates or yaw in config."
        ):
            UnitreeGo2GPSOdomReader(config=config)


@pytest.mark.asyncio
async def test_update_pose_updates_internal_state_and_io_provider(
    mock_io_provider, mock_odom_provider
):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    mock_odom_provider.x = 10.0
    mock_odom_provider.y = 20.0
    mock_odom_provider.odom_yaw_m180_p180 = 30.0  # degrees

    expected_yaw_world_rad = math.radians(30.0)
    expected_yaw_offset_rad = math.radians(45.0)
    expected_yaw_raw = expected_yaw_world_rad + expected_yaw_offset_rad
    expected_yaw_wrapped = (expected_yaw_raw + math.pi) % (2 * math.pi) - math.pi

    await instance._update_pose()

    assert instance.pose_x == 10.0
    assert instance.pose_y == 20.0
    assert abs(instance.pose_yaw - expected_yaw_wrapped) < 1e-9

    expected_lat, expected_lon = instance._xy_to_latlon(10.0, 20.0)
    mock_io_provider.add_dynamic_variable.assert_any_call("latitude", expected_lat)
    mock_io_provider.add_dynamic_variable.assert_any_call("longitude", expected_lon)
    mock_io_provider.add_dynamic_variable.assert_any_call(
        "yaw_deg", math.degrees(expected_yaw_wrapped)
    )


@pytest.mark.asyncio
async def test_poll_calls_update_pose_and_returns_none(
    mock_io_provider, mock_odom_provider
):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    with patch.object(instance, "_update_pose") as mock_update:
        with patch("asyncio.sleep"):
            result = await instance._poll()

    mock_update.assert_awaited_once()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_message_to_buffer_and_calls_io_provider(
    mock_io_provider, mock_odom_provider
):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    test_input = "Test GPS message"
    initial_len = len(instance.buf)

    with patch("time.time", return_value=1234.0):
        await instance.raw_to_text(test_input)

    assert len(instance.buf) == initial_len + 1
    assert instance.buf[-1].message == test_input
    assert instance.buf[-1].timestamp == 1234.0
    mock_io_provider.add_input.assert_called_once_with(
        "UnitreeGo2GPSOdomReader", test_input, 1234.0
    )


@pytest.mark.asyncio
async def test_raw_to_text_does_nothing_if_input_none_or_empty(
    mock_io_provider, mock_odom_provider
):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    initial_len = len(instance.buf)

    await instance.raw_to_text(None)
    await instance.raw_to_text("")

    await instance.raw_to_text("   ")

    assert len(instance.buf) == initial_len + 1
    assert instance.buf[-1].message == ""
    mock_io_provider.add_input.assert_called_once_with(
        "UnitreeGo2GPSOdomReader", "", instance.buf[-1].timestamp
    )


def test_formatted_latest_buffer_empty_returns_none(
    mock_io_provider, mock_odom_provider
):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    result = instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_formats_and_clears_latest_message(
    mock_io_provider, mock_odom_provider
):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    msg = Message(timestamp=1234.0, message="Formatted GPS data")
    instance.buf = [msg]

    result = instance.formatted_latest_buffer()

    assert result is not None
    assert "Latitude, Longitude, and Yaw INPUT" in result
    assert "Formatted GPS data" in result
    assert len(instance.buf) == 0
    mock_io_provider.add_input.assert_called_once_with(
        "UnitreeGo2GPSOdomReader", "Formatted GPS data", 1234.0
    )


def test_xy_to_latlon_conversion(mock_io_provider, mock_odom_provider):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    dx = 100.0
    dy = 200.0

    assert instance.lat0 is not None
    assert instance.lon0 is not None
    lat0_rad = math.radians(instance.lat0)
    lon0_rad = math.radians(instance.lon0)

    lat_new_rad = lat0_rad + dy / 6_371_000.0
    lon_new_rad = lon0_rad + dx / (6_371_000.0 * math.cos(lat0_rad))

    expected_lat = math.degrees(lat_new_rad)
    expected_lon = math.degrees(lon_new_rad)

    calculated_lat, calculated_lon = instance._xy_to_latlon(dx, dy)

    assert abs(calculated_lat - expected_lat) < 1e-4
    assert abs(calculated_lon - expected_lon) < 1e-4


def test_wrap_angle_function(mock_io_provider, mock_odom_provider):
    config = UnitreeGo2GPSOdomReaderConfig(
        origin_lat=-33.868820, origin_lon=151.209295, origin_yaw_deg=45.0
    )
    with (
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.IOProvider",
            return_value=mock_io_provider,
        ),
        patch(
            "inputs.plugins.unitree_go2_gps_odom_reader.UnitreeGo2OdomProvider",
            return_value=mock_odom_provider,
        ),
    ):
        instance = UnitreeGo2GPSOdomReader(config=config)

    assert abs(instance._wrap_angle(0.0) - 0.0) < 1e-9
    assert abs(instance._wrap_angle(math.pi) - (-math.pi)) < 1e-9
    assert abs(instance._wrap_angle(-math.pi) - (-math.pi)) < 1e-9

    assert abs(instance._wrap_angle(2 * math.pi) - 0.0) < 1e-9
    assert abs(instance._wrap_angle(3 * math.pi) - (-math.pi)) < 1e-9
    assert abs(instance._wrap_angle(-2 * math.pi) - 0.0) < 1e-9
    assert abs(instance._wrap_angle(-3 * math.pi) - (-math.pi)) < 1e-9
