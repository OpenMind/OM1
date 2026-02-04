from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.actions.manage_resources.connector.zenoh_resource_mgr import (
    ZenohResourceManager,
)
from src.actions.manage_resources.interface import ManageResourcesInterface


@pytest.mark.asyncio
async def test_adjust_network_qos_success():
    config = {}
    manager = ZenohResourceManager(config)

    mock_session = MagicMock()
    with (
        patch.object(
            manager,
            "_initialize_zenoh_session",
            side_effect=lambda: setattr(manager, "zenoh_session", mock_session)
            or AsyncMock(return_value=True),
        ),
        patch("logging.info") as mock_log_info,
    ):

        result = await manager.adjust_network_qos("sensor/lidar/data", "high")

        assert result
        assert manager.zenoh_session is mock_session
        mock_log_info.assert_any_call(
            "[ZenohResourceManager] Attempting to apply QoS profile for 'sensor/lidar/data' with priority 'high': {'zenoh_priority': 'high', 'zenoh_congestion_control': 'block', 'zenoh_reliability': 'reliable'}"
        )


@pytest.mark.asyncio
async def test_adjust_network_qos_invalid_priority():
    config = {}
    manager = ZenohResourceManager(config)

    mock_session = MagicMock()
    manager.zenoh_session = mock_session

    with (
        patch.object(manager, "_initialize_zenoh_session") as mock_init,
        patch("logging.error") as mock_log_error,
    ):

        result = await manager.adjust_network_qos(
            "sensor/lidar/data", "invalid_priority"
        )

        assert not result
        mock_init.assert_not_called()
        mock_log_error.assert_called_once_with(
            "[ZenohResourceManager] Invalid priority level: invalid_priority. Valid options: ['realtime', 'high', 'medium', 'low']"
        )


@pytest.mark.asyncio
async def test_adjust_cpu_priority_success():
    config = {}
    manager = ZenohResourceManager(config)

    with patch("logging.info") as mock_log_info:

        result = await manager.adjust_cpu_priority("path_planner", "high")

        assert result
        mock_log_info.assert_called_once_with(
            "[ManageResources] Attempting to adjust CPU priority for task 'path_planner' to 'high'."
        )


@pytest.mark.asyncio
async def test_adjust_cpu_priority_invalid_priority():
    config = {}
    manager = ZenohResourceManager(config)

    with patch("logging.error") as mock_log_error:

        result = await manager.adjust_cpu_priority("path_planner", "invalid_priority")

        assert not result
        mock_log_error.assert_called_once_with(
            "[ManageResources] Invalid CPU priority level: invalid_priority. Valid options: ['critical', 'high', 'normal', 'low']"
        )


class MockManageResourcesInterface(ManageResourcesInterface):
    def __init__(self, config):
        super().__init__(config)
        self.adjust_network_qos_called_with = None
        self.adjust_cpu_priority_called_with = None
        self.adjust_network_qos_return_value = True
        self.adjust_cpu_priority_return_value = True

    async def adjust_network_qos(
        self,
        target: str,
        priority: str,
        reliability: str = "reliable",
        durability: str = "volatile",
    ) -> bool:
        self.adjust_network_qos_called_with = (
            target,
            priority,
            reliability,
            durability,
        )
        return self.adjust_network_qos_return_value

    async def adjust_cpu_priority(self, task_name: str, priority: str) -> bool:
        self.adjust_cpu_priority_called_with = (task_name, priority)
        return self.adjust_cpu_priority_return_value


@pytest.mark.asyncio
async def test_execute_adjust_qos_command():
    config = {}
    interface_impl = MockManageResourcesInterface(config)

    mock_message = MagicMock()
    mock_message.message = "Adjust QoS for sensor/lidar/data to high priority"
    mock_message.timestamp = 1234567890

    with patch.object(interface_impl.io_provider, "add_input") as mock_add_input:
        result = await interface_impl.execute(mock_message)

    assert result
    assert interface_impl.adjust_network_qos_called_with == (
        "sensor/lidar/data",
        "high",
        "reliable",
        "volatile",
    )
    mock_add_input.assert_called_once_with(
        "Resource Manager", "Adjusted QoS for sensor/lidar/data to high.", 1234567890
    )


@pytest.mark.asyncio
async def test_execute_adjust_cpu_command():
    config = {}
    interface_impl = MockManageResourcesInterface(config)

    mock_message = MagicMock()
    mock_message.message = "Adjust CPU priority for path_planner to high"
    mock_message.timestamp = 1234567890

    with patch.object(interface_impl.io_provider, "add_input") as mock_add_input:
        result = await interface_impl.execute(mock_message)

    assert result
    assert interface_impl.adjust_cpu_priority_called_with == ("path_planner", "high")
    mock_add_input.assert_called_once_with(
        "Resource Manager",
        "Adjusted CPU priority for path_planner to high.",
        1234567890,
    )


@pytest.mark.asyncio
async def test_execute_unknown_command():
    config = {}
    interface_impl = MockManageResourcesInterface(config)

    mock_message = MagicMock()
    mock_message.message = "Do something completely different"
    mock_message.timestamp = 1234567890

    result = await interface_impl.execute(mock_message)

    assert not result
    assert interface_impl.adjust_network_qos_called_with is None
    assert interface_impl.adjust_cpu_priority_called_with is None


@pytest.mark.asyncio
async def test_execute_parse_error():
    config = {}
    interface_impl = MockManageResourcesInterface(config)

    mock_message = MagicMock()
    mock_message.message = "Adjust QoS for sensor/lidar/data"
    mock_message.timestamp = 1234567890

    result = await interface_impl.execute(mock_message)

    assert not result
    assert interface_impl.adjust_network_qos_called_with is None
    assert interface_impl.adjust_cpu_priority_called_with is None


@pytest.mark.asyncio
async def test_execute_adjust_qos_failure():
    config = {}
    interface_impl = MockManageResourcesInterface(config)
    interface_impl.adjust_network_qos_return_value = False

    mock_message = MagicMock()
    mock_message.message = "Adjust QoS for sensor/lidar/data to high priority"
    mock_message.timestamp = 1234567890

    with patch.object(interface_impl.io_provider, "add_input") as mock_add_input:
        result = await interface_impl.execute(mock_message)

    assert not result
    assert interface_impl.adjust_network_qos_called_with == (
        "sensor/lidar/data",
        "high",
        "reliable",
        "volatile",
    )
    mock_add_input.assert_called_once_with(
        "Resource Manager", "Failed to adjust QoS for sensor/lidar/data.", 1234567890
    )
