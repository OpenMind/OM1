import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from zenoh_msgs import String


class FakeNode:
    def __init__(self, node_name):
        pass

    def create_publisher(self, msg_type, topic, qos_profile):
        return Mock()


mock_rclpy = MagicMock()
mock_rclpy.ok.return_value = True
sys.modules["rclpy"] = mock_rclpy

mock_node_module = MagicMock()
mock_node_module.Node = FakeNode
sys.modules["rclpy.node"] = mock_node_module

sys.modules["std_msgs"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()

from providers.ros2_publisher_provider import ROS2PublisherProvider  # noqa: E402


@pytest.fixture
def provider():
    """Create a provider instance with mocked dependencies."""
    with patch("providers.ros2_publisher_provider.rclpy") as mock_rclpy_module:
        mock_rclpy_module.ok.return_value = True

        provider = ROS2PublisherProvider("test_topic")

        yield provider


def test_initialization(provider):
    assert provider is not None
    assert not provider.running


def test_add_pending_message(provider):
    provider.add_pending_message("Hello")

    assert not provider._pending_messages.empty()


def test_start(provider):
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    provider.stop()


def test_start_already_running(provider):
    provider.start()

    thread1 = provider._thread

    provider.start()

    assert provider._thread == thread1

    provider.stop()


def test_stop(provider):
    provider.start()
    provider.stop()

    assert not provider.running


def test_publish_message(provider):
    mock_publisher = MagicMock()
    provider.publisher_ = mock_publisher

    msg = String(data="test")
    provider._publish_message(msg)

    mock_publisher.publish.assert_called_once_with(msg)
