import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from zenoh_msgs import String


class FakeNode:
    def __init__(self, node_name):
        self._destroyed_publishers = []
        self._node_destroyed = False

    def create_publisher(self, msg_type, topic, qos_profile):
        return Mock()

    def destroy_publisher(self, publisher):
        self._destroyed_publishers.append(publisher)

    def destroy_node(self):
        self._node_destroyed = True


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


def test_init_calls_rclpy_init_when_not_ok():
    """
    Ensure __init__ only calls rclpy.init() when rclpy.ok() is False.
    This covers the non-ok initialization branch without requiring ROS2.
    """
    with patch("providers.ros2_publisher_provider.rclpy") as mock_rclpy_module:
        mock_rclpy_module.ok.return_value = False

        _provider = ROS2PublisherProvider("test_topic")  # noqa: F841

        mock_rclpy_module.init.assert_called_once()


def test_init_handles_rclpy_init_exception():
    """
    If rclpy.init() fails, provider should not raise during construction.
    """
    with patch("providers.ros2_publisher_provider.rclpy") as mock_rclpy_module:
        mock_rclpy_module.ok.return_value = False
        mock_rclpy_module.init.side_effect = Exception("boom")

        _provider = ROS2PublisherProvider("test_topic")  # noqa: F841


def test_stop_destroys_publisher_and_node(provider):
    """
    stop() should best-effort destroy publisher and node when APIs exist.
    """
    pub = Mock()
    provider.publisher_ = pub

    provider.stop()

    assert pub in provider._destroyed_publishers
    assert provider._node_destroyed is True


def test_stop_handles_destroy_exceptions(provider):
    """
    stop() should never raise if destroy_publisher/destroy_node fail.
    """
    provider.publisher_ = Mock()

    provider.destroy_publisher = Mock(side_effect=Exception("destroy pub failed"))
    provider.destroy_node = Mock(side_effect=Exception("destroy node failed"))

    provider.stop()


def test_stop_no_publisher_does_not_destroy(provider):
    """
    If publisher_ is None, destroy_publisher should not be invoked.
    """
    provider.publisher_ = None

    provider.destroy_publisher = Mock()
    provider.destroy_node = Mock()

    provider.stop()

    provider.destroy_publisher.assert_not_called()
    provider.destroy_node.assert_called_once()
