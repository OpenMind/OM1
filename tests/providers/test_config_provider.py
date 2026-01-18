"""Tests for config_provider."""

import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Mock ALL external dependencies BEFORE any provider imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["json5"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["rclpy"] = MagicMock()
sys.modules["rclpy.node"] = MagicMock()
sys.modules["rclpy.qos"] = MagicMock()
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["nav_msgs.msg"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["websocket"] = MagicMock()
sys.modules["websockets"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["pyrealsense2"] = MagicMock()
sys.modules["mjpeg"] = MagicMock()
sys.modules["mjpeg.client"] = MagicMock()
sys.modules["unitree"] = MagicMock()
sys.modules["unitree_sdk2py"] = MagicMock()
sys.modules["unitree_sdk2py.core"] = MagicMock()
sys.modules["unitree_sdk2py.core.channel"] = MagicMock()


class TestConfigProvider:
    """Tests for ConfigProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Reset module cache before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if "providers" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

    @pytest.fixture
    def mock_zenoh_msgs(self):
        """Mock zenoh_msgs module."""
        mock_msgs = MagicMock()
        mock_msgs.ConfigRequest = MagicMock()
        mock_msgs.ConfigResponse = MagicMock()
        mock_msgs.String = MagicMock()
        mock_msgs.open_zenoh_session = MagicMock()
        mock_msgs.prepare_header = MagicMock()
        sys.modules["zenoh_msgs"] = mock_msgs
        return mock_msgs

    @pytest.fixture
    def mock_json5(self):
        """Mock json5 module."""
        mock_j5 = MagicMock()
        mock_j5.loads = MagicMock()
        mock_j5.load = MagicMock()
        sys.modules["json5"] = mock_j5
        return mock_j5

    def test_initialization(self, mock_zenoh_msgs, mock_json5):
        """Test provider initializes correctly."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath") as mock_abspath:
            mock_abspath.return_value = "/test/config/memory/.runtime.json5"
            provider = ConfigProvider()

            assert provider is not None
            assert provider.session is not None
            assert provider.config_response_publisher is not None
            assert provider.config_request_subscriber is not None
            assert provider.running is True

    def test_singleton_pattern(self, mock_zenoh_msgs, mock_json5):
        """Test ConfigProvider follows singleton pattern."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider1 = ConfigProvider()
            provider2 = ConfigProvider()
            assert provider1 is provider2

    def test_get_runtime_config_path(self, mock_zenoh_msgs, mock_json5):
        """Test _get_runtime_config_path returns correct path."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath") as mock_abspath:
            mock_abspath.return_value = "/test/config/memory/.runtime.json5"
            provider = ConfigProvider()

            assert provider.config_path == "/test/config/memory/.runtime.json5"

    def test_handle_config_request_get_config(self, mock_zenoh_msgs, mock_json5):
        """Test _handle_config_request for get config request."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test_payload"

            mock_request = MagicMock()
            mock_request.request_id = "test_request_id"
            mock_request.config = None
            mock_zenoh_msgs.ConfigRequest.deserialize.return_value = mock_request

            with patch.object(provider, "_send_config_response") as mock_send:
                provider._handle_config_request(mock_sample)
                mock_send.assert_called_once_with("test_request_id")

    def test_handle_config_request_set_config(self, mock_zenoh_msgs, mock_json5):
        """Test _handle_config_request for set config request."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test_payload"

            mock_config = MagicMock()
            mock_config.data = '{"test": "config"}'

            mock_request = MagicMock()
            mock_request.request_id = "test_request_id"
            mock_request.config = mock_config
            mock_zenoh_msgs.ConfigRequest.deserialize.return_value = mock_request

            with patch.object(provider, "_handle_set_config") as mock_set:
                provider._handle_config_request(mock_sample)
                mock_set.assert_called_once_with(
                    "test_request_id", '{"test": "config"}'
                )

    def test_handle_set_config_success(self, mock_zenoh_msgs, mock_json5):
        """Test _handle_set_config successfully updates config."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.config_path = "/test/config.json5"

            mock_json5.loads.return_value = {"test": "config"}

            with (
                patch("builtins.open", mock_open()) as mock_file,
                patch("os.rename") as mock_rename,
                patch.object(provider, "_send_config_response") as mock_send,
                patch("json.dump") as mock_dump,
            ):
                provider._handle_set_config("test_id", '{"test": "config"}')

                mock_file.assert_called_once_with("/test/config.json5.tmp", "w")
                mock_dump.assert_called_once()
                mock_rename.assert_called_once_with(
                    "/test/config.json5.tmp", "/test/config.json5"
                )
                mock_send.assert_called_once_with("test_id")

    def test_handle_set_config_failure(self, mock_zenoh_msgs, mock_json5):
        """Test _handle_set_config handles failure gracefully."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()

            mock_json5.loads.side_effect = ValueError("Invalid JSON")

            with patch.object(provider, "_send_error_response") as mock_error:
                provider._handle_set_config("test_id", "invalid_json")
                mock_error.assert_called_once()

    def test_send_config_response_success(self, mock_zenoh_msgs, mock_json5):
        """Test _send_config_response sends response successfully."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()

            mock_response = MagicMock()
            mock_zenoh_msgs.ConfigResponse.return_value = mock_response
            mock_zenoh_msgs.String.return_value = MagicMock()
            mock_zenoh_msgs.prepare_header.return_value = MagicMock()

            with patch.object(provider, "_get_config_snapshot") as mock_get:
                mock_get.return_value = {"test": "config"}

                provider._send_config_response("test_id")

                mock_response.serialize.assert_called_once()
                provider.config_response_publisher.put.assert_called_once()

    def test_send_error_response(self, mock_zenoh_msgs, mock_json5):
        """Test _send_error_response sends error message."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()

            mock_response = MagicMock()
            mock_zenoh_msgs.ConfigResponse.return_value = mock_response
            mock_zenoh_msgs.String.return_value = MagicMock()
            mock_zenoh_msgs.prepare_header.return_value = MagicMock()

            provider._send_error_response("test_id", "Test error")

            mock_response.serialize.assert_called_once()
            provider.config_response_publisher.put.assert_called_once()

    def test_get_config_snapshot_file_exists(self, mock_zenoh_msgs, mock_json5):
        """Test _get_config_snapshot when config file exists."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.config_path = "/test/config.json5"

            mock_json5.load.return_value = {"test": "config"}

            with (
                patch("os.path.exists", return_value=True),
                patch("builtins.open", mock_open()),
            ):
                result = provider._get_config_snapshot()

                assert result == {"test": "config"}

    def test_get_config_snapshot_file_not_exists(self, mock_zenoh_msgs, mock_json5):
        """Test _get_config_snapshot when config file doesn't exist."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.config_path = "/test/config.json5"

            with patch("os.path.exists", return_value=False):
                result = provider._get_config_snapshot()

                assert result == {}

    def test_get_config_snapshot_read_error(self, mock_zenoh_msgs, mock_json5):
        """Test _get_config_snapshot handles read errors."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.config_path = "/test/config.json5"

            mock_json5.load.side_effect = Exception("Read error")

            with (
                patch("os.path.exists", return_value=True),
                patch("builtins.open", mock_open()),
            ):
                result = provider._get_config_snapshot()

                assert result == {}

    def test_stop_when_running(self, mock_zenoh_msgs, mock_json5):
        """Test stop method when provider is running."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.running = True

            provider.stop()

            assert provider.running is False

    def test_stop_when_not_running(self, mock_zenoh_msgs, mock_json5):
        """Test stop method when provider is not running."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.running = False

            provider.stop()

            assert provider.running is False

    def test_initialization_zenoh_failure(self, mock_zenoh_msgs, mock_json5):
        """Test initialization handles Zenoh failure gracefully."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        mock_zenoh_msgs.open_zenoh_session.side_effect = Exception("Zenoh error")

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()

            assert provider is not None
            assert provider.running is False

    def test_handle_config_request_deserialization_error(
        self, mock_zenoh_msgs, mock_json5
    ):
        """Test _handle_config_request handles deserialization errors."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"invalid_payload"
            mock_zenoh_msgs.ConfigRequest.deserialize.side_effect = Exception(
                "Deserialization error"
            )

            # Should not raise exception
            provider._handle_config_request(mock_sample)

    def test_send_config_response_no_publisher(self, mock_zenoh_msgs, mock_json5):
        """Test _send_config_response when publisher is None."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.config_response_publisher = None

            with patch.object(provider, "_get_config_snapshot") as mock_get:
                mock_get.return_value = {"test": "config"}

                # Should not raise exception
                provider._send_config_response("test_id")

    def test_send_error_response_no_publisher(self, mock_zenoh_msgs, mock_json5):
        """Test _send_error_response when publisher is None."""
        from providers.config_provider import ConfigProvider

        if hasattr(ConfigProvider, "reset"):
            ConfigProvider.reset()

        with patch("os.path.exists"), patch("os.path.abspath"):
            provider = ConfigProvider()
            provider.config_response_publisher = None

            # Should not raise exception
            provider._send_error_response("test_id", "Test error")
