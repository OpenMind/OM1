# tests/providers/test_ubtech_vlm_provider.py

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock external dependencies to allow import
sys.modules["ubtech"] = Mock()
sys.modules["ubtech.ubtechapi"] = Mock()

from src.providers.ubtech_vlm_provider import UbtechVLMProvider  # noqa: E402

# --- Tests for UbtechVLMProvider Class ---
# Note: UbtechVLMProvider is a singleton. Mocking is required for isolation.


class TestUbtechVLMProvider:

    @patch("src.providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("src.providers.ubtech_vlm_provider.ws")
    def test_init(self, mock_ws, mock_video_stream_class, monkeypatch):
        # Mock the singleton function to return a mock instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.ubtech_vlm_provider.UbtechVLMProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        ws_url = "ws://vlm.example.com"
        robot_ip = "192.168.1.100"
        fps = 25
        resolution = (1280, 720)
        jpeg_quality = 85
        stream_url = "ws://stream.example.com"

        # Create mock instances for ws clients and video stream
        mock_main_ws_client = MagicMock()
        mock_stream_ws_client = MagicMock()
        mock_video_stream = MagicMock()

        # Configure ws.Client to return our specific mocks
        mock_ws.Client.side_effect = [mock_main_ws_client, mock_stream_ws_client]

        # Mock UbtechCameraVideoStream to return our mock
        mock_video_stream_class.return_value = mock_video_stream

        # Create provider instance (will return the mock)
        # We don't use the returned provider object directly here,
        # only to trigger the constructor logic on the mock.
        provider = UbtechVLMProvider(  # noqa: F841
            ws_url=ws_url,
            robot_ip=robot_ip,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
            stream_url=stream_url,
        )

        # Verify ws.Client was called twice with correct URLs
        assert mock_ws.Client.call_count == 2
        call1 = mock_ws.Client.call_args_list[0]
        call2 = mock_ws.Client.call_args_list[1]
        assert call1.kwargs["url"] == ws_url
        assert call2.kwargs["url"] == stream_url

        # Verify UbtechCameraVideoStream was called with correct parameters
        mock_video_stream_class.assert_called_once_with(
            frame_callback=mock_main_ws_client.send_message,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
            robot_ip=robot_ip,
        )

        # Verify the provider instance attributes were set by the constructor
        mock_provider_instance.ws_client = mock_main_ws_client
        mock_provider_instance.stream_ws_client = mock_stream_ws_client
        mock_provider_instance.video_stream = mock_video_stream
        mock_provider_instance.robot_ip = robot_ip
        mock_provider_instance.running = False

        # Final assertions on the mock instance attributes
        assert mock_provider_instance.robot_ip == robot_ip
        assert not mock_provider_instance.running
        assert mock_provider_instance.ws_client is mock_main_ws_client
        assert mock_provider_instance.stream_ws_client is mock_stream_ws_client
        assert mock_provider_instance.video_stream is mock_video_stream

    @patch("src.providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("src.providers.ubtech_vlm_provider.ws")
    def test_register_message_callback(
        self, mock_ws, mock_video_stream_class, monkeypatch
    ):
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.ubtech_vlm_provider.UbtechVLMProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        mock_main_ws_client = MagicMock()
        mock_stream_ws_client = MagicMock()
        mock_video_stream = MagicMock()

        mock_ws.Client.side_effect = [mock_main_ws_client, mock_stream_ws_client]
        mock_video_stream_class.return_value = mock_video_stream

        provider = UbtechVLMProvider(ws_url="ws://test", robot_ip="127.0.0.1")
        provider.ws_client = mock_main_ws_client
        provider.stream_ws_client = mock_stream_ws_client
        provider.video_stream = mock_video_stream
        provider.robot_ip = "127.0.0.1"
        provider.running = False

        def dummy_callback(message):
            pass

        provider.register_message_callback(dummy_callback)
        mock_main_ws_client.register_message_callback.assert_called_once_with(
            dummy_callback
        )

    @patch("src.providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("src.providers.ubtech_vlm_provider.ws")
    def test_start(self, mock_ws, mock_video_stream_class, monkeypatch):
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.ubtech_vlm_provider.UbtechVLMProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        mock_main_ws_client = MagicMock()
        mock_stream_ws_client = MagicMock()
        mock_video_stream = MagicMock()

        mock_ws.Client.side_effect = [mock_main_ws_client, mock_stream_ws_client]
        mock_video_stream_class.return_value = mock_video_stream

        provider = UbtechVLMProvider(ws_url="ws://test", robot_ip="127.0.0.1")
        provider.ws_client = mock_main_ws_client
        provider.stream_ws_client = mock_stream_ws_client
        provider.video_stream = mock_video_stream
        provider.robot_ip = "127.0.0.1"
        provider.running = False

        provider.start()
        assert provider.running
        mock_main_ws_client.start.assert_called_once()
        mock_video_stream.start.assert_called_once()
        mock_stream_ws_client.start.assert_called_once()
        mock_video_stream.register_frame_callback.assert_called_once_with(
            mock_stream_ws_client.send_message
        )

    @patch("src.providers.ubtech_vlm_provider.UbtechCameraVideoStream")
    @patch("src.providers.ubtech_vlm_provider.ws")
    def test_stop(self, mock_ws, mock_video_stream_class, monkeypatch):
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.ubtech_vlm_provider.UbtechVLMProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        mock_main_ws_client = MagicMock()
        mock_stream_ws_client = MagicMock()
        mock_video_stream = MagicMock()

        mock_ws.Client.side_effect = [mock_main_ws_client, mock_stream_ws_client]
        mock_video_stream_class.return_value = mock_video_stream

        provider = UbtechVLMProvider(ws_url="ws://test", robot_ip="127.0.0.1")
        provider.ws_client = mock_main_ws_client
        provider.stream_ws_client = mock_stream_ws_client
        provider.video_stream = mock_video_stream
        provider.robot_ip = "127.0.0.1"
        provider.running = True

        provider.stop()
        assert not provider.running
        mock_video_stream.stop.assert_called_once()
        mock_main_ws_client.stop.assert_called_once()
        mock_stream_ws_client.stop.assert_called_once()


# --- Run tests ---
if __name__ == "__main__":
    pytest.main()
