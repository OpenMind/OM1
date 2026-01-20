import sys
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def setup_ubtech_mock(monkeypatch):
    monkeypatch.setitem(sys.modules, "ubtech", Mock())
    monkeypatch.setitem(sys.modules, "ubtech.ubtechapi", Mock())
    monkeypatch.setitem(sys.modules, "mjpeg", Mock())
    monkeypatch.setitem(sys.modules, "mjpeg.client", Mock())


class TestUbtechCameraVideoStream:

    @patch("src.providers.ubtech_video_stream.YanAPI")
    @patch("src.providers.ubtech_video_stream.VideoStream.__init__")
    def test_init(self, mock_super_init, mock_YanAPI, monkeypatch):
        from src.providers.ubtech_video_stream import UbtechCameraVideoStream

        robot_ip = "192.168.1.100"

        # Define a simple callback function instead of lambda
        def frame_callback(x):
            pass  # Or print(x) if needed for debugging

        fps = 25
        resolution = (1280, 720)
        jpeg_quality = 80

        stream = UbtechCameraVideoStream(
            robot_ip=robot_ip,
            frame_callback=frame_callback,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )

        mock_super_init.assert_called_once_with(
            frame_callback=frame_callback,
            frame_callbacks=None,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )

        mock_YanAPI.yan_api_init.assert_called_once_with(robot_ip)

        assert stream.robot_ip == robot_ip
        assert stream.url == f"http://{robot_ip}:8000/stream.mjpg"
        assert stream.stream_client is None

    @patch("src.providers.ubtech_video_stream.YanAPI")
    @patch("src.providers.ubtech_video_stream.VideoStream.__init__")
    def test_init_with_defaults(self, mock_super_init, mock_YanAPI, monkeypatch):
        from src.providers.ubtech_video_stream import UbtechCameraVideoStream

        robot_ip = "192.168.1.101"

        stream = UbtechCameraVideoStream(robot_ip=robot_ip)

        mock_super_init.assert_called_once_with(
            frame_callback=None,
            frame_callbacks=None,
            fps=30,
            resolution=(640, 480),
            jpeg_quality=70,
        )

        mock_YanAPI.yan_api_init.assert_called_once_with(robot_ip)

        assert stream.robot_ip == robot_ip
        assert stream.url == f"http://{robot_ip}:8000/stream.mjpg"
        assert stream.stream_client is None

    @patch("src.providers.ubtech_video_stream.YanAPI")
    @patch("src.providers.ubtech_video_stream.VideoStream.__init__")
    def test_init_with_frame_callbacks(self, mock_super_init, mock_YanAPI, monkeypatch):
        from src.providers.ubtech_video_stream import UbtechCameraVideoStream

        robot_ip = "192.168.1.102"

        # Define simple callback functions instead of lambdas
        def frame_cb1(x):
            pass  # Or print(x) if needed for debugging

        def frame_cb2(x):
            pass  # Or print("another callback") if needed

        frame_callbacks_list = [frame_cb1, frame_cb2]

        stream = UbtechCameraVideoStream(
            robot_ip=robot_ip, frame_callbacks=frame_callbacks_list
        )

        mock_super_init.assert_called_once_with(
            frame_callback=None,
            frame_callbacks=frame_callbacks_list,
            fps=30,
            resolution=(640, 480),
            jpeg_quality=70,
        )

        mock_YanAPI.yan_api_init.assert_called_once_with(robot_ip)

        assert stream.robot_ip == robot_ip
        assert stream.url == f"http://{robot_ip}:8000/stream.mjpg"
        assert stream.stream_client is None
