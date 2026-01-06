import base64
import time
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from providers.unitree_camera_vlm_provider import (
    UnitreeCameraVideoStream,
    UnitreeCameraVLMProvider,
)


class MockVideoClient:
    def __init__(self):
        self.init_called = False

    def Init(self):
        self.init_called = True
        return 0

    def GetImageSample(self):
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        _, img_encoded = cv2.imencode(".jpg", img)
        return 0, img_encoded.tobytes()


@pytest.fixture
def mock_video_client():
    # Check if VideoClient is available (might not be if Unitree SDK is not installed)
    import providers.unitree_camera_vlm_provider as provider_module

    if not hasattr(provider_module, "VideoClient"):
        pytest.skip("Unitree SDK not available, VideoClient not found")

    mock_client = MockVideoClient()
    with patch(
        "providers.unitree_camera_vlm_provider.VideoClient", return_value=mock_client
    ):
        yield mock_client


@pytest.fixture
def ws_url():
    return "ws://test.url"


@pytest.fixture
def fps():
    return 30


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeCameraVLMProvider.reset()  # type: ignore
    yield
    UnitreeCameraVLMProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    mock_ws_client = Mock()
    mock_video_stream = Mock()
    with (
        patch(
            "providers.unitree_camera_vlm_provider.ws.Client",
            return_value=mock_ws_client,
        ) as mock_ws_client_class,
        patch(
            "providers.unitree_camera_vlm_provider.UnitreeCameraVideoStream",
            return_value=mock_video_stream,
        ) as mock_video_stream_class,
    ):
        yield mock_ws_client_class, mock_video_stream_class, mock_ws_client, mock_video_stream


def test_video_stream_initialization(mock_video_client):
    callback = Mock()
    stream = UnitreeCameraVideoStream(frame_callback=callback, fps=30)

    assert len(stream.frame_callbacks) == 1
    assert stream.frame_callbacks[0] == callback
    assert stream.fps == 30
    assert stream.frame_delay == 1 / 30
    assert mock_video_client.init_called


def test_video_stream_processing(mock_video_client):
    callback = Mock()
    stream = UnitreeCameraVideoStream(frame_callback=callback, fps=30)

    stream.start()
    time.sleep(1)
    stream.stop()

    assert callback.call_count > 0

    call_args = callback.call_args_list[0][0][0]
    assert isinstance(call_args, str)
    assert len(call_args) > 0

    try:
        decoded = base64.b64decode(call_args)
        assert len(decoded) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 string: {e}")


def test_video_stream_resize(mock_video_client):
    callback = Mock()
    stream = UnitreeCameraVideoStream(frame_callback=callback, fps=30)

    test_img = np.zeros((1200, 1600, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", test_img)

    stream.video_client.GetImageSample = Mock(return_value=(0, img_encoded.tobytes()))

    stream.start()
    time.sleep(0.1)
    stream.stop()

    assert callback.call_count > 0

    call_args = callback.call_args_list[0][0][0]
    img_data = base64.b64decode(call_args)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    height, width = decoded_img.shape[:2]
    assert width <= 640
    assert height <= 480


def test_vlm_provider_singleton_pattern(mock_video_client, ws_url, fps):
    provider1 = UnitreeCameraVLMProvider(ws_url, fps=fps)
    provider2 = UnitreeCameraVLMProvider(ws_url, fps=fps)

    assert provider1 is provider2
    assert provider1.ws_client is provider2.ws_client
    assert provider1.video_stream is provider2.video_stream
