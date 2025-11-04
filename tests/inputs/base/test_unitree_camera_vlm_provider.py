import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from providers.unitree_camera_vlm_provider import (
    UnitreeCameraVideoStream,
    UnitreeCameraVLMProvider,
)


@pytest.fixture
def mock_video_client():
    with patch("providers.unitree_camera_vlm_provider.VideoClient") as mock:
        mock_instance = Mock()
        mock_instance.Init.return_value = None
        mock_instance.GetImageSample.return_value = (
            0,
            np.zeros((480, 640, 3), dtype=np.uint8).tobytes(),
        )
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_ws_client():
    with patch("om1_utils.ws.Client") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


def test_video_stream_init(mock_video_client):
    callback = Mock()
    stream = UnitreeCameraVideoStream(frame_callback=callback)

    assert len(stream.frame_callbacks) == 1
    assert stream.frame_callbacks[0] == callback
    assert stream.running is True
    mock_video_client.Init.assert_called_once()


def test_video_stream_start_stop(mock_video_client):
    stream = UnitreeCameraVideoStream()
    stream.start()
    assert stream.running is True

    time.sleep(0.1)

    stream.stop()
    assert stream.running is False


def test_vlm_provider_init(mock_ws_client, mock_video_client):
    provider = UnitreeCameraVLMProvider("ws://test.url")
    assert provider.running is False


def test_vlm_provider_start_stop(mock_ws_client, mock_video_client):
    provider = UnitreeCameraVLMProvider("ws://test.url")
    provider.start()
    assert provider.running is True

    provider.stop()
    assert provider.running is False
