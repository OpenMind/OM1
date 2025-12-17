from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from inputs.plugins.vlm_coco_local import Message, VLM_COCO_Local, VLM_COCO_LocalConfig


@pytest.fixture
def mock_model():
    with patch(
        "inputs.plugins.vlm_coco_local.detection_model.fasterrcnn_mobilenet_v3_large_320_fpn"
    ) as mock:
        mock_instance = Mock()
        mock_instance.eval = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_check_webcam():
    with patch("inputs.plugins.vlm_coco_local.check_webcam", return_value=True):
        yield


@pytest.fixture
def mock_cv2_video_capture():
    with patch("inputs.plugins.vlm_coco_local.cv2.VideoCapture") as mock:
        mock_instance = Mock()
        # Simulate .read() returning a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_instance.get.side_effect = lambda x: {3: 640, 4: 480}[x]
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def vlm_coco_local(mock_model, mock_check_webcam, mock_cv2_video_capture):
    config = VLM_COCO_LocalConfig(camera_index=0)
    return VLM_COCO_Local(config=config)


@pytest.mark.asyncio
async def test_poll_returns_frame(vlm_coco_local):
    frame = await vlm_coco_local._poll()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)


@pytest.mark.asyncio
async def test_raw_to_text_none(vlm_coco_local):
    result = await vlm_coco_local._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_detection(vlm_coco_local, mock_model):
    # Setup mock model output
    dummy_detection = {
        "labels": [1, 2],
        "boxes": [torch.tensor([10, 10, 100, 100]), torch.tensor([200, 10, 300, 100])],
        "scores": [torch.tensor(0.9), torch.tensor(0.8)],
    }
    vlm_coco_local.model = Mock(return_value=[dummy_detection])
    vlm_coco_local.class_labels = ["__background__", "cat", "dog"]
    vlm_coco_local.cam_third = 213  # 640/3

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await vlm_coco_local._raw_to_text(dummy_frame)
    assert isinstance(result, Message)
    assert "You see a cat" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management(vlm_coco_local):
    dummy_message = Message(timestamp=123.456, message="test message")
    vlm_coco_local.messages = []
    vlm_coco_local.messages.append(dummy_message)
    assert len(vlm_coco_local.messages) == 1


def test_formatted_latest_buffer(vlm_coco_local):
    vlm_coco_local.messages = [Message(message="test message", timestamp=123.456)]
    result = vlm_coco_local.formatted_latest_buffer()
    assert "test message" in result
    assert vlm_coco_local.messages == []


def test_formatted_latest_buffer_empty(vlm_coco_local):
    vlm_coco_local.messages = []
    assert vlm_coco_local.formatted_latest_buffer() is None
