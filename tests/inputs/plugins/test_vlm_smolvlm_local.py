from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from inputs.base import Message
from inputs.plugins.vlm_smolvlm_local import (
    VLM_SmolVLM_Local,
    VLM_SmolVLM_LocalConfig,
)

cv2_CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
cv2_CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT


@pytest.fixture
def mock_transformers():
    with (
        patch("inputs.plugins.vlm_smolvlm_local.HAS_TRANSFORMERS", True),
        patch(
            "inputs.plugins.vlm_smolvlm_local.SmolVLMForConditionalGeneration"
        ) as mock_model_cls,
        patch(
            "inputs.plugins.vlm_smolvlm_local.SmolVLMProcessor"
        ) as mock_processor_cls,
    ):
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        yield mock_model, mock_processor


@pytest.fixture
def mock_check_webcam():
    with patch("inputs.plugins.vlm_smolvlm_local.check_webcam", return_value=True):
        yield


@pytest.fixture
def mock_cv2_video_capture():
    with patch("inputs.plugins.vlm_smolvlm_local.cv2.VideoCapture") as mock:
        mock_instance = MagicMock()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, dummy_frame)
        mock_instance.get.side_effect = lambda x: {
            cv2_CAP_PROP_FRAME_WIDTH: 640,
            cv2_CAP_PROP_FRAME_HEIGHT: 480,
        }.get(x, 0)
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sensor(mock_transformers, mock_check_webcam, mock_cv2_video_capture):
    with (
        patch("inputs.plugins.vlm_smolvlm_local.IOProvider"),
        patch(
            "inputs.plugins.vlm_smolvlm_local.torch.cuda.is_available",
            return_value=False,
        ),
    ):
        config = VLM_SmolVLM_LocalConfig(camera_index=0)
        return VLM_SmolVLM_Local(config=config)


def test_initialization(sensor):
    """Test basic initialization."""
    assert hasattr(sensor, "messages")
    assert hasattr(sensor, "have_cam")
    assert hasattr(sensor, "descriptor_for_LLM")
    assert sensor.descriptor_for_LLM == "Vision"
    assert sensor.have_cam is True


def test_initialization_without_transformers():
    """Test graceful degradation when transformers is not installed."""
    with (
        patch("inputs.plugins.vlm_smolvlm_local.HAS_TRANSFORMERS", False),
        patch("inputs.plugins.vlm_smolvlm_local.IOProvider"),
    ):
        config = VLM_SmolVLM_LocalConfig(camera_index=0)
        s = VLM_SmolVLM_Local(config=config)
        assert s.have_cam is False
        assert s.model is None
        assert s.processor is None


@pytest.mark.asyncio
async def test_poll_returns_frame(sensor, mock_cv2_video_capture):
    """Test _poll returns a valid numpy frame."""
    with patch("inputs.plugins.vlm_smolvlm_local.asyncio.sleep", new=AsyncMock()):
        frame = await sensor._poll()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)


@pytest.mark.asyncio
async def test_poll_returns_none_on_failed_frame_read(
    mock_transformers, mock_check_webcam
):
    """Test _poll returns None when cap.read() fails."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    with (
        patch("inputs.plugins.vlm_smolvlm_local.IOProvider"),
        patch(
            "inputs.plugins.vlm_smolvlm_local.torch.cuda.is_available",
            return_value=False,
        ),
        patch(
            "inputs.plugins.vlm_smolvlm_local.cv2.VideoCapture",
            return_value=mock_cap,
        ),
        patch("inputs.plugins.vlm_smolvlm_local.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_SmolVLM_LocalConfig(camera_index=0)
        s = VLM_SmolVLM_Local(config=config)
        result = await s._poll()
    assert result is None


@pytest.mark.asyncio
async def test_poll_returns_none_without_camera():
    """Test _poll returns None when no camera is available."""
    with (
        patch("inputs.plugins.vlm_smolvlm_local.HAS_TRANSFORMERS", True),
        patch("inputs.plugins.vlm_smolvlm_local.SmolVLMForConditionalGeneration"),
        patch("inputs.plugins.vlm_smolvlm_local.SmolVLMProcessor"),
        patch("inputs.plugins.vlm_smolvlm_local.IOProvider"),
        patch(
            "inputs.plugins.vlm_smolvlm_local.torch.cuda.is_available",
            return_value=False,
        ),
        patch("inputs.plugins.vlm_smolvlm_local.check_webcam", return_value=False),
        patch("inputs.plugins.vlm_smolvlm_local.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_SmolVLM_LocalConfig(camera_index=0)
        s = VLM_SmolVLM_Local(config=config)
        result = await s._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_none(sensor):
    """Test _raw_to_text returns None for None input."""
    result = await sensor._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_frame(sensor, mock_transformers):
    """Test _raw_to_text returns a Message when given a valid frame."""
    mock_model, mock_processor = mock_transformers

    mock_output = MagicMock()
    mock_output.__getitem__ = MagicMock(return_value=MagicMock())
    mock_model.generate.return_value = mock_output

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.apply_chat_template.return_value = "chat_text"
    mock_processor.decode.return_value = "A chair on a wooden floor."

    sensor.processor = mock_processor
    sensor.model = mock_model

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("inputs.plugins.vlm_smolvlm_local.torch") as mock_torch:
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.float16 = MagicMock()
        mock_torch.float32 = MagicMock()
        result = await sensor._raw_to_text(dummy_frame)

    assert isinstance(result, Message)
    assert "chair" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_no_model(sensor):
    """Test _raw_to_text returns None when model is not loaded."""
    sensor.model = None
    sensor.processor = None
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await sensor._raw_to_text(dummy_frame)
    assert result is None


def test_formatted_latest_buffer_empty(sensor):
    """Test formatted_latest_buffer returns None when buffer is empty."""
    sensor.messages = []
    assert sensor.formatted_latest_buffer() is None


def test_formatted_latest_buffer(sensor):
    """Test formatted_latest_buffer returns formatted string and clears buffer."""
    sensor.messages = [
        Message(timestamp=123.456, message="A person is sitting on a chair.")
    ]
    result = sensor.formatted_latest_buffer()
    assert isinstance(result, str)
    assert "INPUT:" in result
    assert "Vision" in result
    assert "A person is sitting on a chair." in result
    assert "// START" in result
    assert "// END" in result
    assert len(sensor.messages) == 0


@pytest.mark.asyncio
async def test_raw_to_text_appends_message(sensor, mock_transformers):
    """Test raw_to_text appends message to buffer when valid frame given."""
    mock_model, mock_processor = mock_transformers

    mock_output = MagicMock()
    mock_output.__getitem__ = MagicMock(return_value=MagicMock())
    mock_model.generate.return_value = mock_output

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.apply_chat_template.return_value = "chat_text"
    mock_processor.decode.return_value = "A person is standing."

    sensor.processor = mock_processor
    sensor.model = mock_model

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("inputs.plugins.vlm_smolvlm_local.torch") as mock_torch:
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.float16 = MagicMock()
        mock_torch.float32 = MagicMock()
        await sensor.raw_to_text(dummy_frame)

    assert len(sensor.messages) == 1
    assert "person" in sensor.messages[0].message


def test_check_webcam_not_found():
    """Test check_webcam returns False when camera not found."""
    from inputs.plugins.vlm_smolvlm_local import check_webcam

    with patch("inputs.plugins.vlm_smolvlm_local.cv2.VideoCapture") as mock_cap:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance

        result = check_webcam(0)
        assert result is False
        mock_instance.release.assert_called_once()


def test_check_webcam_found():
    """Test check_webcam returns True when camera found."""
    from inputs.plugins.vlm_smolvlm_local import check_webcam

    with patch("inputs.plugins.vlm_smolvlm_local.cv2.VideoCapture") as mock_cap:
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_cap.return_value = mock_instance

        result = check_webcam(0)
        assert result is True
        mock_instance.release.assert_called_once()


def test_initialization_model_load_failure():
    """Test graceful degradation when model fails to load."""
    with (
        patch("inputs.plugins.vlm_smolvlm_local.HAS_TRANSFORMERS", True),
        patch(
            "inputs.plugins.vlm_smolvlm_local.SmolVLMProcessor"
        ) as mock_processor_cls,
        patch("inputs.plugins.vlm_smolvlm_local.SmolVLMForConditionalGeneration"),
        patch("inputs.plugins.vlm_smolvlm_local.IOProvider"),
        patch(
            "inputs.plugins.vlm_smolvlm_local.torch.cuda.is_available",
            return_value=False,
        ),
    ):
        mock_processor_cls.from_pretrained.side_effect = RuntimeError("load failed")
        config = VLM_SmolVLM_LocalConfig(camera_index=0)
        s = VLM_SmolVLM_Local(config=config)
        assert s.have_cam is False
        assert s.cap is None


@pytest.mark.asyncio
async def test_raw_to_text_empty_response(sensor, mock_transformers):
    """Test _raw_to_text returns None when model returns empty response."""
    mock_model, mock_processor = mock_transformers

    mock_output = MagicMock()
    mock_output.__getitem__ = MagicMock(return_value=MagicMock())
    mock_model.generate.return_value = mock_output

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.apply_chat_template.return_value = "chat_text"
    mock_processor.decode.return_value = ""

    sensor.processor = mock_processor
    sensor.model = mock_model

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("inputs.plugins.vlm_smolvlm_local.torch") as mock_torch:
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.float16 = MagicMock()
        mock_torch.float32 = MagicMock()
        result = await sensor._raw_to_text(dummy_frame)

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_inference_exception(sensor, mock_transformers):
    """Test _raw_to_text returns None when inference raises exception."""
    mock_model, mock_processor = mock_transformers

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.apply_chat_template.return_value = "chat_text"
    mock_model.generate.side_effect = RuntimeError("inference failed")

    sensor.processor = mock_processor
    sensor.model = mock_model

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("inputs.plugins.vlm_smolvlm_local.torch") as mock_torch:
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.float16 = MagicMock()
        mock_torch.float32 = MagicMock()
        result = await sensor._raw_to_text(dummy_frame)

    assert result is None
