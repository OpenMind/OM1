"""Tests for subprocess usage in unitree_realsense_dev_vlm_provider."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestSubprocessUsage:
    """Test cases for subprocess usage instead of os.popen."""

    @patch("providers.unitree_realsense_dev_vlm_provider.subprocess.run")
    @patch("providers.unitree_realsense_dev_vlm_provider.logger")
    @patch("providers.unitree_realsense_dev_vlm_provider.glob.glob")
    @patch("providers.unitree_realsense_dev_vlm_provider.cv2.VideoCapture")
    def test_find_rgb_device_uses_subprocess(
        self, mock_cv2, mock_glob, mock_logger, mock_subprocess_run
    ):
        """Test that _find_rgb_device uses subprocess.run instead of os.popen."""
        from providers.unitree_realsense_dev_vlm_provider import (
            UnitreeRealSenseDevVideoStream,
        )

        # Mock successful subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "format1\nformat2\n"
        mock_subprocess_run.return_value = mock_result
        mock_glob.return_value = ["/dev/video0"]

        stream = UnitreeRealSenseDevVideoStream()

        # Call the method that uses subprocess
        device = stream._find_rgb_device()

        # Verify subprocess.run was called
        assert mock_subprocess_run.called

        # Verify it was called with proper arguments
        call_args = mock_subprocess_run.call_args
        assert isinstance(call_args[0][0], list)  # Command as list
        assert "v4l2-ctl" in call_args[0][0][0]
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["timeout"] == 5

    @patch("providers.unitree_realsense_dev_vlm_provider.subprocess.run")
    @patch("providers.unitree_realsense_dev_vlm_provider.logger")
    @patch("providers.unitree_realsense_dev_vlm_provider.glob.glob")
    @patch("providers.unitree_realsense_dev_vlm_provider.cv2.VideoCapture")
    def test_subprocess_timeout_handling(
        self, mock_cv2, mock_glob, mock_logger, mock_subprocess_run
    ):
        """Test that subprocess timeout is handled correctly."""
        from providers.unitree_realsense_dev_vlm_provider import (
            UnitreeRealSenseDevVideoStream,
        )

        # Mock timeout exception
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(
            cmd=["v4l2-ctl", "--device=/dev/video0", "--list-formats"], timeout=5
        )
        mock_glob.return_value = ["/dev/video0"]

        stream = UnitreeRealSenseDevVideoStream()
        device = stream._find_rgb_device()

        # Verify timeout was logged
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "timed out" in warning_call.lower()

    @patch("providers.unitree_realsense_dev_vlm_provider.subprocess.run")
    @patch("providers.unitree_realsense_dev_vlm_provider.logger")
    @patch("providers.unitree_realsense_dev_vlm_provider.glob.glob")
    @patch("providers.unitree_realsense_dev_vlm_provider.cv2.VideoCapture")
    def test_subprocess_non_zero_exit_code(
        self, mock_cv2, mock_glob, mock_logger, mock_subprocess_run
    ):
        """Test handling of non-zero exit codes."""
        from providers.unitree_realsense_dev_vlm_provider import (
            UnitreeRealSenseDevVideoStream,
        )

        # Mock non-zero exit code
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_subprocess_run.return_value = mock_result
        mock_glob.return_value = ["/dev/video0"]

        stream = UnitreeRealSenseDevVideoStream()
        device = stream._find_rgb_device()

        # Verify warning was logged for non-zero exit code
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "non-zero exit code" in warning_call.lower()
