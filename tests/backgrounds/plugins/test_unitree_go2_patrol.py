from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backgrounds.plugins.unitree_go2_patrol import (
    UnitreeGo2Patrol,
    UnitreeGo2PatrolConfig,
)


class TestUnitreeGo2PatrolConfig:
    """Tests for UnitreeGo2PatrolConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeGo2PatrolConfig()

        assert config.patrol_base_url == "http://localhost:5000"
        assert config.face_presence_base_url == "http://localhost:6793"
        assert config.patrol_image_report_base_url == "https://api.openmind.com"
        assert config.api_key == ""
        assert config.unknown_capture_threshold == 2
        assert config.upload_cooldown_seconds == 5
        assert config.force_resume_seconds == 15
        assert config.safe_force_resume_seconds == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnitreeGo2PatrolConfig(
            patrol_base_url="http://robot.local:8000",
            face_presence_base_url="http://robot.local:9000",
            patrol_image_report_base_url="https://custom.api.com",
            api_key="test_key",
            unknown_capture_threshold=5,
            upload_cooldown_seconds=10,
            force_resume_seconds=30,
            safe_force_resume_seconds=10,
        )

        assert config.patrol_base_url == "http://robot.local:8000"
        assert config.face_presence_base_url == "http://robot.local:9000"
        assert config.patrol_image_report_base_url == "https://custom.api.com"
        assert config.api_key == "test_key"
        assert config.unknown_capture_threshold == 5
        assert config.upload_cooldown_seconds == 10
        assert config.force_resume_seconds == 30
        assert config.safe_force_resume_seconds == 10


class TestUnitreeGo2PatrolInitialization:
    """Tests for UnitreeGo2Patrol initialization."""

    @patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider")
    @patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider")
    def test_initialization_default_config(self, mock_tts_provider, mock_patrol_provider):
        """Test initialization with default configuration."""
        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        assert background.config is config
        assert background.uploaded_track_ids == set()
        assert background.is_paused is False
        assert background.last_pause_time == 0
        assert background.last_force_resume_time == 0

        mock_patrol_provider.assert_called_once_with(
            api_key="",
            patrol_base_url="http://localhost:5000",
            face_presence_base_url="http://localhost:6793",
            patrol_image_report_base_url="https://api.openmind.com",
        )
        mock_tts_provider.assert_called_once()

    @patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider")
    @patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider")
    def test_initialization_custom_config(self, mock_tts_provider, mock_patrol_provider):
        """Test initialization with custom configuration."""
        config = UnitreeGo2PatrolConfig(
            patrol_base_url="http://robot.local:8000",
            api_key="test_key",
        )
        UnitreeGo2Patrol(config)

        mock_patrol_provider.assert_called_once_with(
            api_key="test_key",
            patrol_base_url="http://robot.local:8000",
            face_presence_base_url="http://localhost:6793",
            patrol_image_report_base_url="https://api.openmind.com",
        )

    @patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider")
    @patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider")
    def test_initialization_logs_message(self, mock_tts_provider, mock_patrol_provider, caplog):
        """Test that initialization logs the correct message."""
        config = UnitreeGo2PatrolConfig()

        with caplog.at_level("INFO"):
            UnitreeGo2Patrol(config)

        assert "Initialized Unitree Go2 Patrol Background Task" in caplog.text


class TestUnitreeGo2PatrolMethods:
    """Tests for UnitreeGo2Patrol methods."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers."""
        with (
            patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider") as mock_patrol,
            patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider") as mock_tts,
        ):
            yield mock_patrol, mock_tts

    @pytest.mark.asyncio
    async def test_start_patrol(self, mock_providers):
        """Test start_patrol method."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.start_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        await background.start_patrol()

        mock_patrol_instance.start_patrol.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_patrol(self, mock_providers):
        """Test stop_patrol method."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.stop_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        await background.stop_patrol()

        mock_patrol_instance.stop_patrol.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_patrol(self, mock_providers):
        """Test pause_patrol method."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.pause_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        await background.pause_patrol()

        mock_patrol_instance.pause_patrol.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_patrol(self, mock_providers):
        """Test resume_patrol method."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.resume_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        await background.resume_patrol()

        mock_patrol_instance.resume_patrol.assert_called_once()


class TestUnitreeGo2PatrolRun:
    """Tests for UnitreeGo2Patrol run method."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers."""
        with (
            patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider") as mock_patrol,
            patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider") as mock_tts,
        ):
            yield mock_patrol, mock_tts

    def test_run_no_unknown_captures(self, mock_providers):
        """Test run method with no unknown captures."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(return_value={"frame_b64": "", "unknown_captures": []})
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        with patch.object(background, "sleep"):
            background.run()

        assert background.is_paused is False
        assert len(background.uploaded_track_ids) == 0

    def test_run_new_unknown_captures_below_threshold(self, mock_providers):
        """Test run method with new unknown captures below threshold."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_base64",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 1}],
            }
        )
        mock_patrol_instance.pause_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig(unknown_capture_threshold=2)
        background = UnitreeGo2Patrol(config)

        with patch.object(background, "sleep"):
            background.run()

        # Should pause but not upload since below threshold
        assert background.is_paused is True
        assert len(background.uploaded_track_ids) == 0
        mock_patrol_instance.pause_patrol.assert_called_once()

    def test_run_unknown_captures_above_threshold(self, mock_providers):
        """Test run method with unknown captures above threshold."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_base64",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 5}],
            }
        )
        mock_patrol_instance.pause_patrol = AsyncMock()
        mock_patrol_instance.upload_patrol_image = AsyncMock()
        mock_patrol_instance.resume_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance

        config = UnitreeGo2PatrolConfig(unknown_capture_threshold=2)
        background = UnitreeGo2Patrol(config)

        with patch.object(background, "sleep"):
            background.run()

        assert background.is_paused is False
        assert 1 in background.uploaded_track_ids
        mock_patrol_instance.pause_patrol.assert_called_once()
        mock_patrol_instance.upload_patrol_image.assert_called_once()
        mock_patrol_instance.resume_patrol.assert_called_once()
        mock_tts_instance.add_pending_message.assert_called_once()

    def test_run_already_uploaded_track_id(self, mock_providers):
        """Test run method with already uploaded track_id."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_base64",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 5}],
            }
        )
        mock_patrol_instance.pause_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)
        background.uploaded_track_ids.add(1)

        with patch.object(background, "sleep"):
            background.run()

        # Should not pause because track_id already uploaded
        assert background.is_paused is False
        mock_patrol_instance.pause_patrol.assert_not_called()

    def test_run_force_resume_after_timeout(self, mock_providers):
        """Test run method with force resume after timeout."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_base64",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 1}],
            }
        )
        mock_patrol_instance.resume_patrol = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig(force_resume_seconds=1)
        background = UnitreeGo2Patrol(config)
        background.is_paused = True
        background.last_pause_time = 0  # Force timeout

        with patch.object(background, "sleep"), patch("time.time", return_value=100):
            background.run()

        assert background.is_paused is False
        mock_patrol_instance.resume_patrol.assert_called_once()

    def test_run_get_report_error(self, mock_providers):
        """Test run method with get_report error."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(side_effect=Exception("Network error"))
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        with patch.object(background, "sleep"):
            # Should not raise exception
            background.run()

    def test_run_pause_error(self, mock_providers, caplog):
        """Test run method with pause error."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_base64",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 5}],
            }
        )
        mock_patrol_instance.pause_patrol = AsyncMock(side_effect=Exception("Pause failed"))
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        with caplog.at_level("ERROR"):
            with patch.object(background, "sleep"):
                background.run()

        assert "Failed to pause patrol" in caplog.text

    def test_run_upload_error(self, mock_providers, caplog):
        """Test run method with upload error."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_base64",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 5}],
            }
        )
        mock_patrol_instance.upload_patrol_image = AsyncMock(side_effect=Exception("Upload failed"))
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig(unknown_capture_threshold=2)
        background = UnitreeGo2Patrol(config)
        background.is_paused = True

        with caplog.at_level("ERROR"):
            with patch.object(background, "sleep"):
                background.run()

        assert "Failed to upload patrol data" in caplog.text

    def test_run_resume_error(self, mock_providers, caplog):
        """Test run method with resume error."""
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_base64",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 5}],
            }
        )
        mock_patrol_instance.upload_patrol_image = AsyncMock()
        mock_patrol_instance.resume_patrol = AsyncMock(side_effect=Exception("Resume failed"))
        mock_patrol.return_value = mock_patrol_instance

        config = UnitreeGo2PatrolConfig(unknown_capture_threshold=2)
        background = UnitreeGo2Patrol(config)
        background.is_paused = True

        with caplog.at_level("ERROR"):
            with patch.object(background, "sleep"):
                background.run()

        assert "Failed to resume patrol" in caplog.text

    def test_run_back_and_forth_unknown_detection_cycles(self, mock_providers):
        """Test run method with back-and-forth unknown face detection cycles.

        Simulates:
        1. Unknown face detected → patrol pauses
        2. Unknown face duration exceeds threshold → upload and resume patrol
        3. No unknown faces detected → patrol continues running
        4. Unknown face detected again → patrol pauses again
        5. Unknown face duration exceeds threshold → upload and resume again
        """
        mock_patrol, mock_tts = mock_providers
        mock_patrol_instance = MagicMock()
        mock_patrol_instance.pause_patrol = AsyncMock()
        mock_patrol_instance.resume_patrol = AsyncMock()
        mock_patrol_instance.upload_patrol_image = AsyncMock()
        mock_patrol.return_value = mock_patrol_instance

        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance

        config = UnitreeGo2PatrolConfig(unknown_capture_threshold=2)
        background = UnitreeGo2Patrol(config)

        # Cycle 1: Detect unknown face (below threshold)
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_1",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 1}],
            }
        )
        with patch.object(background, "sleep"):
            background.run()

        assert background.is_paused is True
        assert mock_patrol_instance.pause_patrol.call_count == 1
        assert mock_patrol_instance.upload_patrol_image.call_count == 0
        assert mock_patrol_instance.resume_patrol.call_count == 0

        # Cycle 2: Unknown face exceeds threshold → upload and resume
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_2",
                "unknown_captures": [{"track_id": 1, "unknown_duration": 5}],
            }
        )
        with patch.object(background, "sleep"):
            background.run()

        assert background.is_paused is False
        assert 1 in background.uploaded_track_ids
        assert mock_patrol_instance.upload_patrol_image.call_count == 1
        assert mock_patrol_instance.resume_patrol.call_count == 1
        assert mock_tts_instance.add_pending_message.call_count == 1

        # Cycle 3: No unknown faces → patrol continues
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_3",
                "unknown_captures": [],
            }
        )
        with patch.object(background, "sleep"):
            background.run()

        assert background.is_paused is False
        assert mock_patrol_instance.pause_patrol.call_count == 1
        assert mock_patrol_instance.upload_patrol_image.call_count == 1
        assert mock_patrol_instance.resume_patrol.call_count == 1

        # Cycle 4: New unknown face detected → pause again
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_4",
                "unknown_captures": [{"track_id": 2, "unknown_duration": 1}],
            }
        )
        with patch.object(background, "sleep"), patch("time.time", return_value=100):
            background.run()

        assert background.is_paused is True
        assert mock_patrol_instance.pause_patrol.call_count == 2

        # Cycle 5: Second unknown face exceeds threshold → upload and resume again
        mock_patrol_instance.get_report = AsyncMock(
            return_value={
                "frame_b64": "test_image_5",
                "unknown_captures": [{"track_id": 2, "unknown_duration": 5}],
            }
        )
        with patch.object(background, "sleep"):
            background.run()

        assert background.is_paused is False
        assert 2 in background.uploaded_track_ids
        assert mock_patrol_instance.upload_patrol_image.call_count == 2
        assert mock_patrol_instance.resume_patrol.call_count == 2
        assert mock_tts_instance.add_pending_message.call_count == 2


class TestUnitreeGo2PatrolStop:
    """Tests for UnitreeGo2Patrol stop method."""

    @patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider")
    @patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider")
    def test_stop_closes_loop(self, mock_tts_provider, mock_patrol_provider):
        """Test that stop method closes the event loop."""
        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        assert background.loop is not None
        assert not background.loop.is_closed()

        background.stop()

        assert background.loop.is_closed()

    @patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider")
    @patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider")
    def test_stop_logs_message(self, mock_tts_provider, mock_patrol_provider, caplog):
        """Test that stop method logs the correct message."""
        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        with caplog.at_level("INFO"):
            background.stop()

        assert "Stopping Unitree Go2 Patrol Background Task" in caplog.text

    @patch("backgrounds.plugins.unitree_go2_patrol.UnitreeGo2PatrolProvider")
    @patch("backgrounds.plugins.unitree_go2_patrol.ElevenLabsTTSProvider")
    def test_stop_with_already_closed_loop(self, mock_tts_provider, mock_patrol_provider):
        """Test stop method when loop is already closed."""
        config = UnitreeGo2PatrolConfig()
        background = UnitreeGo2Patrol(config)

        background.loop.close()
        background.stop()
