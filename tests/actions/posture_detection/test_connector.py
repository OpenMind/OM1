"""Unit tests for PostureReminderConnector."""
import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.posture_detection.connector.reminder import (
    PostureReminderConfig,
    PostureReminderConnector,
)
from actions.posture_detection.interface import (
    PostureDetectionInput,
    PostureSeverity,
    PostureType,
)


@pytest.fixture
def mock_config():
    """Create a mock PostureReminderConfig."""
    return PostureReminderConfig(
        elevenlabs_api_key="test_key",
        voice_id="test_voice",
        reminder_interval_minutes=30.0,
        enable_gentle_reminders=True,
    )


@pytest.fixture
def mock_providers():
    """Mock all provider dependencies."""
    with patch("actions.posture_detection.connector.reminder.IOProvider") as mock_io, \
         patch("actions.posture_detection.connector.reminder.HealthDetectionProvider") as mock_health, \
         patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider") as mock_tts, \
         patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider") as mock_conv, \
         patch("actions.posture_detection.connector.reminder.open_zenoh_session") as mock_zenoh:
        
        mock_io_instance = MagicMock()
        mock_io_instance.llm_prompt = None
        mock_io.return_value = mock_io_instance
        
        mock_health_instance = MagicMock()
        mock_health_instance.should_remind_posture.return_value = True
        mock_health_instance.record_posture = MagicMock()
        mock_health.return_value = mock_health_instance
        
        mock_tts_instance = MagicMock()
        mock_tts_instance.start = MagicMock()
        mock_tts_instance.get_pending_message_count.return_value = 0
        mock_tts_instance.create_pending_message.return_value = {"text": "test message"}
        mock_tts.return_value = mock_tts_instance
        
        mock_conv_instance = MagicMock()
        mock_conv.return_value = mock_conv_instance
        
        mock_session = MagicMock()
        mock_pub = MagicMock()
        mock_session.declare_publisher.return_value = mock_pub
        mock_session.declare_subscriber = MagicMock()
        mock_zenoh.return_value = mock_session
        
        yield {
            "io": mock_io_instance,
            "health": mock_health_instance,
            "tts": mock_tts_instance,
            "conv": mock_conv_instance,
            "session": mock_session,
            "pub": mock_pub,
        }


class TestPostureReminderConfig:
    """Test cases for PostureReminderConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = PostureReminderConfig()
        assert config.reminder_interval_minutes == 30.0
        assert config.enable_gentle_reminders is True
        assert config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert config.model_id == "eleven_flash_v2_5"

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = PostureReminderConfig(
            reminder_interval_minutes=15.0,
            enable_gentle_reminders=False,
            voice_id="custom_voice",
        )
        assert config.reminder_interval_minutes == 15.0
        assert config.enable_gentle_reminders is False
        assert config.voice_id == "custom_voice"


class TestPostureClassification:
    """Test cases for posture classification logic."""

    @pytest.mark.asyncio
    async def test_good_posture_no_reminder(self, mock_config, mock_providers):
        """Test that good posture does not trigger reminder."""
        connector = PostureReminderConnector(mock_config)
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.GOOD,
            severity=PostureSeverity.MILD,
            person_name="Alice"
        )
        
        await connector.connect(input_data)
        
        # Should record posture but not send reminder
        mock_providers["health"].record_posture.assert_called_once()
        # TTS should not be called for good posture
        assert mock_providers["tts"].create_pending_message.call_count == 0

    @pytest.mark.asyncio
    async def test_slumped_posture_triggers_reminder(self, mock_config, mock_providers):
        """Test that slumped posture triggers reminder."""
        connector = PostureReminderConnector(mock_config)
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            duration_minutes=30.0,
            person_name="Bob"
        )
        
        await connector.connect(input_data)
        
        # Should record and send reminder
        mock_providers["health"].record_posture.assert_called_once()
        mock_providers["tts"].create_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_poor_posture_types_trigger_reminder(self, mock_config, mock_providers):
        """Test that all poor posture types trigger reminders."""
        connector = PostureReminderConnector(mock_config)
        
        poor_postures = [
            PostureType.SLUMPED,
            PostureType.HUNCHED,
            PostureType.LEANING,
            PostureType.ASYMMETRIC,
            PostureType.LAYING,
        ]
        
        for posture_type in poor_postures:
            mock_providers["health"].reset_mock()
            mock_providers["tts"].reset_mock()
            
            input_data = PostureDetectionInput(
                posture_type=posture_type,
                severity=PostureSeverity.MODERATE,
                person_name="Test"
            )
            
            await connector.connect(input_data)
            
            # Should trigger reminder for all poor postures
            mock_providers["tts"].create_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_severity_affects_message_tone(self, mock_config, mock_providers):
        """Test that severity level affects reminder message."""
        connector = PostureReminderConnector(mock_config)
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.HUNCHED,
            severity=PostureSeverity.SEVERE,
            person_name="Alice"
        )
        
        await connector.connect(input_data)
        
        # Check that message was generated
        call_args = mock_providers["tts"].create_pending_message.call_args
        message = call_args[0][0]
        assert "Important" in message or "important" in message.lower()


class TestReminderIntervalTiming:
    """Test cases for reminder interval timing logic."""

    @pytest.mark.asyncio
    @patch("actions.posture_detection.connector.reminder.time.time")
    async def test_reminder_interval_enforced(self, mock_time, mock_config, mock_providers):
        """Test that reminder interval is properly enforced."""
        connector = PostureReminderConnector(mock_config)
        connector.last_reminder_times = {}
        
        # First reminder
        mock_time.return_value = 1000.0
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        await connector.connect(input_data)
        assert mock_providers["tts"].create_pending_message.call_count == 1
        assert connector.last_reminder_times.get("Alice") == 1000.0
        
        # Second reminder too soon (15 minutes later, but interval is 30)
        mock_time.return_value = 1000.0 + (15 * 60)
        mock_providers["health"].should_remind_posture.return_value = False
        
        mock_providers["tts"].reset_mock()
        await connector.connect(input_data)
        
        # Should not send reminder
        assert mock_providers["tts"].create_pending_message.call_count == 0

    @pytest.mark.asyncio
    @patch("actions.posture_detection.connector.reminder.time.time")
    async def test_reminder_after_interval_passes(self, mock_time, mock_config, mock_providers):
        """Test that reminder is sent after interval passes."""
        connector = PostureReminderConnector(mock_config)
        connector.last_reminder_times = {"Alice": 1000.0}
        
        # Reminder after interval (31 minutes later)
        mock_time.return_value = 1000.0 + (31 * 60)
        mock_providers["health"].should_remind_posture.return_value = True
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        await connector.connect(input_data)
        
        # Should send reminder
        assert mock_providers["tts"].create_pending_message.call_count == 1
        assert connector.last_reminder_times.get("Alice") == 1000.0 + (31 * 60)

    @pytest.mark.asyncio
    async def test_different_persons_have_separate_intervals(self, mock_config, mock_providers):
        """Test that different persons have separate reminder intervals."""
        connector = PostureReminderConnector(mock_config)
        connector.last_reminder_times = {}
        
        input_data1 = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        input_data2 = PostureDetectionInput(
            posture_type=PostureType.HUNCHED,
            severity=PostureSeverity.MODERATE,
            person_name="Bob"
        )
        
        await connector.connect(input_data1)
        await connector.connect(input_data2)
        
        # Both should have separate reminder times
        assert "Alice" in connector.last_reminder_times
        assert "Bob" in connector.last_reminder_times
        assert connector.last_reminder_times["Alice"] != connector.last_reminder_times["Bob"]


class TestEdgeCases:
    """Test cases for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_camera_unavailable_no_crash(self, mock_config, mock_providers):
        """Test handling when camera is unavailable."""
        connector = PostureReminderConnector(mock_config)
        
        # Simulate camera unavailable by having health provider return False
        mock_providers["health"].should_remind_posture.return_value = False
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        # Should not crash
        await connector.connect(input_data)
        
        # Should still record posture
        mock_providers["health"].record_posture.assert_called_once()

    @pytest.mark.asyncio
    async def test_vlm_timeout_handling(self, mock_config, mock_providers):
        """Test handling of VLM timeout scenarios."""
        connector = PostureReminderConnector(mock_config)
        
        # Simulate VLM timeout by making TTS fail
        mock_providers["tts"].create_pending_message.side_effect = Exception("VLM timeout")
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        # Should handle gracefully
        try:
            await connector.connect(input_data)
        except Exception:
            pytest.fail("Should handle VLM timeout gracefully")

    @pytest.mark.asyncio
    async def test_tts_disabled_skips_reminder(self, mock_config, mock_providers):
        """Test that reminders are skipped when TTS is disabled."""
        connector = PostureReminderConnector(mock_config)
        connector.tts_enabled = False
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        await connector.connect(input_data)
        
        # Should record but not send reminder
        mock_providers["health"].record_posture.assert_called_once()
        assert mock_providers["tts"].create_pending_message.call_count == 0

    @pytest.mark.asyncio
    async def test_too_many_pending_messages_skips_reminder(self, mock_config, mock_providers):
        """Test that reminder is skipped when too many TTS messages are pending."""
        connector = PostureReminderConnector(mock_config)
        mock_providers["tts"].get_pending_message_count.return_value = 5
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        await connector.connect(input_data)
        
        # Should not create new message
        assert mock_providers["tts"].create_pending_message.call_count == 0

    @pytest.mark.asyncio
    async def test_unknown_person_handling(self, mock_config, mock_providers):
        """Test handling of unknown person (empty name)."""
        connector = PostureReminderConnector(mock_config)
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name=""  # Unknown person
        )
        
        await connector.connect(input_data)
        
        # Should use "unknown" as key
        assert "unknown" in connector.last_reminder_times
        mock_providers["tts"].create_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_zenoh_unavailable_fallback(self, mock_config, mock_providers):
        """Test fallback when Zenoh is unavailable."""
        connector = PostureReminderConnector(mock_config)
        connector.audio_pub = None  # Simulate Zenoh unavailable
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            person_name="Alice"
        )
        
        await connector.connect(input_data)
        
        # Should use TTS directly as fallback
        mock_providers["tts"].add_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_gentle_vs_direct_reminders(self, mock_config, mock_providers):
        """Test that gentle and direct reminder modes generate different messages."""
        # Test gentle reminders
        gentle_config = PostureReminderConfig(enable_gentle_reminders=True)
        connector_gentle = PostureReminderConnector(gentle_config)
        
        input_data = PostureDetectionInput(
            posture_type=PostureType.SLUMPED,
            severity=PostureSeverity.MODERATE,
            duration_minutes=30.0,
            person_name="Alice"
        )
        
        message_gentle = connector_gentle._generate_reminder_message(input_data)
        
        # Test direct reminders
        direct_config = PostureReminderConfig(enable_gentle_reminders=False)
        connector_direct = PostureReminderConnector(direct_config)
        
        message_direct = connector_direct._generate_reminder_message(input_data)
        
        # Messages should be different
        assert message_gentle != message_direct
        # Gentle should be more encouraging
        assert "alert" not in message_gentle.lower() or "Posture alert" not in message_gentle
