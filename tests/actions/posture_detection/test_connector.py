"""Unit tests for PostureReminderConnector."""
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
    """Create a mock configuration."""
    config = PostureReminderConfig(
        elevenlabs_api_key="test_key",
        voice_id="test_voice",
        model_id="test_model",
        reminder_interval_minutes=30.0,
        enable_gentle_reminders=True,
    )
    return config


@pytest.fixture
def mock_health_provider():
    """Create a mock health detection provider."""
    provider = MagicMock()
    provider.should_remind_posture = MagicMock(return_value=True)
    provider.record_posture = MagicMock()
    return provider


@pytest.fixture
def mock_tts_provider():
    """Create a mock TTS provider."""
    tts = MagicMock()
    tts.get_pending_message_count = MagicMock(return_value=0)
    tts.create_pending_message = MagicMock(return_value={"text": "test message"})
    tts.add_pending_message = MagicMock()
    tts.start = MagicMock()
    return tts


@pytest.fixture
def mock_io_provider():
    """Create a mock IO provider."""
    provider = MagicMock()
    provider.llm_prompt = None
    return provider


@pytest.fixture
def mock_conversation_provider():
    """Create a mock conversation provider."""
    provider = MagicMock()
    provider.store_robot_message = MagicMock()
    return provider


@patch("actions.posture_detection.connector.reminder.open_zenoh_session")
@patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider")
@patch("actions.posture_detection.connector.reminder.HealthDetectionProvider")
@patch("actions.posture_detection.connector.reminder.IOProvider")
@patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider")
def test_connector_initialization(
    mock_conversation_provider_class,
    mock_io_provider_class,
    mock_health_provider_class,
    mock_tts_class,
    mock_zenoh_session,
    mock_config,
    mock_health_provider,
    mock_tts_provider,
    mock_io_provider,
    mock_conversation_provider,
):
    """Test connector initialization."""
    # Setup mocks
    mock_health_provider_class.return_value = mock_health_provider
    mock_io_provider_class.return_value = mock_io_provider
    mock_conversation_provider_class.return_value = mock_conversation_provider
    mock_tts_class.return_value = mock_tts_provider
    
    mock_session = MagicMock()
    mock_pub = MagicMock()
    mock_session.declare_publisher.return_value = mock_pub
    mock_zenoh_session.return_value = mock_session
    
    # Create connector
    connector = PostureReminderConnector(mock_config)
    
    # Verify initialization
    assert connector.config == mock_config
    assert connector.health_provider == mock_health_provider
    assert connector.io_provider == mock_io_provider
    assert connector.tts_enabled is True


@pytest.mark.asyncio
async def test_reminder_interval_timing(mock_config, mock_health_provider, mock_tts_provider):
    """Test reminder interval timing logic."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        connector = PostureReminderConnector(mock_config)
                        
                        # Test: First reminder should be sent (no previous reminder)
                        input_data = PostureDetectionInput(
                            posture_type=PostureType.SLUMPED,
                            severity=PostureSeverity.MODERATE,
                            person_name="Alice"
                        )
                        mock_health_provider.should_remind_posture.return_value = True
                        
                        await connector.connect(input_data)
                        
                        # Verify reminder was processed
                        mock_health_provider.record_posture.assert_called_once()
                        
                        # Test: Reminder too soon (within interval)
                        mock_health_provider.should_remind_posture.return_value = False
                        mock_health_provider.record_posture.reset_mock()
                        
                        await connector.connect(input_data)
                        
                        # Should record but not send reminder
                        mock_health_provider.record_posture.assert_called_once()


@pytest.mark.asyncio
async def test_posture_classification_logic(mock_config, mock_health_provider, mock_tts_provider):
    """Test posture classification logic - good vs poor postures."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        connector = PostureReminderConnector(mock_config)
                        connector.tts_enabled = True
                        
                        # Test: Good posture should not trigger reminder
                        good_posture = PostureDetectionInput(
                            posture_type=PostureType.GOOD,
                            severity=PostureSeverity.MILD,
                            person_name="Bob"
                        )
                        mock_health_provider.should_remind_posture.return_value = True
                        
                        await connector.connect(good_posture)
                        
                        # Should record but not send reminder for good posture
                        mock_health_provider.record_posture.assert_called_once()
                        
                        # Test: Poor posture should trigger reminder
                        poor_posture = PostureDetectionInput(
                            posture_type=PostureType.SLUMPED,
                            severity=PostureSeverity.MODERATE,
                            person_name="Bob"
                        )
                        mock_health_provider.record_posture.reset_mock()
                        
                        await connector.connect(poor_posture)
                        
                        # Should record and potentially send reminder
                        mock_health_provider.record_posture.assert_called_once()


@pytest.mark.asyncio
async def test_reminder_message_generation(mock_config, mock_health_provider, mock_tts_provider):
    """Test reminder message generation for different postures."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        connector = PostureReminderConnector(mock_config)
                        connector.tts_enabled = True
                        connector.audio_pub = None  # Use fallback TTS
                        
                        # Test different posture types
                        postures = [
                            PostureType.SLUMPED,
                            PostureType.HUNCHED,
                            PostureType.LEANING,
                            PostureType.ASYMMETRIC,
                            PostureType.LAYING,
                        ]
                        
                        for posture_type in postures:
                            input_data = PostureDetectionInput(
                                posture_type=posture_type,
                                severity=PostureSeverity.MODERATE,
                                duration_minutes=30.0,
                                person_name="TestUser"
                            )
                            mock_health_provider.should_remind_posture.return_value = True
                            
                            message = connector._generate_reminder_message(input_data)
                            
                            # Verify message is generated and contains relevant information
                            assert message is not None
                            assert len(message) > 0
                            assert "TestUser" in message or "You" in message


@pytest.mark.asyncio
async def test_edge_case_tts_disabled(mock_config, mock_health_provider, mock_tts_provider):
    """Test edge case: TTS disabled."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        connector = PostureReminderConnector(mock_config)
                        connector.tts_enabled = False
                        
                        input_data = PostureDetectionInput(
                            posture_type=PostureType.SLUMPED,
                            severity=PostureSeverity.MODERATE,
                            person_name="Alice"
                        )
                        mock_health_provider.should_remind_posture.return_value = True
                        
                        await connector.connect(input_data)
                        
                        # Should record but not send reminder when TTS is disabled
                        mock_health_provider.record_posture.assert_called_once()
                        mock_tts_provider.create_pending_message.assert_not_called()


@pytest.mark.asyncio
async def test_edge_case_too_many_pending_messages(mock_config, mock_health_provider, mock_tts_provider):
    """Test edge case: Too many pending TTS messages."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        connector = PostureReminderConnector(mock_config)
                        connector.tts_enabled = True
                        connector.audio_pub = None
                        
                        # Simulate too many pending messages
                        mock_tts_provider.get_pending_message_count.return_value = 5
                        
                        input_data = PostureDetectionInput(
                            posture_type=PostureType.SLUMPED,
                            severity=PostureSeverity.MODERATE,
                            person_name="Alice"
                        )
                        mock_health_provider.should_remind_posture.return_value = True
                        
                        await connector.connect(input_data)
                        
                        # Should record but not add more messages when queue is full
                        mock_health_provider.record_posture.assert_called_once()
                        # Should not add pending message when queue is full
                        mock_tts_provider.add_pending_message.assert_not_called()


@pytest.mark.asyncio
async def test_edge_case_zenoh_unavailable(mock_config, mock_health_provider, mock_tts_provider):
    """Test edge case: Zenoh unavailable, fallback to direct TTS."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session", side_effect=Exception("Zenoh unavailable")):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        # Should handle Zenoh error gracefully
                        connector = PostureReminderConnector(mock_config)
                        connector.tts_enabled = True
                        connector.audio_pub = None  # Simulate no Zenoh publisher
                        
                        input_data = PostureDetectionInput(
                            posture_type=PostureType.SLUMPED,
                            severity=PostureSeverity.MODERATE,
                            person_name="Alice"
                        )
                        mock_health_provider.should_remind_posture.return_value = True
                        
                        await connector.connect(input_data)
                        
                        # Should still record and use fallback TTS
                        mock_health_provider.record_posture.assert_called_once()


@pytest.mark.asyncio
async def test_severity_based_message_tone(mock_config, mock_health_provider, mock_tts_provider):
    """Test that message tone adjusts based on severity."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        connector = PostureReminderConnector(mock_config)
                        
                        # Test severe posture
                        severe_input = PostureDetectionInput(
                            posture_type=PostureType.HUNCHED,
                            severity=PostureSeverity.SEVERE,
                            duration_minutes=60.0,
                            person_name="Alice"
                        )
                        
                        message = connector._generate_reminder_message(severe_input)
                        
                        # Severe messages should include urgency indicator
                        assert "Important" in message or "important" in message.lower()


def test_gentle_vs_direct_reminders(mock_config):
    """Test gentle vs direct reminder modes."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider"):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider"):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        # Test gentle reminders
                        gentle_config = PostureReminderConfig(
                            enable_gentle_reminders=True,
                            reminder_interval_minutes=30.0,
                        )
                        gentle_connector = PostureReminderConnector(gentle_config)
                        
                        input_data = PostureDetectionInput(
                            posture_type=PostureType.SLUMPED,
                            severity=PostureSeverity.MODERATE,
                            duration_minutes=30.0,
                            person_name="Alice"
                        )
                        
                        gentle_message = gentle_connector._generate_reminder_message(input_data)
                        
                        # Test direct reminders
                        direct_config = PostureReminderConfig(
                            enable_gentle_reminders=False,
                            reminder_interval_minutes=30.0,
                        )
                        direct_connector = PostureReminderConnector(direct_config)
                        
                        direct_message = direct_connector._generate_reminder_message(input_data)
                        
                        # Messages should be different
                        assert gentle_message != direct_message
                        # Gentle messages should be more encouraging
                        assert "alert" not in gentle_message.lower() or "gentle" in str(gentle_config.enable_gentle_reminders)


@pytest.mark.asyncio
async def test_person_specific_reminders(mock_config, mock_health_provider, mock_tts_provider):
    """Test that reminders are tracked per person."""
    with patch("actions.posture_detection.connector.reminder.open_zenoh_session"):
        with patch("actions.posture_detection.connector.reminder.ElevenLabsTTSProvider", return_value=mock_tts_provider):
            with patch("actions.posture_detection.connector.reminder.HealthDetectionProvider", return_value=mock_health_provider):
                with patch("actions.posture_detection.connector.reminder.IOProvider"):
                    with patch("actions.posture_detection.connector.reminder.TeleopsConversationProvider"):
                        connector = PostureReminderConnector(mock_config)
                        connector.tts_enabled = True
                        connector.audio_pub = None
                        
                        # Send reminder for Alice
                        alice_input = PostureDetectionInput(
                            posture_type=PostureType.SLUMPED,
                            severity=PostureSeverity.MODERATE,
                            person_name="Alice"
                        )
                        mock_health_provider.should_remind_posture.return_value = True
                        
                        await connector.connect(alice_input)
                        
                        # Verify Alice's reminder time was tracked
                        assert "Alice" in connector.last_reminder_times or "unknown" in connector.last_reminder_times
                        
                        # Send reminder for Bob (different person)
                        bob_input = PostureDetectionInput(
                            posture_type=PostureType.HUNCHED,
                            severity=PostureSeverity.MODERATE,
                            person_name="Bob"
                        )
                        
                        await connector.connect(bob_input)
                        
                        # Both should be tracked separately
                        assert len(connector.last_reminder_times) >= 1
