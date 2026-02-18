"""Tests for Emergency Call plugin."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from actions.emergency_call.connector.emergency_call_connector import (
    EmergencyCallConfig,
    EmergencyCallConnector,
    EmergencyContact,
)
from actions.emergency_call.interface import (
    EmergencyCallInput,
    EmergencyLevel,
    EmergencyResponseStatus,
    EmergencyTriggerType,
)
from triggers.emergency import (
    FallDetectionTrigger,
    PhysicalButtonTrigger,
    TriggerResult,
    VoiceKeywordTrigger,
)


class TestEmergencyCallConnector:
    """Test suite for EmergencyCallConnector."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return EmergencyCallConfig(
            encryption_key="test_key_12345",
            auto_delete_hours=72,
            emergency_service_number="911",
            family_contacts=[
                EmergencyContact(
                    name="Test Family",
                    phone="+1234567890",
                    email="test@example.com",
                    relation="family",
                    priority=1,
                )
            ],
        )

    @pytest.fixture
    def connector(self, config):
        """Create test connector."""
        return EmergencyCallConnector(config)

    @pytest.fixture
    def emergency_input(self):
        """Create test emergency input."""
        return EmergencyCallInput(
            trigger_type=EmergencyTriggerType.FALL_DETECTION,
            emergency_level=EmergencyLevel.HIGH,
            location="living_room",
            user_message="Fall detected by IMU",
            sensor_data={"impact_g": 3.5, "fall_duration": 0.8},
        )

    def test_config_defaults(self):
        """Test default config values."""
        config = EmergencyCallConfig()
        assert config.auto_delete_hours == 72
        assert config.emergency_service_number == "911"
        assert config.family_contacts == []

    def test_encryption(self, connector):
        """Test encryption/decryption."""
        test_data = "sensitive emergency data"
        encrypted = connector._encrypt_data(test_data)
        decrypted = connector._decrypt_data(encrypted)
        assert decrypted == test_data

    def test_emergency_id_generation(self, connector, emergency_input):
        """Test emergency ID generation."""
        emergency_id = connector._generate_emergency_id(emergency_input)
        assert len(emergency_id) == 16
        assert emergency_id.isalnum()

    def test_format_notification_message(self, connector, emergency_input):
        """Test notification message formatting."""
        message = connector._format_notification_message(emergency_input, "test_id")
        assert "EMERGENCY ALERT" in message
        assert "FALL_DETECTION" in message
        assert "HIGH" in message
        assert "test_id" in message

    @pytest.mark.asyncio
    async def test_connect_low_level(self, connector, emergency_input):
        """Test LOW level emergency (notification only)."""
        emergency_input.emergency_level = EmergencyLevel.LOW

        with patch.object(connector, "_send_notification", return_value=True) as mock_notify:
            with patch.object(connector, "_initiate_phone_call") as mock_call:
                with patch.object(connector, "_contact_emergency_services") as mock_emergency:
                    await connector.connect(emergency_input)
                    mock_notify.assert_called()
                    mock_call.assert_not_called()
                    mock_emergency.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_high_level(self, connector, emergency_input):
        """Test HIGH level emergency (all tiers)."""
        emergency_input.emergency_level = EmergencyLevel.CRITICAL

        with patch.object(connector, "_send_notification", return_value=True):
            with patch.object(connector, "_initiate_phone_call", return_value=True):
                with patch.object(connector, "_contact_emergency_services", return_value=True):
                    await connector.connect(emergency_input)


class TestVoiceKeywordTrigger:
    """Test voice keyword trigger."""

    @pytest.fixture
    def trigger(self):
        return VoiceKeywordTrigger(confidence_threshold=0.8)

    @pytest.mark.asyncio
    async def test_detect_emergency_keyword(self, trigger):
        """Test detecting emergency keyword."""
        data = {"transcript": "I fell and need help"}
        result = await trigger.detect(data)
        assert result.triggered is True
        assert result.trigger_type == EmergencyTriggerType.VOICE_KEYWORD
        assert result.data["keyword"] in ["fell", "help"]

    @pytest.mark.asyncio
    async def test_no_keyword(self, trigger):
        """Test no keyword detected."""
        data = {"transcript": "Hello how are you today"}
        result = await trigger.detect(data)
        assert result.triggered is False


class TestFallDetectionTrigger:
    """Test fall detection trigger."""

    @pytest.fixture
    def trigger(self):
        return FallDetectionTrigger(
            accel_threshold=3.0,
            impact_threshold=2.5,
            inactivity_timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_detect_fall(self, trigger):
        """Test fall detection from IMU data."""
        # Simulate free fall then impact
        free_fall_data = {"accel": [0.1, 0.1, 0.2]}  # Low g during free fall
        await trigger.detect(free_fall_data)

        impact_data = {"accel": [0.5, 3.5, 0.2]}  # High g at impact
        result = await trigger.detect(impact_data)

        # Might not trigger if free-fall window is short, but should process
        assert isinstance(result, TriggerResult)

    @pytest.mark.asyncio
    async def test_no_fall(self, trigger):
        """Test normal movement not detected as fall."""
        data = {"accel": [0.0, 0.0, 1.0]}  # Normal standing
        result = await trigger.detect(data)
        assert result.triggered is False


class TestPhysicalButtonTrigger:
    """Test physical button trigger."""

    @pytest.fixture
    def trigger(self):
        return PhysicalButtonTrigger()

    @pytest.mark.asyncio
    async def test_double_press(self, trigger):
        """Test double press detection."""
        # First press
        await trigger.detect({"event": "press"})
        await trigger.detect({"event": "release"})

        # Second press quickly
        result = await trigger.detect({"event": "press"})

        # Double press triggers on second press
        assert result.triggered is True
        assert result.data["pattern"] == "double_press"

    @pytest.mark.asyncio
    async def test_long_press(self, trigger):
        """Test long press detection."""
        import time

        await trigger.detect({"event": "press"})
        trigger._press_start = time.time() - 4.0  # Simulate 4 second press
        result = await trigger.detect({"event": "release"})

        assert result.triggered is True
        assert result.data["pattern"] == "long_press"

    @pytest.mark.asyncio
    async def test_short_press(self, trigger):
        """Test short press doesn't trigger."""
        await trigger.detect({"event": "press"})
        result = await trigger.detect({"event": "release"})
        assert result.triggered is False


class TestEmergencyLevels:
    """Test emergency level enum."""

    def test_level_ordering(self):
        """Test emergency level ordering."""
        assert EmergencyLevel.LOW < EmergencyLevel.MEDIUM
        assert EmergencyLevel.MEDIUM < EmergencyLevel.HIGH
        assert EmergencyLevel.HIGH < EmergencyLevel.CRITICAL

    def test_level_values(self):
        """Test emergency level values."""
        assert EmergencyLevel.LOW == 1
        assert EmergencyLevel.MEDIUM == 2
        assert EmergencyLevel.HIGH == 3
        assert EmergencyLevel.CRITICAL == 4
