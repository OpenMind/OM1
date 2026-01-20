# tests/actions/emergency_alert/test_emergency_alert_connector.py
"""Unit tests for the EmergencyAlert action connector."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from actions.emergency_alert.interface import EmergencyAlert, EmergencyAlertInput

# Mock all external dependencies before importing the connector
sys.modules["zenoh"] = MagicMock()
sys.modules["providers.asr_rtsp_provider"] = MagicMock()
sys.modules["providers.elevenlabs_tts_provider"] = MagicMock()
sys.modules["providers.io_provider"] = MagicMock()
sys.modules["providers.teleops_conversation_provider"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()


class TestEmergencyAlertInterface:
    """Tests for EmergencyAlert interface."""

    def test_emergency_alert_input_initialization(self):
        """Test EmergencyAlertInput initialization."""
        alert_input = EmergencyAlertInput(action="Fire detected in sector 5!")
        assert alert_input.action == "Fire detected in sector 5!"

    def test_emergency_alert_input_empty_message(self):
        """Test EmergencyAlertInput with empty message."""
        alert_input = EmergencyAlertInput(action="")
        assert alert_input.action == ""

    def test_emergency_alert_interface_initialization(self):
        """Test EmergencyAlert interface initialization."""
        input_data = EmergencyAlertInput(action="Intruder alert!")
        output_data = EmergencyAlertInput(action="Intruder alert!")

        alert = EmergencyAlert(input=input_data, output=output_data)

        assert alert.input == input_data
        assert alert.output == output_data


class TestSpeakElevenLabsTTSConfig:
    """Tests for SpeakElevenLabsTTSConfig."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from actions.emergency_alert.connector.elevenlabs_tts import (
            SpeakElevenLabsTTSConfig,
        )

        config = SpeakElevenLabsTTSConfig()

        assert config.elevenlabs_api_key is None
        assert config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert config.model_id == "eleven_flash_v2_5"
        assert config.output_format == "mp3_44100_128"
        assert config.silence_rate == 0

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        from actions.emergency_alert.connector.elevenlabs_tts import (
            SpeakElevenLabsTTSConfig,
        )

        config = SpeakElevenLabsTTSConfig(
            elevenlabs_api_key="test_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="wav_44100",
            silence_rate=3,
        )

        assert config.elevenlabs_api_key == "test_key"
        assert config.voice_id == "custom_voice"
        assert config.model_id == "custom_model"
        assert config.output_format == "wav_44100"
        assert config.silence_rate == 3


class TestEmergencyAlertElevenLabsTTSConnector:
    """Tests for EmergencyAlertElevenLabsTTSConnector."""

    @pytest.fixture
    def mock_dependencies(self):
        """Set up mock dependencies for the connector."""
        mock_io_provider = MagicMock()
        mock_io_provider.llm_prompt = None

        mock_tts = MagicMock()
        mock_tts.get_pending_message_count.return_value = 0
        mock_tts.create_pending_message.return_value = {"id": "test", "text": "alert"}

        mock_asr = MagicMock()
        mock_conversation_provider = MagicMock()

        mock_session = MagicMock()
        mock_audio_pub = MagicMock()

        return {
            "io_provider": mock_io_provider,
            "tts": mock_tts,
            "asr": mock_asr,
            "conversation_provider": mock_conversation_provider,
            "session": mock_session,
            "audio_pub": mock_audio_pub,
        }

    def test_connector_inherits_from_action_connector(self):
        """Test that EmergencyAlertElevenLabsTTSConnector inherits from ActionConnector."""
        from actions.base import ActionConnector
        from actions.emergency_alert.connector.elevenlabs_tts import (
            EmergencyAlertElevenLabsTTSConnector,
        )

        assert issubclass(EmergencyAlertElevenLabsTTSConnector, ActionConnector)

    @pytest.mark.asyncio
    async def test_connect_with_tts_disabled(self, mock_dependencies, caplog):
        """Test connect method when TTS is disabled."""
        from actions.emergency_alert.connector.elevenlabs_tts import (
            EmergencyAlertElevenLabsTTSConnector,
            SpeakElevenLabsTTSConfig,
        )

        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.IOProvider",
            return_value=mock_dependencies["io_provider"],
        ):
            with patch(
                "actions.emergency_alert.connector.elevenlabs_tts.ElevenLabsTTSProvider",
                return_value=mock_dependencies["tts"],
            ):
                with patch(
                    "actions.emergency_alert.connector.elevenlabs_tts.ASRRTSPProvider",
                    return_value=mock_dependencies["asr"],
                ):
                    with patch(
                        "actions.emergency_alert.connector.elevenlabs_tts.TeleopsConversationProvider",
                        return_value=mock_dependencies["conversation_provider"],
                    ):
                        with patch(
                            "actions.emergency_alert.connector.elevenlabs_tts.open_zenoh_session",
                            return_value=mock_dependencies["session"],
                        ):
                            config = SpeakElevenLabsTTSConfig()
                            connector = EmergencyAlertElevenLabsTTSConnector(config)

                            # Disable TTS
                            connector.tts_enabled = False

                            alert_input = EmergencyAlertInput(action="Emergency!")

                            with caplog.at_level(logging.INFO):
                                await connector.connect(alert_input)

                            assert "TTS is disabled" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_with_too_many_pending_messages(
        self, mock_dependencies, caplog
    ):
        """Test connect method when too many messages are pending."""
        from actions.emergency_alert.connector.elevenlabs_tts import (
            EmergencyAlertElevenLabsTTSConnector,
            SpeakElevenLabsTTSConfig,
        )

        # Set pending message count > 0
        mock_dependencies["tts"].get_pending_message_count.return_value = 5

        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.IOProvider",
            return_value=mock_dependencies["io_provider"],
        ):
            with patch(
                "actions.emergency_alert.connector.elevenlabs_tts.ElevenLabsTTSProvider",
                return_value=mock_dependencies["tts"],
            ):
                with patch(
                    "actions.emergency_alert.connector.elevenlabs_tts.ASRRTSPProvider",
                    return_value=mock_dependencies["asr"],
                ):
                    with patch(
                        "actions.emergency_alert.connector.elevenlabs_tts.TeleopsConversationProvider",
                        return_value=mock_dependencies["conversation_provider"],
                    ):
                        with patch(
                            "actions.emergency_alert.connector.elevenlabs_tts.open_zenoh_session",
                            return_value=mock_dependencies["session"],
                        ):
                            config = SpeakElevenLabsTTSConfig()
                            connector = EmergencyAlertElevenLabsTTSConnector(config)

                            alert_input = EmergencyAlertInput(
                                action="Emergency message!"
                            )

                            with caplog.at_level(logging.WARNING):
                                await connector.connect(alert_input)

                            assert "Too many pending TTS messages" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_creates_pending_message(self, mock_dependencies):
        """Test connect method creates a pending message."""
        from actions.emergency_alert.connector.elevenlabs_tts import (
            EmergencyAlertElevenLabsTTSConnector,
            SpeakElevenLabsTTSConfig,
        )

        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.IOProvider",
            return_value=mock_dependencies["io_provider"],
        ):
            with patch(
                "actions.emergency_alert.connector.elevenlabs_tts.ElevenLabsTTSProvider",
                return_value=mock_dependencies["tts"],
            ):
                with patch(
                    "actions.emergency_alert.connector.elevenlabs_tts.ASRRTSPProvider",
                    return_value=mock_dependencies["asr"],
                ):
                    with patch(
                        "actions.emergency_alert.connector.elevenlabs_tts.TeleopsConversationProvider",
                        return_value=mock_dependencies["conversation_provider"],
                    ):
                        with patch(
                            "actions.emergency_alert.connector.elevenlabs_tts.open_zenoh_session",
                            return_value=mock_dependencies["session"],
                        ):
                            config = SpeakElevenLabsTTSConfig()
                            connector = EmergencyAlertElevenLabsTTSConnector(config)
                            connector.audio_pub = mock_dependencies["audio_pub"]

                            alert_input = EmergencyAlertInput(
                                action="Fire in building A!"
                            )

                            await connector.connect(alert_input)

                            mock_dependencies[
                                "tts"
                            ].create_pending_message.assert_called_with(
                                "Fire in building A!"
                            )

    def test_zenoh_tts_status_request_enable(self, mock_dependencies):
        """Test _zenoh_tts_status_request enables TTS when code is 1."""
        from actions.emergency_alert.connector.elevenlabs_tts import (
            EmergencyAlertElevenLabsTTSConnector,
            SpeakElevenLabsTTSConfig,
        )

        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.IOProvider",
            return_value=mock_dependencies["io_provider"],
        ):
            with patch(
                "actions.emergency_alert.connector.elevenlabs_tts.ElevenLabsTTSProvider",
                return_value=mock_dependencies["tts"],
            ):
                with patch(
                    "actions.emergency_alert.connector.elevenlabs_tts.ASRRTSPProvider",
                    return_value=mock_dependencies["asr"],
                ):
                    with patch(
                        "actions.emergency_alert.connector.elevenlabs_tts.TeleopsConversationProvider",
                        return_value=mock_dependencies["conversation_provider"],
                    ):
                        with patch(
                            "actions.emergency_alert.connector.elevenlabs_tts.open_zenoh_session",
                            return_value=mock_dependencies["session"],
                        ):
                            with patch(
                                "actions.emergency_alert.connector.elevenlabs_tts.TTSStatusRequest"
                            ) as mock_tts_status:
                                config = SpeakElevenLabsTTSConfig()
                                connector = EmergencyAlertElevenLabsTTSConnector(config)

                                # Disable TTS first
                                connector.tts_enabled = False

                                # Mock the deserialized status with code = 1 (enable)
                                mock_status = MagicMock()
                                mock_status.code = 1
                                mock_tts_status.deserialize.return_value = mock_status

                                # Simulate receiving enable message
                                mock_data = MagicMock()
                                mock_data.payload.to_bytes.return_value = b"test"
                                connector._zenoh_tts_status_request(mock_data)

                                assert connector.tts_enabled is True

    def test_zenoh_tts_status_request_disable(self, mock_dependencies):
        """Test _zenoh_tts_status_request disables TTS when code is 0."""
        from actions.emergency_alert.connector.elevenlabs_tts import (
            EmergencyAlertElevenLabsTTSConnector,
            SpeakElevenLabsTTSConfig,
        )

        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.IOProvider",
            return_value=mock_dependencies["io_provider"],
        ):
            with patch(
                "actions.emergency_alert.connector.elevenlabs_tts.ElevenLabsTTSProvider",
                return_value=mock_dependencies["tts"],
            ):
                with patch(
                    "actions.emergency_alert.connector.elevenlabs_tts.ASRRTSPProvider",
                    return_value=mock_dependencies["asr"],
                ):
                    with patch(
                        "actions.emergency_alert.connector.elevenlabs_tts.TeleopsConversationProvider",
                        return_value=mock_dependencies["conversation_provider"],
                    ):
                        with patch(
                            "actions.emergency_alert.connector.elevenlabs_tts.open_zenoh_session",
                            return_value=mock_dependencies["session"],
                        ):
                            with patch(
                                "actions.emergency_alert.connector.elevenlabs_tts.TTSStatusRequest"
                            ) as mock_tts_status:
                                config = SpeakElevenLabsTTSConfig()
                                connector = EmergencyAlertElevenLabsTTSConnector(config)

                                # Ensure TTS is enabled first
                                connector.tts_enabled = True

                                # Mock the deserialized status with code = 0 (disable)
                                mock_status = MagicMock()
                                mock_status.code = 0
                                mock_tts_status.deserialize.return_value = mock_status

                                # Simulate receiving disable message
                                mock_data = MagicMock()
                                mock_data.payload.to_bytes.return_value = b"test"
                                connector._zenoh_tts_status_request(mock_data)

                                assert connector.tts_enabled is False
