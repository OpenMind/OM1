from unittest.mock import MagicMock, patch

from src.backgrounds.plugins.elevenlabs_tts import (
    ElevenLabsTTS,
    ElevenLabsTTSConfig,
)


class TestElevenLabsTTSConfig:
    def test_config_defaults(self):
        config = ElevenLabsTTSConfig()
        assert config.api_key is None
        assert config.elevenlabs_api_key is None
        assert config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert config.model_id == "eleven_flash_v2_5"
        assert config.output_format == "mp3_44100_128"


class TestElevenLabsTTS:
    @patch("src.backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    @patch("src.backgrounds.plugins.elevenlabs_tts.logging")
    def test_initialization_and_start_configure(
        self, mock_logging, mock_provider_class
    ):
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        config = ElevenLabsTTSConfig(
            api_key="test_api_key",
            elevenlabs_api_key="test_e_key",
            voice_id="test_voice",
            model_id="test_model",
            output_format="test_format",
        )

        tts_bg = ElevenLabsTTS(config)

        expected_url = "https://api.openmind.org/api/core/elevenlabs/tts"
        mock_provider_class.assert_called_once_with(
            url=expected_url,
            api_key="test_api_key",
            elevenlabs_api_key="test_e_key",
            voice_id="test_voice",
            model_id="test_model",
            output_format="test_format",
        )

        mock_provider_instance.start.assert_called_once()
        mock_provider_instance.configure.assert_called_once_with(
            url=expected_url,
            api_key="test_api_key",
            elevenlabs_api_key="test_e_key",
            voice_id="test_voice",
            model_id="test_model",
            output_format="test_format",
        )

        mock_logging.info.assert_called_once_with(
            "Eleven Labs TTS Provider initialized in background"
        )
        assert tts_bg.tts is mock_provider_instance

    @patch("src.backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    @patch("src.backgrounds.plugins.elevenlabs_tts.logging")
    def test_initialization_with_defaults(self, mock_logging, mock_provider_class):
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        config = ElevenLabsTTSConfig()
        tts_bg = ElevenLabsTTS(config)

        expected_url = "https://api.openmind.org/api/core/elevenlabs/tts"
        mock_provider_class.assert_called_once_with(
            url=expected_url,
            api_key=None,
            elevenlabs_api_key=None,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )

        mock_provider_instance.start.assert_called_once()
        mock_provider_instance.configure.assert_called_once_with(
            url=expected_url,
            api_key=None,
            elevenlabs_api_key=None,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )

        mock_logging.info.assert_called_once_with(
            "Eleven Labs TTS Provider initialized in background"
        )
        assert tts_bg.tts is mock_provider_instance