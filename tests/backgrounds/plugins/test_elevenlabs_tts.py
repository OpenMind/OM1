from unittest.mock import MagicMock, patch

from backgrounds.plugins.elevenlabs_tts import ElevenLabsTTS, ElevenLabsTTSConfig


class TestElevenLabsTTSConfig:
    """Test cases for ElevenLabsTTSConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ElevenLabsTTSConfig()
        assert config.api_key is None
        assert config.elevenlabs_api_key is None
        assert config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert config.model_id == "eleven_flash_v2_5"
        assert config.output_format == "pcm_16000"

    def test_custom_api_key(self):
        """Test custom api_key configuration."""
        config = ElevenLabsTTSConfig(api_key="test-om-key")
        assert config.api_key == "test-om-key"

    def test_custom_elevenlabs_api_key(self):
        """Test custom elevenlabs_api_key configuration."""
        config = ElevenLabsTTSConfig(elevenlabs_api_key="test-el-key")
        assert config.elevenlabs_api_key == "test-el-key"

    def test_custom_voice_id(self):
        """Test custom voice_id configuration."""
        config = ElevenLabsTTSConfig(voice_id="custom_voice")
        assert config.voice_id == "custom_voice"

    def test_custom_model_id(self):
        """Test custom model_id configuration."""
        config = ElevenLabsTTSConfig(model_id="eleven_turbo_v2")
        assert config.model_id == "eleven_turbo_v2"

    def test_custom_output_format(self):
        """Test custom output_format configuration."""
        config = ElevenLabsTTSConfig(output_format="pcm_16000")
        assert config.output_format == "pcm_16000"

    def test_all_custom_values(self):
        """Test configuration with all custom values."""
        config = ElevenLabsTTSConfig(
            api_key="om-key",
            elevenlabs_api_key="el-key",
            voice_id="voice123",
            model_id="model456",
            output_format="wav_44100",
        )
        assert config.api_key == "om-key"
        assert config.elevenlabs_api_key == "el-key"
        assert config.voice_id == "voice123"
        assert config.model_id == "model456"
        assert config.output_format == "wav_44100"


class TestElevenLabsTTS:
    """Test cases for ElevenLabsTTS background plugin."""

    @patch("backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_initialization(self, mock_provider_class):
        """Test background initialization creates provider, starts, and configures it."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = ElevenLabsTTSConfig()
        background = ElevenLabsTTS(config)

        assert background.config is config
        assert background.tts == mock_provider
        mock_provider_class.assert_called_once()
        mock_provider.start.assert_called_once()
        mock_provider.configure.assert_called_once()

    @patch("backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_provider_initialized_with_correct_params(self, mock_provider_class):
        """Test that provider is initialized with correct parameters."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = ElevenLabsTTSConfig(
            api_key="om-key",
            elevenlabs_api_key="el-key",
            voice_id="voice123",
            model_id="model456",
            output_format="wav_44100",
        )
        ElevenLabsTTS(config)

        mock_provider_class.assert_called_once_with(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key="om-key",
            elevenlabs_api_key="el-key",
            voice_id="voice123",
            model_id="model456",
            output_format="wav_44100",
        )

    @patch("backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_provider_configure_called_with_correct_params(self, mock_provider_class):
        """Test that provider.configure() is called with correct parameters."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = ElevenLabsTTSConfig(
            api_key="om-key",
            elevenlabs_api_key="el-key",
            voice_id="voice123",
            model_id="model456",
            output_format="wav_44100",
        )
        ElevenLabsTTS(config)

        mock_provider.configure.assert_called_once_with(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key="om-key",
            elevenlabs_api_key="el-key",
            voice_id="voice123",
            model_id="model456",
            output_format="wav_44100",
        )

    @patch("backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_initialization_logging(self, mock_provider_class, caplog):
        """Test that initialization logs the correct message."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = ElevenLabsTTSConfig()
        with caplog.at_level("INFO"):
            ElevenLabsTTS(config)

        assert "Eleven Labs TTS Provider initialized in background" in caplog.text

    @patch("backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_start_called_before_configure(self, mock_provider_class):
        """Test that start() is called before configure()."""
        call_order = []
        mock_provider = MagicMock()
        mock_provider.start.side_effect = lambda: call_order.append("start")
        mock_provider.configure.side_effect = lambda **kwargs: call_order.append(
            "configure"
        )
        mock_provider_class.return_value = mock_provider

        config = ElevenLabsTTSConfig()
        ElevenLabsTTS(config)

        assert call_order == ["start", "configure"]

    @patch("backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_default_config_params_passed_to_provider(self, mock_provider_class):
        """Test that default config params are passed to provider."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = ElevenLabsTTSConfig()
        ElevenLabsTTS(config)

        mock_provider_class.assert_called_once_with(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=None,
            elevenlabs_api_key=None,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_flash_v2_5",
            output_format="pcm_16000",
        )

    @patch("backgrounds.plugins.elevenlabs_tts.ElevenLabsTTSProvider")
    def test_config_stored(self, mock_provider_class):
        """Test that config is stored correctly."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = ElevenLabsTTSConfig(api_key="test-key")
        background = ElevenLabsTTS(config)

        assert background.config is config
        assert background.config.api_key == "test-key"
