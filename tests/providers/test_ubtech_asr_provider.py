from unittest.mock import MagicMock, patch

import pytest

from providers.ubtech_asr_provider import UbtechASRProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UbtechASRProvider.reset()  # type: ignore
    yield

    try:
        # Get instance by accessing the singleton class's _singleton_instance
        provider = UbtechASRProvider._singleton_class._singleton_instance  # type: ignore
        if provider:
            provider.stop()
    except Exception:
        pass

    UbtechASRProvider.reset()  # type: ignore


@pytest.fixture
def mock_requests():
    """Mock requests.Session for UbtechASRProvider."""
    with patch("providers.ubtech_asr_provider.requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        yield mock_session


def test_initialization(mock_requests):
    """Test UbtechASRProvider initialization."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100", language_code="en")

    assert provider.robot_ip == "192.168.1.100"
    assert provider.language == "en"
    assert provider.basic_url == "http://192.168.1.100:9090/v1/"
    assert provider.running is False
    assert provider.paused is False
    assert provider.just_resumed is False
    assert UbtechASRProvider() == provider  # type: ignore


def test_singleton_pattern(mock_requests):
    """Test that UbtechASRProvider follows singleton pattern."""
    provider1 = UbtechASRProvider(robot_ip="192.168.1.100")
    provider2 = UbtechASRProvider(robot_ip="192.168.1.101")
    assert provider1 is provider2


def test_register_message_callback(mock_requests):
    """Test registering message callback."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    def callback(text):
        pass

    provider.register_message_callback(callback)

    assert provider._message_callback == callback


def test_start(mock_requests):
    """Test starting the ASR provider."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    provider.start()

    assert provider.running is True
    assert provider._thread is not None
    assert provider._thread.daemon is True


def test_start_already_running(mock_requests):
    """Test starting when already running."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    provider.start()
    first_thread = provider._thread

    provider.start()

    assert provider._thread is first_thread


def test_stop(mock_requests):
    """Test stopping the ASR provider."""
    with patch.object(UbtechASRProvider._singleton_class, "_stop_voice_iat") as mock_stop_iat:  # type: ignore
        provider = UbtechASRProvider(robot_ip="192.168.1.100")
        provider.start()

        assert provider.running is True

        provider.stop()

        assert provider.running is False
        assert mock_stop_iat.called


def test_stop_when_not_running(mock_requests):
    """Test stopping when not running."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")
    provider.stop()

    assert provider.running is False


def test_pause(mock_requests):
    """Test pausing the ASR provider."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    assert provider.paused is False

    provider.pause()

    assert provider.paused is True


def test_resume(mock_requests):
    """Test resuming the ASR provider."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")
    provider.pause()

    assert provider.paused is True
    assert provider.just_resumed is False

    provider.resume()

    assert provider.paused is False
    assert provider.just_resumed is True


def test_pause_resume_cycle(mock_requests):
    """Test pause and resume cycle."""
    provider = UbtechASRProvider(robot_ip="192.168.1.100")

    assert provider.paused is False
    assert provider.just_resumed is False

    provider.pause()
    assert provider.paused is True

    provider.resume()
    assert provider.paused is False
    assert provider.just_resumed is True


def test_language_setting(mock_requests):
    """Test language code setting."""
    provider_en = UbtechASRProvider(robot_ip="192.168.1.100", language_code="en")
    assert provider_en.language == "en"

    UbtechASRProvider.reset()  # type: ignore

    provider_zh = UbtechASRProvider(robot_ip="192.168.1.100", language_code="zh")
    assert provider_zh.language == "zh"


def test_session_headers(mock_requests):
    """Test that session headers are set correctly."""
    UbtechASRProvider(robot_ip="192.168.1.100")

    expected_headers = {"Content-Type": "application/json"}
    mock_requests.headers.update.assert_called_once_with(expected_headers)
