from unittest.mock import Mock, patch

import pytest

from inputs.plugins.google_asr import (
    LANGUAGE_CODE_MAP,
    GoogleASRInput,
    GoogleASRSensorConfig,
)


@pytest.fixture
def mock_asr_provider():
    with patch("inputs.plugins.google_asr.ASRProvider") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_sleep_ticker():
    with patch("inputs.plugins.google_asr.SleepTickerProvider") as mock:
        yield mock


@pytest.fixture
def mock_conversation():
    with patch("inputs.plugins.google_asr.TeleopsConversationProvider") as mock:
        yield mock


def test_language_code_map_contains_required_languages():
    """Test that language code map has all required languages."""
    required_languages = [
        "english",
        "chinese",
        "german",
        "french",
        "japanese",
        "korean",
    ]
    for lang in required_languages:
        assert lang in LANGUAGE_CODE_MAP


def test_korean_language_code():
    """Test that Korean language has correct code."""
    assert LANGUAGE_CODE_MAP["korean"] == "ko-KR"


def test_spanish_language_code():
    """Test that Spanish language has correct code."""
    assert LANGUAGE_CODE_MAP["spanish"] == "es-ES"


def test_italian_language_code():
    """Test that Italian language has correct code."""
    assert LANGUAGE_CODE_MAP["italian"] == "it-IT"


def test_portuguese_language_code():
    """Test that Portuguese language has correct code."""
    assert LANGUAGE_CODE_MAP["portuguese"] == "pt-BR"


def test_russian_language_code():
    """Test that Russian language has correct code."""
    assert LANGUAGE_CODE_MAP["russian"] == "ru-RU"


def test_arabic_language_code():
    """Test that Arabic language has correct code."""
    assert LANGUAGE_CODE_MAP["arabic"] == "ar-SA"


def test_init_with_korean_language(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test ASR initialization with Korean language."""
    config = GoogleASRSensorConfig(api_key="test_key", language="korean")
    _ = GoogleASRInput(config=config)

    # Verify ASR provider was called with Korean language code
    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "ko-KR"


def test_init_with_spanish_language(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test ASR initialization with Spanish language."""
    config = GoogleASRSensorConfig(api_key="test_key", language="spanish")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "es-ES"


def test_init_with_japanese_language(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test ASR initialization with Japanese language."""
    config = GoogleASRSensorConfig(api_key="test_key", language="japanese")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "ja-JP"


def test_init_with_case_insensitive_language(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test that language names are case-insensitive."""
    config = GoogleASRSensorConfig(api_key="test_key", language="KOREAN")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "ko-KR"


def test_init_with_whitespace_in_language(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test that language names handle whitespace."""
    config = GoogleASRSensorConfig(api_key="test_key", language="  korean  ")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "ko-KR"


def test_init_with_unsupported_language_defaults_to_english(
    mock_asr_provider, mock_sleep_ticker, mock_conversation, caplog
):
    """Test that unsupported language defaults to English with warning."""
    config = GoogleASRSensorConfig(api_key="test_key", language="klingon")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "en-US"
    assert "not supported" in caplog.text


def test_init_without_language_defaults_to_english(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test that missing language config defaults to English."""
    config = GoogleASRSensorConfig(api_key="test_key")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "en-US"


def test_all_language_codes_are_valid_format():
    """Test that all language codes follow proper format."""
    for lang_code in LANGUAGE_CODE_MAP.values():
        # Google language codes are typically xx-XX or xxx-Xxxx-XX format
        assert "-" in lang_code or len(lang_code) >= 2


def test_language_map_has_no_duplicates():
    """Test that no two languages map to the same code."""
    codes = list(LANGUAGE_CODE_MAP.values())
    assert len(codes) == len(set(codes))


# Tests for automatic language detection


def test_init_with_auto_language_detection(
    mock_asr_provider, mock_sleep_ticker, mock_conversation, caplog
):
    """Test ASR initialization with 'auto' language detection."""
    config = GoogleASRSensorConfig(api_key="test_key", language="auto")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    # Auto mode uses English as primary
    assert call_args[1]["language_code"] == "en-US"
    # Auto mode includes default alternatives
    assert call_args[1]["alternative_language_codes"] == [
        "cmn-Hans-CN",
        "ja-JP",
        "es-ES",
        "fr-FR",
    ]
    assert "Auto language detection enabled" in caplog.text


def test_init_with_auto_case_insensitive(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test that 'AUTO' works case-insensitively."""
    config = GoogleASRSensorConfig(api_key="test_key", language="AUTO")
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "en-US"
    assert len(call_args[1]["alternative_language_codes"]) > 0


# Tests for language list configuration


def test_init_with_language_list(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test ASR initialization with a list of languages."""
    config = GoogleASRSensorConfig(
        api_key="test_key", language=["chinese", "english", "japanese"]
    )
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    # First language is primary
    assert call_args[1]["language_code"] == "cmn-Hans-CN"
    # Rest are alternatives
    assert call_args[1]["alternative_language_codes"] == ["en-US", "ja-JP"]


def test_init_with_single_language_in_list(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test ASR initialization with a single language in list format."""
    config = GoogleASRSensorConfig(api_key="test_key", language=["korean"])
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "ko-KR"
    assert call_args[1]["alternative_language_codes"] == []


def test_init_with_empty_language_list(
    mock_asr_provider, mock_sleep_ticker, mock_conversation, caplog
):
    """Test ASR initialization with empty language list defaults to English."""
    config = GoogleASRSensorConfig(api_key="test_key", language=[])
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "en-US"
    assert "Empty language list" in caplog.text


def test_init_with_unsupported_primary_in_list(
    mock_asr_provider, mock_sleep_ticker, mock_conversation, caplog
):
    """Test that unsupported primary language in list defaults to English."""
    config = GoogleASRSensorConfig(
        api_key="test_key", language=["klingon", "chinese"]
    )
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "en-US"
    assert call_args[1]["alternative_language_codes"] == ["cmn-Hans-CN"]
    assert "not supported" in caplog.text


def test_init_with_unsupported_alternative_in_list(
    mock_asr_provider, mock_sleep_ticker, mock_conversation, caplog
):
    """Test that unsupported alternative languages are skipped."""
    config = GoogleASRSensorConfig(
        api_key="test_key", language=["english", "klingon", "chinese"]
    )
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "en-US"
    # klingon should be skipped
    assert call_args[1]["alternative_language_codes"] == ["cmn-Hans-CN"]
    assert "not supported" in caplog.text


def test_init_with_language_list_case_insensitive(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test that language list handles case-insensitivity."""
    config = GoogleASRSensorConfig(
        api_key="test_key", language=["CHINESE", "English", "JAPANESE"]
    )
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "cmn-Hans-CN"
    assert call_args[1]["alternative_language_codes"] == ["en-US", "ja-JP"]


def test_init_with_language_list_whitespace(
    mock_asr_provider, mock_sleep_ticker, mock_conversation
):
    """Test that language list handles whitespace in entries."""
    config = GoogleASRSensorConfig(
        api_key="test_key", language=["  chinese  ", " english "]
    )
    _ = GoogleASRInput(config=config)

    call_args = mock_asr_provider.call_args
    assert call_args[1]["language_code"] == "cmn-Hans-CN"
    assert call_args[1]["alternative_language_codes"] == ["en-US"]

