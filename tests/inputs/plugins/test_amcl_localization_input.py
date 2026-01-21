from unittest.mock import Mock, PropertyMock, patch

import pytest

from inputs.plugins.amcl_localization_input import (
    AMCLLocalizationInput,
    Message,
    SensorConfig,
)


@pytest.fixture
def mock_amcl_provider():
    with patch(
        "inputs.plugins.amcl_localization_input.UnitreeGo2AMCLProvider"
    ) as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.amcl_localization_input.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def amcl_input_instance(mock_amcl_provider, mock_io_provider):
    config = SensorConfig()
    instance = AMCLLocalizationInput(config=config)
    instance.amcl_provider = mock_amcl_provider
    instance.io_provider = mock_io_provider
    return instance


@pytest.mark.asyncio
async def test_poll_localized(amcl_input_instance, mock_amcl_provider):
    mock_amcl_provider.is_localized = True
    mock_pose = Mock()
    mock_position = Mock()
    mock_position.x = 1.0
    mock_position.y = 2.0
    mock_position.z = 0.0
    mock_pose.position = mock_position
    mock_amcl_provider.pose = mock_pose

    result = await amcl_input_instance._poll()

    assert "LOCALIZED" in result


@pytest.mark.asyncio
async def test_poll_not_localized(amcl_input_instance, mock_amcl_provider):
    mock_amcl_provider.is_localized = False
    mock_amcl_provider.pose = None

    result = await amcl_input_instance._poll()

    assert "NOT LOCALIZED" in result


@pytest.mark.asyncio
async def test_poll_pose_is_none(amcl_input_instance, mock_amcl_provider):
    mock_amcl_provider.is_localized = True
    mock_amcl_provider.pose = None

    result = await amcl_input_instance._poll()

    assert "NOT LOCALIZED" in result


@pytest.mark.asyncio
async def test_poll_exception(amcl_input_instance, mock_amcl_provider):
    type(mock_amcl_provider).is_localized = PropertyMock(
        side_effect=AttributeError("Mock error")
    )

    result = await amcl_input_instance._poll()

    assert "LOCALIZATION ERROR" in result


@pytest.mark.asyncio
async def test_raw_to_text_none(amcl_input_instance):
    result = await amcl_input_instance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_message(amcl_input_instance):
    with patch("time.time", return_value=1234.0):
        result = await amcl_input_instance._raw_to_text("test message")

    assert isinstance(result, Message)
    assert result.timestamp == 1234.0
    assert result.message == "test message"


@pytest.mark.asyncio
async def test_raw_to_text_adds_to_buffer(amcl_input_instance):
    with patch("time.time", return_value=1234.0):
        await amcl_input_instance.raw_to_text("test message")

    assert len(amcl_input_instance.messages) == 1
    assert amcl_input_instance.messages[0].message == "test message"


@pytest.mark.asyncio
async def test_raw_to_text_none_does_not_add_to_buffer(amcl_input_instance):
    initial_len = len(amcl_input_instance.messages)
    await amcl_input_instance.raw_to_text(None)

    assert len(amcl_input_instance.messages) == initial_len


def test_formatted_latest_buffer_empty(amcl_input_instance):
    result = amcl_input_instance.formatted_latest_buffer()

    assert result is None


def test_formatted_latest_buffer_with_message(amcl_input_instance):
    msg = Message(timestamp=1234.0, message="buffered message")
    amcl_input_instance.messages = [msg]

    result = amcl_input_instance.formatted_latest_buffer()

    assert "INPUT:" in result
    assert "buffered message" in result
    assert len(amcl_input_instance.messages) == 0
    amcl_input_instance.io_provider.add_input.assert_called_once()


def test_initialization_sets_providers_and_buffer(amcl_input_instance):
    assert amcl_input_instance.amcl_provider is not None
    assert amcl_input_instance.io_provider is not None
    assert hasattr(amcl_input_instance, "messages")
    assert isinstance(amcl_input_instance.messages, list)
    assert len(amcl_input_instance.messages) == 0
