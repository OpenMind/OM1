import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from providers.home_assistant_provider import HomeAssistantProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before each test."""
    HomeAssistantProvider.reset()  # type: ignore
    yield
    HomeAssistantProvider.reset()  # type: ignore


@pytest.fixture
def provider():
    """Create a HomeAssistantProvider with test defaults."""
    return HomeAssistantProvider(
        base_url="http://ha.local:8123",
        token="test-token-direct",
        token_env="HA_TEST_TOKEN",
    )


def create_aiohttp_mock(status=200, json_data=None, text_data="OK"):
    """Create aiohttp ClientSession mock with proper async context managers."""
    mock_response = Mock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)
    mock_response.text = AsyncMock(return_value=text_data)

    mock_request_cm = Mock()
    mock_request_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_request_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = Mock()
    mock_session.get = Mock(return_value=mock_request_cm)
    mock_session.post = Mock(return_value=mock_request_cm)

    mock_session_cm = Mock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    return mock_session_cm, mock_session


# --- Singleton behavior ---


def test_singleton_returns_same_instance():
    """Test that multiple calls return the same instance."""
    p1 = HomeAssistantProvider(base_url="http://ha.local:8123", token="tok1")
    p2 = HomeAssistantProvider(base_url="http://ha.local:9999", token="tok2")
    assert p1 is p2
    assert p1.base_url == "http://ha.local:8123"


def test_singleton_reset_creates_new_instance():
    """Test that reset allows a new instance to be created."""
    p1 = HomeAssistantProvider(base_url="http://ha.local:8123", token="tok1")
    HomeAssistantProvider.reset()  # type: ignore
    p2 = HomeAssistantProvider(base_url="http://ha.local:9999", token="tok2")
    assert p1 is not p2
    assert p2.base_url == "http://ha.local:9999"


# --- Token resolution ---


def test_get_token_from_env(provider):
    """Test that token is read from environment variable first."""
    with patch.dict(os.environ, {"HA_TEST_TOKEN": "env-token-value"}):
        assert provider._get_token() == "env-token-value"


def test_get_token_from_config(provider):
    """Test that direct token is used when env var is not set."""
    with patch.dict(os.environ, {}, clear=True):
        env_backup = os.environ.pop("HA_TEST_TOKEN", None)
        try:
            assert provider._get_token() == "test-token-direct"
        finally:
            if env_backup is not None:
                os.environ["HA_TEST_TOKEN"] = env_backup


def test_get_token_raises_when_missing():
    """Test that ValueError is raised when no token is available."""
    HomeAssistantProvider.reset()  # type: ignore
    p = HomeAssistantProvider(
        base_url="http://ha.local:8123",
        token="",
        token_env="NONEXISTENT_TOKEN_VAR_12345",
    )
    with patch.dict(os.environ, {}, clear=True):
        env_backup = os.environ.pop("NONEXISTENT_TOKEN_VAR_12345", None)
        try:
            with pytest.raises(ValueError, match="No Home Assistant token found"):
                p._get_token()
        finally:
            if env_backup is not None:
                os.environ["NONEXISTENT_TOKEN_VAR_12345"] = env_backup


# --- get_state ---


@pytest.mark.asyncio
async def test_get_state_success(provider):
    """Test successful state retrieval for a single entity."""
    state_data = {
        "entity_id": "light.living_room",
        "state": "on",
        "attributes": {"brightness": 200},
    }
    mock_session_cm, mock_session = create_aiohttp_mock(
        status=200, json_data=state_data
    )

    with patch(
        "providers.home_assistant_provider.aiohttp.ClientSession",
        return_value=mock_session_cm,
    ):
        result = await provider.get_state("light.living_room")

    assert result == state_data
    mock_session.get.assert_called_once()
    call_args = mock_session.get.call_args
    assert "/api/states/light.living_room" in call_args[0][0]
    headers = call_args[1]["headers"]
    assert "Bearer" in headers["Authorization"]


@pytest.mark.asyncio
async def test_get_state_http_error(provider):
    """Test that HTTP errors raise RuntimeError."""
    mock_session_cm, _ = create_aiohttp_mock(status=404, text_data="Not found")

    with patch(
        "providers.home_assistant_provider.aiohttp.ClientSession",
        return_value=mock_session_cm,
    ):
        with pytest.raises(RuntimeError, match="Failed to get state"):
            await provider.get_state("light.nonexistent")


# --- get_states ---


@pytest.mark.asyncio
async def test_get_states_with_filter(provider):
    """Test that get_states filters entities by ID."""
    all_states = [
        {"entity_id": "light.living_room", "state": "on"},
        {"entity_id": "sensor.temperature", "state": "22.5"},
        {"entity_id": "switch.garage", "state": "off"},
    ]
    mock_session_cm, _ = create_aiohttp_mock(status=200, json_data=all_states)

    with patch(
        "providers.home_assistant_provider.aiohttp.ClientSession",
        return_value=mock_session_cm,
    ):
        result = await provider.get_states(["light.living_room", "sensor.temperature"])

    assert len(result) == 2
    entity_ids = {s["entity_id"] for s in result}
    assert entity_ids == {"light.living_room", "sensor.temperature"}


@pytest.mark.asyncio
async def test_get_states_without_filter(provider):
    """Test that get_states returns all entities when no filter is given."""
    all_states = [
        {"entity_id": "light.living_room", "state": "on"},
        {"entity_id": "sensor.temperature", "state": "22.5"},
    ]
    mock_session_cm, _ = create_aiohttp_mock(status=200, json_data=all_states)

    with patch(
        "providers.home_assistant_provider.aiohttp.ClientSession",
        return_value=mock_session_cm,
    ):
        result = await provider.get_states()

    assert len(result) == 2


@pytest.mark.asyncio
async def test_get_states_http_error(provider):
    """Test that HTTP errors raise RuntimeError."""
    mock_session_cm, _ = create_aiohttp_mock(status=500, text_data="Server error")

    with patch(
        "providers.home_assistant_provider.aiohttp.ClientSession",
        return_value=mock_session_cm,
    ):
        with pytest.raises(RuntimeError, match="Failed to get states"):
            await provider.get_states(["light.living_room"])


# --- call_service ---


@pytest.mark.asyncio
async def test_call_service_success(provider):
    """Test successful service call with correct payload."""
    mock_session_cm, mock_session = create_aiohttp_mock(status=200)

    with patch(
        "providers.home_assistant_provider.aiohttp.ClientSession",
        return_value=mock_session_cm,
    ):
        await provider.call_service(
            domain="light",
            service="turn_on",
            entity_id="light.living_room",
            brightness_pct=80,
        )

    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args
    assert "/api/services/light/turn_on" in call_args[0][0]
    payload = call_args[1]["json"]
    assert payload["entity_id"] == "light.living_room"
    assert payload["brightness_pct"] == 80


@pytest.mark.asyncio
async def test_call_service_http_error(provider):
    """Test that HTTP errors raise RuntimeError."""
    mock_session_cm, _ = create_aiohttp_mock(status=400, text_data="Bad request")

    with patch(
        "providers.home_assistant_provider.aiohttp.ClientSession",
        return_value=mock_session_cm,
    ):
        with pytest.raises(RuntimeError, match="Failed to call"):
            await provider.call_service(
                domain="light",
                service="turn_on",
                entity_id="light.living_room",
            )


# --- URL construction ---


def test_base_url_trailing_slash_stripped():
    """Test that trailing slash is stripped from base_url."""
    HomeAssistantProvider.reset()  # type: ignore
    p = HomeAssistantProvider(base_url="http://ha.local:8123/", token="tok")
    assert p.base_url == "http://ha.local:8123"
