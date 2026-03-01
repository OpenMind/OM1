import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from inputs.plugins.home_assistant import (
    HomeAssistantInputConfig,
    HomeAssistantStateInput,
)


class TestHomeAssistantInputConfig:
    """Tests for HomeAssistantInputConfig."""

    def test_default_values(self):
        """Test config with default values."""
        config = HomeAssistantInputConfig()
        assert config.base_url == ""
        assert config.token == ""
        assert config.entity_ids == ""
        assert config.poll_interval == 30.0

    def test_custom_values(self):
        """Test config with custom values."""
        config = HomeAssistantInputConfig(
            base_url="http://homeassistant.local:8123",
            token="my_token",
            entity_ids="light.living_room,switch.fan",
            poll_interval=60.0,
        )
        assert config.base_url == "http://homeassistant.local:8123"
        assert config.token == "my_token"
        assert config.entity_ids == "light.living_room,switch.fan"
        assert config.poll_interval == 60.0


class TestHomeAssistantStateInputInit:
    """Tests for HomeAssistantStateInput initialization."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IOProvider."""
        with patch("inputs.plugins.home_assistant.IOProvider") as mock:
            yield mock

    def test_init_parses_entity_ids(self, mock_io_provider):
        """Test that entity_ids string is parsed into list."""
        config = HomeAssistantInputConfig(
            base_url="http://ha.local:8123",
            token="tok",
            entity_ids="light.living_room, switch.fan , climate.bedroom",
        )
        ha = HomeAssistantStateInput(config)
        assert ha.entity_ids == ["light.living_room", "switch.fan", "climate.bedroom"]

    def test_init_strips_trailing_slash(self, mock_io_provider):
        """Test that trailing slash is stripped from base_url."""
        config = HomeAssistantInputConfig(
            base_url="http://ha.local:8123/",
            token="tok",
        )
        ha = HomeAssistantStateInput(config)
        assert ha.base_url == "http://ha.local:8123"

    def test_init_warns_missing_base_url(self, mock_io_provider):
        """Test warning when base_url is missing."""
        with patch("inputs.plugins.home_assistant.logging.warning") as mock_warn:
            config = HomeAssistantInputConfig(token="tok", entity_ids="light.x")
            HomeAssistantStateInput(config)
            assert any("base_url" in str(c) for c in mock_warn.call_args_list)

    def test_init_warns_missing_token(self, mock_io_provider):
        """Test warning when token is missing."""
        with patch("inputs.plugins.home_assistant.logging.warning") as mock_warn:
            config = HomeAssistantInputConfig(
                base_url="http://ha.local:8123", entity_ids="light.x"
            )
            HomeAssistantStateInput(config)
            assert any("token" in str(c) for c in mock_warn.call_args_list)

    def test_init_warns_missing_entity_ids(self, mock_io_provider):
        """Test warning when entity_ids is missing."""
        with patch("inputs.plugins.home_assistant.logging.warning") as mock_warn:
            config = HomeAssistantInputConfig(
                base_url="http://ha.local:8123", token="tok"
            )
            HomeAssistantStateInput(config)
            assert any("entity_ids" in str(c) for c in mock_warn.call_args_list)

    def test_descriptor_for_llm(self, mock_io_provider):
        """Test descriptor_for_LLM is set correctly."""
        config = HomeAssistantInputConfig()
        ha = HomeAssistantStateInput(config)
        assert ha.descriptor_for_LLM == "Home Assistant Device States"


def make_ha_input():
    """Helper to create a HomeAssistantStateInput with mocked IOProvider."""
    with patch("inputs.plugins.home_assistant.IOProvider"):
        config = HomeAssistantInputConfig(
            base_url="http://ha.local:8123",
            token="test_token",
            entity_ids="light.living_room,switch.fan",
            poll_interval=30.0,
        )
        return HomeAssistantStateInput(config)


def mock_ha_get_session(status=200, json_data=None):
    """Helper to mock aiohttp.ClientSession for GET requests."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data or {})

    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_get)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    ctx = patch(
        "inputs.plugins.home_assistant.aiohttp.ClientSession", return_value=mock_session
    )
    return ctx, mock_session


class TestFetchState:
    """Tests for _fetch_state()."""

    @pytest.mark.asyncio
    async def test_fetch_state_success(self):
        """Test successful state fetch."""
        ha = make_ha_input()
        state_data = {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"brightness": 255, "friendly_name": "Living Room Light"},
        }
        ctx, _ = mock_ha_get_session(status=200, json_data=state_data)
        with ctx:
            result = await ha._fetch_state("light.living_room")
            assert result == state_data

    @pytest.mark.asyncio
    async def test_fetch_state_error_status(self):
        """Test handling of non-200 status."""
        ha = make_ha_input()
        ctx, _ = mock_ha_get_session(status=404)
        with ctx:
            with patch("inputs.plugins.home_assistant.logging.error") as mock_err:
                result = await ha._fetch_state("light.missing")
                assert result is None
                assert any("404" in str(c) for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_fetch_state_no_base_url(self):
        """Test that fetch returns None without base_url."""
        with patch("inputs.plugins.home_assistant.IOProvider"):
            config = HomeAssistantInputConfig(token="tok", entity_ids="light.x")
            ha = HomeAssistantStateInput(config)
        result = await ha._fetch_state("light.x")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_state_no_token(self):
        """Test that fetch returns None without token."""
        with patch("inputs.plugins.home_assistant.IOProvider"):
            config = HomeAssistantInputConfig(
                base_url="http://ha.local:8123", entity_ids="light.x"
            )
            ha = HomeAssistantStateInput(config)
        result = await ha._fetch_state("light.x")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_state_timeout(self):
        """Test handling of timeout error."""

        ha = make_ha_input()
        with patch("inputs.plugins.home_assistant.aiohttp.ClientSession") as mock_cls:
            mock_cls.side_effect = asyncio.TimeoutError()
            with patch("inputs.plugins.home_assistant.logging.error") as mock_err:
                result = await ha._fetch_state("light.x")
                assert result is None
                assert any("timeout" in str(c).lower() for c in mock_err.call_args_list)

    @pytest.mark.asyncio
    async def test_fetch_state_client_error(self):
        """Test handling of aiohttp.ClientError."""

        ha = make_ha_input()
        with patch("inputs.plugins.home_assistant.aiohttp.ClientSession") as mock_cls:
            mock_cls.side_effect = aiohttp.ClientError("conn refused")
            with patch("inputs.plugins.home_assistant.logging.error") as mock_err:
                result = await ha._fetch_state("light.x")
                assert result is None
                assert any(
                    "network error" in str(c).lower() for c in mock_err.call_args_list
                )

    @pytest.mark.asyncio
    async def test_fetch_state_unexpected_exception(self):
        """Test handling of unexpected exception."""
        ha = make_ha_input()
        with patch("inputs.plugins.home_assistant.aiohttp.ClientSession") as mock_cls:
            mock_cls.side_effect = RuntimeError("unexpected")
            with patch("inputs.plugins.home_assistant.logging.error") as mock_err:
                result = await ha._fetch_state("light.x")
                assert result is None
                assert any(
                    "unexpected error" in str(c).lower()
                    for c in mock_err.call_args_list
                )


class TestPoll:
    """Tests for _poll() behavior."""

    @pytest.mark.asyncio
    async def test_poll_returns_none_before_interval(self):
        """Test that poll returns None when interval has not elapsed."""
        ha = make_ha_input()
        ha._last_poll_time = time.time()

        with patch(
            "inputs.plugins.home_assistant.asyncio.sleep", new_callable=AsyncMock
        ):
            result = await ha._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_poll_fetches_after_interval(self):
        """Test that poll fetches states after interval elapses."""
        ha = make_ha_input()
        ha._last_poll_time = time.time() - 60.0

        state_data = {"entity_id": "light.living_room", "state": "on", "attributes": {}}

        with patch(
            "inputs.plugins.home_assistant.asyncio.sleep", new_callable=AsyncMock
        ):
            with patch.object(ha, "_fetch_state", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = state_data
                result = await ha._poll()
                assert result is not None
                assert len(result) == 2  # two entity_ids

    @pytest.mark.asyncio
    async def test_poll_returns_none_with_no_entity_ids(self):
        """Test that poll returns None when no entity_ids configured."""
        with patch("inputs.plugins.home_assistant.IOProvider"):
            config = HomeAssistantInputConfig(
                base_url="http://ha.local:8123",
                token="tok",
                entity_ids="",
                poll_interval=0.0,
            )
            ha = HomeAssistantStateInput(config)
        ha._last_poll_time = 0.0

        with patch(
            "inputs.plugins.home_assistant.asyncio.sleep", new_callable=AsyncMock
        ):
            result = await ha._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_poll_skips_failed_fetches(self):
        """Test that failed fetches are excluded from results."""
        ha = make_ha_input()
        ha._last_poll_time = 0.0

        with patch(
            "inputs.plugins.home_assistant.asyncio.sleep", new_callable=AsyncMock
        ):
            with patch.object(ha, "_fetch_state", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = None
                result = await ha._poll()
                assert result is None


class TestFormatState:
    """Tests for _format_state()."""

    @pytest.fixture
    def ha(self):
        return make_ha_input()

    def test_format_basic_state(self, ha):
        """Test formatting a basic state."""
        state = {
            "entity_id": "switch.fan",
            "state": "off",
            "attributes": {"friendly_name": "Bedroom Fan"},
        }
        result = ha._format_state(state)
        assert "Bedroom Fan" in result
        assert "switch.fan" in result
        assert "off" in result

    def test_format_state_with_brightness(self, ha):
        """Test formatting light state with brightness."""
        state = {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Living Room", "brightness": 128},
        }
        result = ha._format_state(state)
        assert "50%" in result

    def test_format_state_with_color(self, ha):
        """Test formatting light state with color."""
        state = {
            "entity_id": "light.lamp",
            "state": "on",
            "attributes": {"friendly_name": "Lamp", "color_name": "red"},
        }
        result = ha._format_state(state)
        assert "red" in result

    def test_format_state_with_temperature(self, ha):
        """Test formatting climate state with temperature."""
        state = {
            "entity_id": "climate.bedroom",
            "state": "heat",
            "attributes": {
                "friendly_name": "Bedroom AC",
                "temperature": 22,
                "current_temperature": 20,
            },
        }
        result = ha._format_state(state)
        assert "22" in result
        assert "20" in result
        assert "°C" in result

    def test_format_state_no_friendly_name(self, ha):
        """Test that entity_id is used when friendly_name is absent."""
        state = {
            "entity_id": "switch.garage",
            "state": "on",
            "attributes": {},
        }
        result = ha._format_state(state)
        assert "switch.garage" in result

    def test_format_state_brightness_none(self, ha):
        """Test that None brightness is handled gracefully."""
        state = {
            "entity_id": "light.x",
            "state": "on",
            "attributes": {"brightness": None},
        }
        result = ha._format_state(state)
        assert "brightness" not in result


class TestRawToText:
    """Tests for _raw_to_text() and raw_to_text()."""

    @pytest.fixture
    def ha(self):
        return make_ha_input()

    @pytest.mark.asyncio
    async def test_raw_to_text_none_returns_none(self, ha):
        """Test that None input returns None."""
        result = await ha._raw_to_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_new_state_returns_message(self, ha):
        """Test that new state change returns a message."""
        states = [{"entity_id": "light.living_room", "state": "on", "attributes": {}}]
        result = await ha._raw_to_text(states)
        assert result is not None
        assert "light.living_room" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_no_change_returns_none(self, ha):
        """Test that unchanged state returns None."""
        states = [{"entity_id": "light.living_room", "state": "on", "attributes": {}}]
        await ha._raw_to_text(states)
        result = await ha._raw_to_text(states)
        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_detects_state_change(self, ha):
        """Test that state change is detected on second poll."""
        states_on = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {}}
        ]
        states_off = [
            {"entity_id": "light.living_room", "state": "off", "attributes": {}}
        ]

        await ha._raw_to_text(states_on)
        result = await ha._raw_to_text(states_off)
        assert result is not None
        assert "off" in result.message

    @pytest.mark.asyncio
    async def test_raw_to_text_updates_last_states(self, ha):
        """Test that _last_states is updated after processing."""
        states = [{"entity_id": "switch.fan", "state": "on", "attributes": {}}]
        await ha._raw_to_text(states)
        assert ha._last_states["switch.fan"] == "on"

    @pytest.mark.asyncio
    async def test_raw_to_text_adds_to_messages(self, ha):
        """Test that raw_to_text adds message to buffer."""
        states = [{"entity_id": "light.x", "state": "on", "attributes": {}}]
        await ha.raw_to_text(states)
        assert len(ha.messages) == 1

    @pytest.mark.asyncio
    async def test_raw_to_text_none_does_not_add_message(self, ha):
        """Test that None input does not add to buffer."""
        await ha.raw_to_text(None)
        assert len(ha.messages) == 0


class TestFormattedLatestBuffer:
    """Tests for formatted_latest_buffer()."""

    @pytest.fixture
    def mock_io_provider(self):
        with patch("inputs.plugins.home_assistant.IOProvider") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def ha(self, mock_io_provider):
        config = HomeAssistantInputConfig(
            base_url="http://ha.local:8123",
            token="tok",
            entity_ids="light.x",
        )
        ha = HomeAssistantStateInput(config)
        ha.io_provider = mock_io_provider
        return ha

    def test_formatted_latest_buffer_empty(self, ha):
        """Test that empty buffer returns None."""
        result = ha.formatted_latest_buffer()
        assert result is None

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_with_message(self, ha):
        """Test formatting with message in buffer."""
        states = [{"entity_id": "light.x", "state": "on", "attributes": {}}]
        await ha.raw_to_text(states)

        result = ha.formatted_latest_buffer()
        assert result is not None
        assert "Home Assistant Device States" in result
        assert "// START" in result
        assert "// END" in result
        assert "light.x" in result

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_clears_messages(self, ha):
        """Test that buffer is cleared after formatting."""
        states = [{"entity_id": "light.x", "state": "on", "attributes": {}}]
        await ha.raw_to_text(states)
        assert len(ha.messages) == 1

        ha.formatted_latest_buffer()
        assert len(ha.messages) == 0

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_calls_io_provider(self, ha):
        """Test that io_provider.add_input is called."""
        states = [{"entity_id": "light.x", "state": "on", "attributes": {}}]
        await ha.raw_to_text(states)
        ha.formatted_latest_buffer()
        ha.io_provider.add_input.assert_called_once()
