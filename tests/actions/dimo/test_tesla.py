from unittest.mock import AsyncMock, Mock, patch

import pytest

from actions.dimo.connector.tesla import DIMOTeslaConfig, DIMOTeslaConnector


@pytest.fixture
def mock_dimo():
    """Mock DIMO SDK."""
    with patch("actions.dimo.connector.tesla.DIMO") as mock:
        mock_instance = Mock()
        mock_instance.auth.get_token.return_value = {"access_token": "test_dev_jwt"}
        mock_instance.token_exchange.exchange.return_value = {
            "token": "test_vehicle_jwt"
        }
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def tesla_connector(mock_dimo):
    """Create DIMOTeslaConnector with mocked dependencies."""
    config = DIMOTeslaConfig(
        client_id="test_client_id",
        domain="test_domain",
        private_key="test_private_key",
        token_id=123456,
    )
    connector = DIMOTeslaConnector(config)
    connector.vehicle_jwt = "test_jwt"
    connector.token_id = "123456"
    return connector


def create_aiohttp_mock(status=200):
    """Create aiohttp ClientSession mock with proper async context managers."""
    mock_response = Mock()
    mock_response.status = status
    mock_response.text = AsyncMock(return_value="OK")

    mock_post_cm = Mock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = Mock()
    mock_session.post = Mock(return_value=mock_post_cm)

    mock_session_cm = Mock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    return mock_session_cm, mock_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_input,expected_endpoint",
    [
        ("lock doors", "/commands/doors/lock"),
        ("Lock Doors", "/commands/doors/lock"),
        ("LOCK DOORS", "/commands/doors/lock"),
        ("Lock doors", "/commands/doors/lock"),
        ("lOcK dOoRs", "/commands/doors/lock"),
    ],
)
async def test_lock_doors_case_insensitive(
    tesla_connector, action_input, expected_endpoint
):
    """Test that 'lock doors' command works regardless of case."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
    ):
        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_session.post.assert_called_once()
        call_url = mock_session.post.call_args[0][0]
        assert expected_endpoint in call_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_input,expected_endpoint",
    [
        ("unlock doors", "/commands/doors/unlock"),
        ("Unlock Doors", "/commands/doors/unlock"),
        ("UNLOCK DOORS", "/commands/doors/unlock"),
    ],
)
async def test_unlock_doors_case_insensitive(
    tesla_connector, action_input, expected_endpoint
):
    """Test that 'unlock doors' command works regardless of case."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
    ):
        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_session.post.assert_called_once()
        call_url = mock_session.post.call_args[0][0]
        assert expected_endpoint in call_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_input,expected_endpoint",
    [
        ("open frunk", "/commands/frunk/open"),
        ("Open Frunk", "/commands/frunk/open"),
        ("OPEN FRUNK", "/commands/frunk/open"),
    ],
)
async def test_open_frunk_case_insensitive(
    tesla_connector, action_input, expected_endpoint
):
    """Test that 'open frunk' command works regardless of case."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
    ):
        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_session.post.assert_called_once()
        call_url = mock_session.post.call_args[0][0]
        assert expected_endpoint in call_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_input,expected_endpoint",
    [
        ("open trunk", "/commands/trunk/open"),
        ("Open Trunk", "/commands/trunk/open"),
        ("OPEN TRUNK", "/commands/trunk/open"),
    ],
)
async def test_open_trunk_case_insensitive(
    tesla_connector, action_input, expected_endpoint
):
    """Test that 'open trunk' command works regardless of case."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
    ):
        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_session.post.assert_called_once()
        call_url = mock_session.post.call_args[0][0]
        assert expected_endpoint in call_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action_input",
    [
        "idle",
        "Idle",
        "IDLE",
    ],
)
async def test_idle_case_insensitive(tesla_connector, action_input):
    """Test that 'idle' command works regardless of case and makes no HTTP call."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ) as mock_client_session,
    ):
        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_client_session.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_action_logs_error(tesla_connector):
    """Test that unknown actions are logged as errors."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ) as mock_client_session,
        patch("actions.dimo.connector.tesla.logging") as mock_logging,
    ):
        input_interface = Mock()
        input_interface.action = "invalid_action"

        await tesla_connector.connect(input_interface)

        mock_client_session.assert_not_called()
        mock_logging.error.assert_called()


@pytest.mark.asyncio
async def test_no_jwt_logs_error(mock_dimo):
    """Test that missing JWT is logged as error."""
    with patch("actions.dimo.connector.tesla.logging") as mock_logging:
        config = DIMOTeslaConfig(
            client_id="test_client_id",
            domain="test_domain",
            private_key="test_private_key",
            token_id=123456,
        )
        connector = DIMOTeslaConnector(config)
        connector.vehicle_jwt = None

        input_interface = Mock()
        input_interface.action = "lock doors"

        await connector.connect(input_interface)

        mock_logging.error.assert_called_with("No vehicle jwt")


@pytest.mark.asyncio
async def test_uses_aiohttp_not_requests(tesla_connector):
    """Test that aiohttp is used for non-blocking HTTP calls."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout") as mock_timeout,
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ) as mock_client_session,
    ):
        input_interface = Mock()
        input_interface.action = "lock doors"

        await tesla_connector.connect(input_interface)

        mock_timeout.assert_called_once_with(total=10)
        mock_client_session.assert_called_once()


def test_requests_not_imported():
    """Verify blocking requests library is not imported in tesla module."""
    from actions.dimo.connector import tesla

    module_source = open(tesla.__file__).read()
    assert "import requests" not in module_source
    assert "from requests" not in module_source


@pytest.mark.asyncio
async def test_http_error_status_logs_error(tesla_connector):
    """Test that HTTP error status (non-200) is logged as error."""
    mock_session_cm, mock_session = create_aiohttp_mock(status=500)

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout"),
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
        patch("actions.dimo.connector.tesla.logging") as mock_logging,
    ):
        input_interface = Mock()
        input_interface.action = "lock doors"

        await tesla_connector.connect(input_interface)

        mock_logging.error.assert_called()
        error_call = mock_logging.error.call_args[0][0]
        assert "500" in error_call


@pytest.mark.asyncio
async def test_timeout_is_configured(tesla_connector):
    """Test that aiohttp timeout is properly configured to 10 seconds."""
    mock_session_cm, mock_session = create_aiohttp_mock()

    with (
        patch("actions.dimo.connector.tesla.aiohttp.ClientTimeout") as mock_timeout,
        patch(
            "actions.dimo.connector.tesla.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ) as mock_client_session,
    ):
        input_interface = Mock()
        input_interface.action = "lock doors"

        await tesla_connector.connect(input_interface)

        mock_timeout.assert_called_once_with(total=10)
        mock_client_session.assert_called_once()
        call_kwargs = mock_client_session.call_args[1]
        assert "timeout" in call_kwargs
