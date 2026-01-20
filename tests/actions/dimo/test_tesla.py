from unittest.mock import Mock, patch

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
    """
    Test that 'lock doors' command works regardless of case.
    """
    with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
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
    with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
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
    with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
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
    with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
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
    """Test that 'idle' command works regardless of case."""
    with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
        input_interface = Mock()
        input_interface.action = action_input

        await tesla_connector.connect(input_interface)

        mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_action_logs_error(tesla_connector):
    """Test that unknown actions are logged as errors."""
    with patch("actions.dimo.connector.tesla.requests.post") as mock_post:
        with patch("actions.dimo.connector.tesla.logging") as mock_logging:
            input_interface = Mock()
            input_interface.action = "invalid_action"

            await tesla_connector.connect(input_interface)

            mock_post.assert_not_called()
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
