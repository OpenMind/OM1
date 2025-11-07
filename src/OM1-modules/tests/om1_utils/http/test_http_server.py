import socket
import time
from contextlib import closing
from typing import Generator, Tuple
from unittest.mock import Mock

import pytest
import requests

from om1_utils import http


def find_free_port() -> int:
    """Find a free port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        return port


def wait_for_server(url: str, timeout: int = 10, interval: float = 0.2) -> bool:
    """Wait for server to be ready to accept connections with increased timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            requests.post(url, json={"ping": "test"}, timeout=1.0)
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(interval)
    return False


@pytest.fixture
def server_config() -> Tuple[str, int]:
    """Fixture that provides server host and a free port."""
    host = "127.0.0.1"
    port = find_free_port()
    return host, port


@pytest.fixture
def server(server_config: Tuple[str, int]) -> Generator[http.Server, None, None]:
    """Fixture that creates and manages a test server instance."""
    host, port = server_config
    test_server = http.Server(host=host, port=port)

    try:
        test_server.start()
        server_url = f"http://{host}:{port}"

        if not wait_for_server(server_url):
            test_server.stop()
            pytest.fail(
                f"Server failed to start on {host}:{port} within timeout period"
            )

        yield test_server

    finally:
        test_server.stop()
        # Wait for server to fully shut down with increased timeout
        time.sleep(1.0)


@pytest.fixture
def server_url(server_config: Tuple[str, int]) -> str:
    """Fixture that returns the test server URL."""
    host, port = server_config
    return f"http://{host}:{port}"


def test_server_initialization(server_config: Tuple[str, int]):
    """Test server initialization with custom parameters."""
    host, port = server_config
    server = http.Server(host=host, port=port, timeout=30)
    assert server.host == host
    assert server.port == port
    assert server.timeout == 30
    assert server.running is True
    assert server.server is None
    assert server.message_callback is None


def test_callback_registration(server: http.Server):
    """Test registering a message callback."""
    mock_callback = Mock(return_value={"status": "ok"})
    server.register_message_callback(mock_callback)
    assert server.message_callback == mock_callback


def test_post_request_with_callback(server: http.Server, server_url: str):
    """Test handling a POST request with a registered callback."""
    # Register mock callback
    mock_response = {"status": "success", "data": "test"}
    mock_callback = Mock(return_value=mock_response)
    server.register_message_callback(mock_callback)

    # Send test request
    test_data = {"message": "test"}
    response = requests.post(f"{server_url}/test", json=test_data, timeout=5)

    assert response.status_code == 200
    assert response.json() == mock_response
    mock_callback.assert_called_once_with(test_data, "/test")


def test_get_request_not_allowed(server: http.Server, server_url: str):
    """Test that GET requests are not allowed."""
    response = requests.get(server_url, timeout=5)
    assert response.status_code == 405
    assert "error" in response.json()


def test_post_request_without_callback(server: http.Server, server_url: str):
    """Test handling a POST request without a registered callback."""
    response = requests.post(server_url, json={"test": "data"}, timeout=5)
    assert response.status_code == 500
    assert response.json()["error"] == "No callback handler registered"


def test_post_request_invalid_json(server: http.Server, server_url: str):
    """Test handling a POST request with invalid JSON."""
    response = requests.post(
        server_url,
        data="invalid json",
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_callback_string_response(server: http.Server, server_url: str):
    """Test handling a string response from the callback."""
    mock_callback = Mock(return_value="test response")
    server.register_message_callback(mock_callback)

    response = requests.post(server_url, json={"test": "data"}, timeout=5)

    assert response.status_code == 200
    assert response.json() == {"response": "test response"}


def test_server_stop(server_config: Tuple[str, int]):
    """Test server stop functionality."""
    host, port = server_config
    server = http.Server(host=host, port=port)

    try:
        server.start()
        server_url = f"http://{host}:{port}"

        # Wait for server to start
        if not wait_for_server(server_url):
            pytest.fail(f"Server failed to start on {host}:{port}")

        # Verify server is running
        response = requests.post(server_url, json={"test": "data"}, timeout=1)
        assert response.status_code in (
            400,
            500,
        )  # Either JSON error or no callback error

        # Stop the server
        server.stop()
        assert server.running is False

        # Verify server is no longer accepting connections
        with pytest.raises(
            (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout)
        ):
            requests.post(server_url, json={"test": "data"}, timeout=1)

    finally:
        if server.running:
            server.stop()
        time.sleep(1.0)


@pytest.mark.parametrize(
    "test_input,expected_code",
    [
        ({"valid": "data"}, 200),
        (b"invalid data", 400),
        (None, 400),
    ],
)
def test_various_inputs(
    server: http.Server, server_url: str, test_input: http.JsonDict, expected_code: int
):
    """Test handling various types of input data."""
    mock_callback = Mock(return_value={"status": "ok"})
    server.register_message_callback(mock_callback)

    if isinstance(test_input, dict):
        response = requests.post(server_url, json=test_input, timeout=5)
    else:
        response = requests.post(server_url, data=test_input, timeout=5)

    assert response.status_code == expected_code
