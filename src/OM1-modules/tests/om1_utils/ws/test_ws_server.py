import asyncio
import random
from unittest.mock import Mock

import pytest
import websockets

from om1_utils.ws import Server


def get_free_port():
    """Get a random port number between 6790 and 7000."""
    return random.randint(6790, 7000)


async def wait_for_server(host, port, timeout=5):
    """Wait for server to become available."""
    start_time = asyncio.get_event_loop().time()
    while True:
        try:
            async with websockets.connect(f"ws://{host}:{port}", close_timeout=0.1):
                return True
        except Exception:
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False
            await asyncio.sleep(0.1)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def server_config():
    """Provide server configuration."""
    return {"host": "localhost", "port": get_free_port()}


@pytest.fixture
async def server(server_config):
    """Fixture to create and clean up a server instance."""
    server_instance = Server(host=server_config["host"], port=server_config["port"])
    server_instance.start()

    # Wait for server to be ready
    is_ready = await wait_for_server(server_config["host"], server_config["port"])
    if not is_ready:
        pytest.fail("Server failed to start")

    yield server_instance

    server_instance.stop()
    await asyncio.sleep(0.2)


@pytest.fixture
async def websocket_client(server, server_config):
    """Fixture to create a WebSocket client connection."""
    uri = f"ws://{server_config['host']}:{server_config['port']}"

    async with websockets.connect(uri, close_timeout=0.1) as websocket:
        await asyncio.sleep(0.2)  # Wait for connection to establish
        yield websocket


@pytest.mark.asyncio
async def test_server_start_stop(server_config):
    """Test basic server start and stop functionality."""
    server = Server(host=server_config["host"], port=server_config["port"])
    server.start()
    assert server.running is True

    is_ready = await wait_for_server(server_config["host"], server_config["port"])
    assert is_ready is True

    server.stop()
    assert server.running is False
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_client_connection(server_config):
    """Test client connection and disconnection."""
    server = Server(host=server_config["host"], port=server_config["port"])
    server.start()

    is_ready = await wait_for_server(server_config["host"], server_config["port"])
    assert is_ready is True

    connection_callback = Mock()
    server.register_connection_callback(connection_callback)

    uri = f"ws://{server_config['host']}:{server_config['port']}"
    async with websockets.connect(uri, close_timeout=0.1):
        await asyncio.sleep(0.2)
        assert server.has_connections() is True
        assert len(server.connections) == 1
        connection_callback.assert_called()

    await asyncio.sleep(0.5)
    assert server.has_connections() is False
    assert connection_callback.call_count >= 2

    server.stop()
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_message_callback(server_config):
    """Test message callback functionality."""
    server = Server(host=server_config["host"], port=server_config["port"])
    server.start()
    await asyncio.sleep(0.2)

    is_ready = await wait_for_server(server_config["host"], server_config["port"])
    assert is_ready is True

    try:
        uri = f"ws://{server_config['host']}:{server_config['port']}"
        async with websockets.connect(uri, close_timeout=0.1) as websocket:
            await asyncio.sleep(0.2)  # Wait for connection to establish

            # Now we should have a connection
            assert len(server.connections) > 0, "No active connections found"
            connection_id = next(iter(server.connections.keys()))

            message_callback = Mock()
            server.register_message_callback(connection_id, message_callback)

            test_message = "Hello, Server!"
            await websocket.send(test_message)
            await asyncio.sleep(0.2)

            message_callback.assert_called_once()
            args = message_callback.call_args[0]
            assert args[0] == connection_id
            assert args[1] == test_message
    finally:
        server.stop()
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_handle_response(server_config):
    """Test sending messages to specific clients."""
    server = Server(host=server_config["host"], port=server_config["port"])
    server.start()
    await asyncio.sleep(0.2)

    is_ready = await wait_for_server(server_config["host"], server_config["port"])
    assert is_ready is True

    try:
        uri = f"ws://{server_config['host']}:{server_config['port']}"
        async with websockets.connect(uri, close_timeout=0.1) as websocket:
            await asyncio.sleep(0.2)  # Wait for connection to establish

            # Now we should have a connection
            assert len(server.connections) > 0, "No active connections found"
            connection_id = next(iter(server.connections.keys()))

            test_message = "Hello, Client!"
            server.handle_response(connection_id, test_message)

            received = await websocket.recv()
            assert received == test_message
    finally:
        server.stop()
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_handle_global_response(server_config):
    """Test broadcasting messages to all clients."""
    server = Server(host=server_config["host"], port=server_config["port"])
    server.start()

    is_ready = await wait_for_server(server_config["host"], server_config["port"])
    assert is_ready is True

    uri = f"ws://{server_config['host']}:{server_config['port']}"
    try:
        async with (
            websockets.connect(uri, close_timeout=0.1) as client1,
            websockets.connect(uri, close_timeout=0.1) as client2,
        ):
            await asyncio.sleep(0.2)

            test_message = "Broadcast Message"
            server.handle_global_response(test_message)

            received1 = await asyncio.wait_for(client1.recv(), timeout=1.0)
            received2 = await asyncio.wait_for(client2.recv(), timeout=1.0)

            assert received1 == test_message
            assert received2 == test_message
    finally:
        server.stop()
        await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_connection_closed_handling(server_config):
    """Test proper handling of connection closure."""
    server = Server(host=server_config["host"], port=server_config["port"])
    server.start()

    is_ready = await wait_for_server(server_config["host"], server_config["port"])
    assert is_ready is True

    connection_callback = Mock()
    server.register_connection_callback(connection_callback)

    uri = f"ws://{server_config['host']}:{server_config['port']}"
    async with websockets.connect(uri, close_timeout=0.1):
        await asyncio.sleep(0.2)
        initial_connections = len(server.connections)
        assert initial_connections > 0

    await asyncio.sleep(0.5)
    assert len(server.connections) == initial_connections - 1
    assert connection_callback.call_count >= 2

    server.stop()
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_format_message():
    """Test message formatting functionality."""
    server = Server()

    normal_msg = "Hello, World!"
    assert server.format_message(normal_msg) == normal_msg

    long_msg = "x" * 300
    formatted = server.format_message(long_msg, max_length=100)
    assert len(formatted) <= 100
    assert "..." in formatted
