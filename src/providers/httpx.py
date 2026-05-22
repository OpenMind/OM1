import httpx


def get_async_httpx_client(
    http2: bool = True,
    max_keepalive_connections: int = 20,
    max_connections: int = 100,
    keepalive_expiry: float = 300.0,
    timeout: float = 60.0,
    connect_timeout: float = 5.0,
) -> httpx.AsyncClient:
    """
    Create and return a configured AsyncClient for HTTP requests.

    Parameters
    ----------
    http2 : bool
        Whether to enable HTTP/2 support.
    max_keepalive_connections : int
        Maximum number of keep-alive connections to maintain.
    max_connections : int
        Maximum number of concurrent connections allowed.
    keepalive_expiry : float
        Time in seconds before idle keep-alive connections are closed.
    timeout : float
        Total timeout for requests in seconds.
    connect_timeout : float
        Timeout for establishing a connection in seconds.

    Returns
    -------
    httpx.AsyncClient
        A configured AsyncClient instance ready for making HTTP requests.
    """
    return httpx.AsyncClient(
        http2=http2,
        limits=httpx.Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
            keepalive_expiry=keepalive_expiry,
        ),
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
    )
