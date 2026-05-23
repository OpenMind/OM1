import logging
import time

import httpx


def get_httpx_event_hooks() -> dict[str, list]:
    """
    Return a dictionary of event hooks for logging HTTP requests and responses.

    Returns
    -------
    dict[str, list]
        A dictionary containing lists of event hook functions for 'request' and 'response' events.
    """

    async def log_request(request: httpx.Request):
        """
        Log details of the outgoing HTTP request.

        Parameters
        ----------
        request : httpx.Request
            The HTTP request object to log.
        """
        request.extensions["start_time"] = time.perf_counter()

    async def log_response(response: httpx.Response):
        """
        Log details of the incoming HTTP response, including latency.

        Parameters
        ----------
        response : httpx.Response
            The HTTP response object to log.
        """
        start_time = response.request.extensions.get("start_time", 0)
        elapsed = (time.perf_counter() - start_time) * 1000
        http_version = response.http_version
        proxy_parse_total_time = response.headers.get("x-proxy-parse-ms", "?")
        upstream_total_time = response.headers.get("x-upstream-total-ms", "?")
        upstream_ttfb_time = response.headers.get("x-upstream-ttfb-ms", "?")
        proxy_total_time = response.headers.get("x-proxy-total-ms", "?")
        logging.info(
            f"HTTP {response.request.method} {response.request.url} - "
            f"Status: {response.status_code}, "
            f"HTTP Version: {http_version}, "
            f"Elapsed: {elapsed:.2f} ms, "
            f"Proxy Parse Time: {proxy_parse_total_time} ms, "
            f"Upstream Total Time: {upstream_total_time} ms, "
            f"Upstream TTFB: {upstream_ttfb_time} ms, "
            f"Proxy Total Time: {proxy_total_time} ms"
        )

    return {
        "request": [log_request],
        "response": [log_response],
    }


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
        event_hooks=get_httpx_event_hooks(),
    )
