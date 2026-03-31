"""
Zenoh session management utilities.

Provides configuration and session initialization with automatic fallback
from local connection to network discovery.
"""
import logging

import zenoh

logging.basicConfig(level=logging.INFO)


def create_zenoh_config(network_discovery: bool = True) -> zenoh.Config:
    """
    Create a Zenoh configuration for a client connecting to a local server.

    Parameters
    ----------
    network_discovery : bool, optional
        Whether to enable network discovery (default is True).

    Returns
    -------
    zenoh.Config
        The Zenoh configuration object.
    """
    config = zenoh.Config()
    if not network_discovery:
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')

    return config


def open_zenoh_session() -> zenoh.Session:
    """
    Open a Zenoh session with a local connection first, then fall back to network discovery.

    Returns
    -------
    zenoh.Session
        The opened Zenoh session.

    Raises
    ------
    Exception
        If unable to open a Zenoh session.
    """
    local_config = create_zenoh_config(network_discovery=False)
    try:
        local_session = zenoh.open(local_config)
        logging.info("Zenoh client opened without network discovery")
        return local_session
    except Exception as local_err:
        logging.warning(
            f"Local Zenoh connection failed (endpoint: tcp/127.0.0.1:7447): {local_err}. "
            "Attempting network discovery fallback..."
        )

    config = create_zenoh_config()
    try:
        discovery_session = zenoh.open(config)
        logging.info("Zenoh client opened with network discovery")
        return discovery_session
    except Exception as discovery_err:
        logging.error(
            f"Zenoh session initialization failed. "
            f"Local connection failed, and network discovery also failed: {discovery_err}. "
            f"Check Zenoh router status and network connectivity."
        )
        raise Exception("Failed to open Zenoh session") from discovery_err


if __name__ == "__main__":
    session = open_zenoh_session()
    if session:
        logging.info("Session opened successfully")
        session.close()
    else:
        logging.error("Failed to open Zenoh session")
