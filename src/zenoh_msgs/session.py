import atexit
import logging
import weakref

import zenoh

logging.basicConfig(level=logging.INFO)

_active_sessions = weakref.WeakSet()


def _cleanup_all_sessions():
    """
    Cleanup all active sessions on program exit.
    """
    sessions_to_close = list(_active_sessions)
    for session in sessions_to_close:
        try:
            if hasattr(session, "close"):
                session.close()
                logging.info("Zenoh session closed automatically on exit")
        except Exception as e:
            logging.warning(f"Error during exit cleanup: {e}")


atexit.register(_cleanup_all_sessions)


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
    Open a Zenoh session with automatic cleanup.

    Returns
    -------
    zenoh.Session
        The opened Zenoh session with automatic cleanup.

    Raises
    ------
    Exception
        If unable to open a Zenoh session.
    """
    local_config = create_zenoh_config(network_discovery=False)

    try:
        session = zenoh.open(local_config)
        logging.info("Zenoh client opened without network discovery")
    except Exception as e:
        logging.warning(f"Local connection failed: {e}")
        logging.info("Falling back to network discovery...")

        config = create_zenoh_config()
        try:
            session = zenoh.open(config)
            logging.info("Zenoh client opened with network discovery")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")
            raise Exception("Failed to open Zenoh session") from e

    _active_sessions.add(session)

    return session


if __name__ == "__main__":
    session = open_zenoh_session()
    logging.info("Session opened successfully with automatic cleanup")
