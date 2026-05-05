import json
import logging
import os
from typing import Optional, Union

import zenoh

from zenoh_msgs.cloud_session.zenoh_adapter import CloudSimZenohSession, _Sample

ZenohSessionType = Union[zenoh.Session, CloudSimZenohSession]
ZenohSampleType = Union[zenoh.Sample, "_Sample"]


def create_zenoh_config(network_discovery: bool = True) -> zenoh.Config:
    """
    Create a Zenoh configuration for a client connecting to a Zenoh router.

    The connect endpoint defaults to tcp/127.0.0.1:7447 (a local router on
    the same host) but can be overridden via OM1_ZENOH_ENDPOINT — for example
    "wss/test-sim.openmind.com:8444" to reach a remote Zenoh router over a
    TLS-terminated WebSocket. For self-signed test deployments,
    OM1_ZENOH_TLS_ROOT_CA can point at the trusted CA cert file path.

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
        endpoint = os.environ.get("OM1_ZENOH_ENDPOINT", "tcp/127.0.0.1:7447")
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))

        if endpoint.startswith(("wss/", "tls/", "quic/")):
            ca_path = os.environ.get("OM1_ZENOH_TLS_ROOT_CA")
            if ca_path:
                config.insert_json5(
                    "transport/link/tls/root_ca_certificate",
                    json.dumps(ca_path),
                )

    return config


def open_zenoh_session() -> ZenohSessionType:
    """
    Open a Zenoh session.

    If OPENMIND_CLOUD_URL is set, return an OpenMindZenohSession that
    looks like a zenoh.Session to OM1 plugins but routes pub/sub through
    the OpenMind cloud broker. This is the pattern customers use in the
    cloud product.

    Otherwise, open a normal Zenoh client (local first, then network
    discovery) — same behavior as before.

    Returns
    -------
    ZenohSessionType
        Either a zenoh.Session or CloudSimZenohSession - both are compatible.
    """
    if os.environ.get("USE_SIM") == "true":
        return _open_cloud_session()

    local_config = create_zenoh_config(network_discovery=False)
    try:
        session = zenoh.open(local_config)
        logging.info("Zenoh client opened without network discovery")
        return session
    except Exception:
        logging.info("Falling back to network discovery...")

    config = create_zenoh_config()
    try:
        session = zenoh.open(config)
        logging.info("Zenoh client opened with network discovery")
        return session
    except Exception as e:
        logging.error(f"Error opening Zenoh client: {e}")
        raise Exception("Failed to open Zenoh session") from e


def _open_cloud_session(
    url: str = "https://api.openmind.com/api/core/simulation/zenoh", token: Optional[str] = None
) -> CloudSimZenohSession:
    """
    Open a CloudSimZenohSession using the URL and token from environment variables.
    """
    return CloudSimZenohSession(url, token=token)


if __name__ == "__main__":
    session = open_zenoh_session()
    if session:
        logging.info("Session opened successfully")
        session.close()
    else:
        logging.error("Failed to open Zenoh session")
