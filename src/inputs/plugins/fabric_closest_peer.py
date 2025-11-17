import asyncio
import logging
import time
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional

# `requests` is **optional** while we mock — keep import guarded
try:
    import requests  # type: ignore
except ImportError:  # unit‑test env without requests installed
    requests = None  # pylint: disable=invalid-name

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Message:
    timestamp: float
    message: str


class FabricClosestPeer(FuserInput[str]):
    """Share our GPS position with the Fabric network and fetch the closest peer.

    **Mock‑friendly:** set `mock_mode=True` in the plugin config (or environment)
    and the connector will fabricate a plausible peer 30‑50 m away instead of
    calling the REST endpoint.  Useful for local testing before the chain is up.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        super().__init__(config)

        self.descriptor_for_LLM = "Closest Peer from Fabric"
        self.io = IOProvider()
        self.messages: List[str] = []
        self.msg_q: Queue[str] = Queue()

        # endpoint / mock toggle -------------------------------------------------
        self.fabric_endpoint = getattr(
            config, "fabric_endpoint", "http://localhost:8545"
        )
        self.mock_mode: bool = bool(
            getattr(config, "mock_mode", True)
        )  # default ON for now

    # ────────────────────────────────────────────────────────────────────────
    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.5)

        # --------------------------------------------------------------------
        if self.mock_mode:
            peer_lat = getattr(self.config, "mock_lat")
            peer_lon = getattr(self.config, "mock_lon")
            logging.info(
                f"FabricClosestPeer (mock): fabricated peer {peer_lat:.6f},{peer_lon:.6f}"
            )
        else:
            if requests is None:
                logging.error(
                    "FabricClosestPeer: requests not available and mock_mode=False"
                )
                return None
            try:
                lat = self.io.get_dynamic_variable("latitude")
                lon = self.io.get_dynamic_variable("longitude")
                if lat is None or lon is None:
                    logging.error("FabricClosestPeer: latitude or longitude not set.")
                    return None
                logging.info(
                    f"FabricClosestPeer: fetching closest peer for {lat:.6f}, {lon:.6f}"
                )
                resp = requests.post(
                    self.fabric_endpoint,
                    json={
                        "method": "omp2p_findClosestPeer",
                        "params": [{"latitude": lat, "longitude": lon}],
                        "id": 1,
                        "jsonrpc": "2.0",
                    },
                    timeout=3.0,
                    headers={"Content-Type": "application/json"},
                )
                data = resp.json()
                logging.debug(f"FabricClosestPeer response: {data}")
                peer_info = (data.get("result") or [{}])[0].get("peer")
                if not peer_info:
                    logging.info("FabricClosestPeer: no peer found.")
                    return None
                peer_lat = peer_info["latitude"]
                peer_lon = peer_info["longitude"]
            except Exception as exc:  # pylint: disable=broad-except
                logging.error(
                    f"FabricClosestPeer: error calling Fabric endpoint – {exc}"
                )
                return None

        # store & enqueue ------------------------------------------------------
        self.io.add_dynamic_variable("closest_peer_lat", peer_lat)
        self.io.add_dynamic_variable("closest_peer_lon", peer_lon)

        human_msg = f"Closest peer at {peer_lat:.5f}, {peer_lon:.5f}"
        self.msg_q.put(human_msg)
        return human_msg

    # ────────────────────────────────────────────────────────────────────────
    async def raw_to_text(self, raw_input: Optional[str]):
        if raw_input is None:
            return
        self.messages.append(raw_input)

    def formatted_latest_buffer(self) -> Optional[str]:
        if not self.msg_q.qsize():
            return None
        msg = self.msg_q.get()
        self.io.add_input(self.descriptor_for_LLM, msg, time.time())
        return f"""
{self.descriptor_for_LLM} INPUT
// START
{msg}
// END
"""
