"""Privacy Guard WebSocket connector.

Sends privacy action commands over WebSocket for the WebSim visualizer
and remote operation scenarios. Use this connector when no physical
robot is available (development / testing mode).
"""

import asyncio
import json
import logging

from actions.base import ActionConfig, ActionConnector
from pydantic import Field

logger = logging.getLogger(__name__)


class PrivacyGuardWSConfig(ActionConfig):
    """Configuration for the WebSocket-based Privacy Guard connector."""

    look_away_angle: int = Field(
        default=90,
        description="Degrees to tilt camera/head down on privacy breach",
    )
    host: str = Field(default="localhost", description="WebSocket host")
    port: int = Field(default=8000, description="WebSocket port")


class PrivacyGuardWebSocketConnector(ActionConnector["PrivacyGuardWSConfig", "PrivacyGuardInput"]):
    """Sends privacy guard actions over WebSocket to the WebSim frontend."""

    def __init__(self, config: PrivacyGuardWSConfig):
        super().__init__(config)
        self._is_looking_away = False

    async def _send_payload(self, payload: dict) -> None:
        """Send a JSON payload to the WebSim server via WebSocket."""
        uri = f"ws://{self.config.host}:{self.config.port}/ws"
        try:
            import websockets

            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps(payload))
                logger.info(f"WebSocket payload sent to {uri}")
        except ImportError:
            logger.warning(
                "websockets package not installed — falling back to log-only mode. "
                "Install with: pip install websockets"
            )
        except Exception as e:
            logger.error(f"WebSocket send failed ({uri}): {e}")

    async def connect(self, output_interface) -> None:
        """Serialize and send the privacy action via WebSocket.

        Parameters
        ----------
        output_interface : PrivacyGuardInput
            The privacy action to execute.
        """
        action = output_interface.action
        angle = self.config.look_away_angle

        payload = {
            "type": "privacy_guard",
            "action": action,
            "params": {},
        }

        if action == "look_away":
            self._is_looking_away = True
            payload["params"] = {
                "angle": angle,
                "message": "Privacy Breach Detected: Tilting down.",
            }
            payload["effort_score"] = "max"
            logger.warning(f"PRIVACY BREACH — Sending look_away ({angle}°) via WebSocket")

        elif action == "resume":
            if self._is_looking_away:
                self._is_looking_away = False
                payload["params"] = {
                    "angle": 0,
                    "message": "Threat cleared. Resuming.",
                }
                payload["effort_score"] = "low"
                logger.info("Privacy threat cleared — Sending resume via WebSocket")
            else:
                payload["params"] = {"message": "Already in neutral position."}
                payload["effort_score"] = "low"

        elif action == "stop_stream":
            self._is_looking_away = True
            payload["params"] = {
                "angle": angle,
                "stream_killed": True,
                "message": "Critical breach — Video feed terminated.",
            }
            payload["effort_score"] = "max"
            logger.critical("CRITICAL BREACH — Sending stop_stream via WebSocket")

        elif action == "idle":
            payload["params"] = {"message": "Environment secure."}
            payload["effort_score"] = "low"

        await self._send_payload(payload)
