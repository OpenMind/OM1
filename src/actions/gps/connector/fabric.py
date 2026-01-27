import logging

import requests
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.gps.interface import GPSAction, GPSInput
from providers.io_provider import IOProvider


class GPSFabricConfig(ActionConfig):
    """
    Configuration for GPS Fabric connector.

    Parameters
    ----------
    fabric_endpoint : str
        The endpoint URL for the Fabric network.
    request_timeout : int
        Timeout in seconds for HTTP requests.
    """

    fabric_endpoint: str = Field(
        default="http://localhost:8545",
        description="The endpoint URL for the Fabric network.",
    )
    request_timeout: int = Field(
        default=10,
        description="Timeout in seconds for HTTP requests.",
    )


class GPSFabricConnector(ActionConnector[GPSFabricConfig, GPSInput]):
    """
    Connector that shares GPS coordinates via a Fabric network.
    """

    def __init__(self, config: GPSFabricConfig):
        """
        Initialize the GPSFabricConnector.

        Parameters
        ----------
        config : GPSFabricConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        # Set IO Provider
        self.io_provider = IOProvider()

        # Set fabric endpoint configuration
        self.fabric_endpoint = self.config.fabric_endpoint
        self.request_timeout = self.config.request_timeout

    async def connect(self, output_interface: GPSInput) -> None:
        """
        Connect to the Fabric network and send GPS coordinates.

        Parameters
        ----------
        output_interface : GPSInput
            The GPS input containing the action to be performed.
        """
        logging.info(f"GPSFabricConnector: {output_interface.action}")

        if output_interface.action == GPSAction.SHARE_LOCATION:
            # Send GPS coordinates to the Fabric network
            self.send_coordinates()

    def send_coordinates(self) -> bool:
        """
        Send GPS coordinates to the Fabric network.

        Returns
        -------
        bool
            True if coordinates were sent successfully, False otherwise.
        """
        logging.info("GPSFabricConnector: Sending coordinates to Fabric network.")
        latitude = self.io_provider.get_dynamic_variable("latitude")
        longitude = self.io_provider.get_dynamic_variable("longitude")
        yaw = self.io_provider.get_dynamic_variable("yaw_deg")
        logging.info(f"GPSFabricConnector: Latitude: {latitude}")
        logging.info(f"GPSFabricConnector: Longitude: {longitude}")
        logging.info(f"GPSFabricConnector: Yaw: {yaw}")

        # FIX: Changed 'and' to 'or' - validates if ANY coordinate is None
        if latitude is None or longitude is None or yaw is None:
            logging.error(
                "GPSFabricConnector: Coordinates not available. "
                f"latitude={latitude}, longitude={longitude}, yaw={yaw}"
            )
            return False

        try:
            response = requests.post(
                f"{self.fabric_endpoint}",
                json={
                    "method": "omp2p_shareStatus",
                    "params": [
                        {"latitude": latitude, "longitude": longitude, "yaw": yaw}
                    ],
                    "id": 1,
                    "jsonrpc": "2.0",
                },
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )

            # FIX: Added HTTP status code validation
            if not response.ok:
                logging.error(
                    f"GPSFabricConnector: HTTP error {response.status_code}: "
                    f"{response.text}"
                )
                return False

            # FIX: Added JSON parsing error handling
            try:
                response_data = response.json()
            except requests.exceptions.JSONDecodeError as e:
                logging.error(
                    f"GPSFabricConnector: Failed to parse JSON response: {e}"
                )
                return False

            # FIX: Added JSON-RPC error field check
            if "error" in response_data:
                error = response_data["error"]
                logging.error(
                    f"GPSFabricConnector: JSON-RPC error: "
                    f"code={error.get('code')}, message={error.get('message')}"
                )
                return False

            if "result" in response_data and response_data["result"]:
                logging.info("GPSFabricConnector: Coordinates shared successfully.")
                return True
            else:
                logging.error("GPSFabricConnector: Failed to share coordinates.")
                return False

        except requests.Timeout:
            logging.error(
                f"GPSFabricConnector: Request timed out after "
                f"{self.request_timeout} seconds"
            )
            return False
        except requests.ConnectionError as e:
            logging.error(f"GPSFabricConnector: Connection error: {e}")
            return False
        except requests.RequestException as e:
            logging.error(f"GPSFabricConnector: Error sending coordinates: {e}")
            return False
