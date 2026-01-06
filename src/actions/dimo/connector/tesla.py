import logging
import time
from typing import Optional

import requests
from dimo import DIMO
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.dimo.interface import TeslaInput
from providers.io_provider import IOProvider


class DIMOTeslaConfig(ActionConfig):
    """
    Configuration for DIMO Tesla connector.

    Parameters
    ----------
    client_id : Optional[str]
        DIMO client ID.
    domain : Optional[str]
        DIMO domain.
    private_key : Optional[str]
        DIMO private key.
    token_id : Optional[int]
        DIMO token ID.
    """

    client_id: Optional[str] = Field(
        default=None,
        description="DIMO client ID",
    )
    domain: Optional[str] = Field(
        default=None,
        description="DIMO domain",
    )
    private_key: Optional[str] = Field(
        default=None,
        description="DIMO private key",
    )
    token_id: Optional[int] = Field(
        default=None,
        description="DIMO token ID",
    )


class DIMOTeslaConnector(ActionConnector[DIMOTeslaConfig, TeslaInput]):
    """
    Connector that interacts with a Tesla vehicle via the DIMO platform.
    """

    def __init__(self, config: DIMOTeslaConfig):
        """
        Initialize the DIMOTeslaConnector.

        Parameters
        ----------
        config : DIMOTeslaConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.io_provider = IOProvider()

        self.base_url = "https://devices-api.dimo.zone/v1/vehicle"

        self.previous_output = None

        self.token_id = self.io_provider.get_dynamic_variable("token_id")
        self.vehicle_jwt = self.io_provider.get_dynamic_variable("vehicle_jwt")
        self.vehicle_jwt_expires = None

        if not self.token_id or not self.vehicle_jwt:
            self.dimo = DIMO("Production")

            # Configure the DIMO Tesla service
            client_id = self.config.client_id
            domain = self.config.domain
            private_key = self.config.private_key
            self.token_id = self.config.token_id

            if not client_id or not domain or not private_key or not self.token_id:
                raise ValueError(
                    "DIMOTeslaConnector: Missing DIMO configuration in config or IOProvider dynamic variables"
                )

            try:
                auth_header = self.dimo.auth.get_dev_jwt(
                    client_id=client_id, domain=domain, private_key=private_key
                )
                self.dev_jwt = auth_header["access_token"]

                get_vehicle_jwt = self.dimo.token_exchange.exchange(
                    developer_jwt=self.dev_jwt,
                    token_id=self.token_id,
                )
                self.vehicle_jwt = get_vehicle_jwt["token"]
                self.vehicle_jwt_expires = time.time() + 8 * 60
            except Exception as e:
                logging.error(
                    f"DIMOTeslaConnector: Error getting DIMO vehicle jwt: {e}"
                )
                self.vehicle_jwt = None

    async def connect(self, output_interface: TeslaInput) -> None:
        """
        Connect the input protocol to the DIMO Tesla action.

        Parameters
        ----------
        output_interface : TeslaInput
            The input protocol containing the action details.
        """
        logging.info(f"DIMOTeslaConnector: {output_interface.action}")
        if output_interface.action != self.previous_output:
            self.previous_output = output_interface.action

            # checkout timeout of vehicle_jwt
            if (
                self.vehicle_jwt_expires is not None
                and time.time() > self.vehicle_jwt_expires
                and self.token_id is not None
            ):
                try:
                    get_vehicle_jwt = self.dimo.token_exchange.exchange(
                        developer_jwt=self.dev_jwt,
                        token_id=self.token_id,
                    )
                    self.vehicle_jwt = get_vehicle_jwt["token"]
                    self.vehicle_jwt_expires = time.time() + 8 * 60
                except Exception as e:
                    logging.error(
                        f"DIMOTeslaConnector: Error getting DIMO vehicle jwt: {e}"
                    )
                    self.vehicle_jwt = None
                    return None

            if self.vehicle_jwt is not None:
                if output_interface.action == "lock doors":
                    url = f"{self.base_url}/{self.token_id}/commands/doors/lock"
                    response = requests.post(
                        url, headers={"Authorization": f"Bearer {self.vehicle_jwt}"}
                    )
                    if response.status_code == 200:
                        logging.info("DIMO Tesla: Door locked")
                    else:
                        logging.error(
                            f"Error locking door: {response.status_code} {response.text}"
                        )
                elif output_interface.action == "unlock doors":
                    url = f"{self.base_url}/{self.token_id}/commands/doors/unlock"
                    response = requests.post(
                        url, headers={"Authorization": f"Bearer {self.vehicle_jwt}"}
                    )
                    if response.status_code == 200:
                        logging.info("DIMO Tesla: Door unlocked")
                    else:
                        logging.error(
                            f"Error unlocking door: {response.status_code} {response.text}"
                        )
                elif output_interface.action == "open frunk":
                    url = f"{self.base_url}/{self.token_id}/commands/frunk/open"
                    response = requests.post(
                        url, headers={"Authorization": f"Bearer {self.vehicle_jwt}"}
                    )
                    if response.status_code == 200:
                        logging.info("DIMO Tesla: Frunk opened")
                    else:
                        logging.error(
                            f"Error opening frunk: {response.status_code} {response.text}"
                        )
                elif output_interface.action == "open trunk":
                    url = f"{self.base_url}/{self.token_id}/commands/trunk/open"
                    response = requests.post(
                        url, headers={"Authorization": f"Bearer {self.vehicle_jwt}"}
                    )
                    if response.status_code == 200:
                        logging.info("DIMO Tesla: Trunk opened")
                    else:
                        logging.error(
                            f"Error opening trunk: {response.status_code} {response.text}"
                        )
                elif output_interface.action == "idle":
                    logging.info("DIMO Tesla: Idle")
                else:
                    logging.error(f"Unknown action: {output_interface.action}")
            else:
                logging.error("No vehicle jwt")
