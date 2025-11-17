import asyncio
import logging
import time
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional

from dimo import DIMO

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


@dataclass
class Message:
    """
    Container for timestamped messages.

    Parameters
    ----------
    timestamp : float
        Unix timestamp of the message
    message : str
        Content of the message
    """

    timestamp: float
    message: str


class DIMOTesla(FuserInput[str]):
    """
    DIMO Tesla input handler.

    A class that process Tesla data and generates text descriptions
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize DIMO Tesla input handler.

        Sets up the required providers and buffers for handling Tesla data processing.
        Initializes connection to the Tesla service and registers message handlers.
        """
        super().__init__(config)

        self.descriptor_for_LLM = "Tesla Data"

        # Track IO
        self.io_provider = IOProvider()

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        self.dimo = DIMO("Production")

        self.vehicle_jwt = None

        # Configure the DIMO Tesla service
        client_id = getattr(config, "client_id", None)
        domain = getattr(config, "domain", None)
        private_key = getattr(config, "private_key", None)

        if client_id is None or client_id == "" or domain is None or private_key is None:
            logging.info("DIMOTesla: You did not provide credentials to your Tesla - aborting")
            return

        self.token_id = getattr(config, "token_id", None)
        if self.token_id is None:
            logging.info("DIMOTesla: You did not provide a token_id - aborting")
            return

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

            # bypass token_id and vehicle_jwt to io_provider
            self.io_provider.add_dynamic_variable("token_id", self.token_id)
            self.io_provider.add_dynamic_variable("vehicle_jwt", self.vehicle_jwt)
        except Exception as e:
            logging.error(f"DIMOTesla: INIT - Error getting DIMO vehicle jwt: {e}")
            self.vehicle_jwt = None

    async def _poll(self) -> Optional[str]:
        """
        Poll for Tesla data.

        Returns
        -------
        str
            The latest Tesla data
        """
        await asyncio.sleep(0.5)

        if self.vehicle_jwt is None:
            logging.error("DIMOTesla: No vehicle jwt - did you provide credentials?")
            return None

        if self.vehicle_jwt_expires is None:
            return None

        if time.time() > self.vehicle_jwt_expires:
            try:
                get_vehicle_jwt = self.dimo.token_exchange.exchange(
                    developer_jwt=self.dev_jwt,
                    token_id=self.token_id,  # type: ignore
                )
                self.vehicle_jwt = get_vehicle_jwt["token"]
                self.vehicle_jwt_expires = time.time() + 8 * 60
            except Exception as e:
                logging.error(f"DIMOTesla: Error getting DIMO vehicle jwt: {e}")
                self.vehicle_jwt = None
                return None

        latest_tesla_signals = self.dimo.query(
            "Telemetry",
            query="""
                    query GetSignalsLatest($tokenId: Int!) {
                        signalsLatest(tokenId: $tokenId){
                            powertrainTransmissionTravelledDistance{
                                timestamp
                                value
                            }
                            exteriorAirTemperature{
                                timestamp
                                value
                            }
                            speed {
                                timestamp
                                value
                            }
                            powertrainRange{
                                timestamp
                                value
                            }
                            currentLocationLatitude{
                                timestamp
                                value
                            }
                            currentLocationLongitude{
                                timestamp
                                value
                            }
                        }
                    }
            """,
            token=self.vehicle_jwt,
            variables={"tokenId": self.token_id},
        )

        try:
            tesla_data = latest_tesla_signals["data"]["signalsLatest"]  # type: ignore
            powertrainTransmissionTravelledDistance = tesla_data[
                "powertrainTransmissionTravelledDistance"
            ]["value"]
            exteriorAirTemperature = tesla_data["exteriorAirTemperature"]["value"]
            speed = tesla_data["speed"]["value"]
            powertrainRange = tesla_data["powertrainRange"]["value"]
            currentLocationLatitude = tesla_data["currentLocationLatitude"]["value"]
            currentLocationLongitude = tesla_data["currentLocationLongitude"]["value"]
        except Exception as e:
            print("Error parsing Tesla data")
            logging.error(f"Error parsing Tesla data: {e}")
            return None

        return """
        Powertrain Transmission Travelled Distance: {powertrainTransmissionTravelledDistance} km
        Exterior Air Temperature: {exteriorAirTemperature} C
        Speed: {speed} km/h
        Powertrain Range: {powertrainRange} km
        Current Location Latitude: {currentLocationLatitude}
        Current Location Longitude: {currentLocationLongitude}
        """.format(
            powertrainTransmissionTravelledDistance=powertrainTransmissionTravelledDistance,
            exteriorAirTemperature=exteriorAirTemperature,
            speed=speed,
            powertrainRange=powertrainRange,
            currentLocationLatitude=currentLocationLatitude,
            currentLocationLongitude=currentLocationLongitude,
        )

    async def _raw_to_text(self, raw_input: str) -> Message:
        """
        Process raw input to generate a timestamped message.

        Returns
        -------
        Message
            Timestamped status or transaction notification
        """
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: str):
        """
        Process Tesla data message buffer.

        """
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message)
            # only update if there has been a change
            elif self.messages[-1].message != pending_message.message:
                self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        return result
