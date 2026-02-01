import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import requests

from .singleton import singleton


@dataclass
class RFData:
    """
    Data class to represent RF scan results.

    Parameters
    ----------
    unix_ts : float
        Unix timestamp of the scan.
    address : str
        Bluetooth address of the device.
    name : Optional[str]
        Name of the device.
    rssi : int
        Received Signal Strength Indicator of the device.
    tx_power : Optional[int]
        Transmission power.
    service_uuid : str
        Service UUID.
    mfgkey : str
        Manufacturing key.
    mfgval : str
        Manufacturing value.
    """

    unix_ts: float
    address: str
    name: Optional[str]
    rssi: int
    tx_power: Optional[int]
    service_uuid: str
    mfgkey: str
    mfgval: str

    def to_dict(self) -> dict:
        """
        Convert the RFData object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the RFData object.
        """
        return {
            "unix_ts": self.unix_ts,
            "address": self.address,
            "name": self.name,
            "rssi": self.rssi,
            "tx_power": self.tx_power,
            "service_uuid": self.service_uuid,
            "mfgkey": self.mfgkey,
            "mfgval": self.mfgval,
        }


@dataclass
class RFDataRaw:
    """
    Data class to represent RF scan results.

    Parameters
    ----------
    unix_ts : float
        Unix timestamp of the scan.
    address : str
        Bluetooth address of the device.
    rssi : int
        Received Signal Strength Indicator of the device.
    packet : str
        Packet data.
    """

    unix_ts: float
    address: str
    rssi: int
    packet: str

    def to_dict(self) -> dict:
        """
        Convert the RFData object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the RFData object.
        """
        return {
            "unix_ts": self.unix_ts,
            "address": self.address,
            "rssi": self.rssi,
            "packet": self.packet,
        }


@dataclass
class FabricData:
    """
    Data class to represent a data payload from a machine to FABRIC.
    """

    machine_id: str
    payload_idx: int
    gps_unix_ts: float
    gps_lat: float
    gps_lon: float
    gps_alt: float
    gps_qua: int
    rtk_unix_ts: float
    rtk_lat: float
    rtk_lon: float
    rtk_alt: float
    rtk_qua: int
    mag: float
    unix_ts: float
    odom_x: float
    odom_y: float
    odom_rockchip_ts: float
    odom_subscriber_ts: float
    odom_yaw_0_360: float
    odom_yaw_m180_p180: float
    rf_data: List[RFData]
    rf_data_raw: List[RFDataRaw]

    def to_dict(self) -> dict:
        """
        Convert the FabricData object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the FabricData object.
        """
        return {
            "machine_id": self.machine_id,
            "payload_idx": self.payload_idx,
            "gps_unix_ts": self.gps_unix_ts,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "gps_alt": self.gps_alt,
            "gps_qua": self.gps_qua,
            "rtk_unix_ts": self.rtk_unix_ts,
            "rtk_lat": self.rtk_lat,
            "rtk_lon": self.rtk_lon,
            "rtk_alt": self.rtk_alt,
            "rtk_qua": self.rtk_qua,
            "mag": self.mag,
            "unix_ts": self.unix_ts,
            "odom_x": self.odom_x,
            "odom_y": self.odom_y,
            "odom_yaw_0_360": self.odom_yaw_0_360,
            "odom_yaw_m180_p180": self.odom_yaw_m180_p180,
            "odom_rockchip_ts": self.odom_rockchip_ts,
            "odom_subscriber_ts": self.odom_subscriber_ts,
            "rf_data": [rf.to_dict() for rf in self.rf_data] if self.rf_data else [],
            "rf_data_raw": (
                [rf.to_dict() for rf in self.rf_data_raw] if self.rf_data_raw else []
            ),
        }


@singleton
class FabricDataSubmitter:
    """
    Allows a machine to locally log mapping data and submit data to FABRIC.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openmind.org/api/core/fabric/submit",
        write_to_local_file: bool = False,
    ):
        """
        Initialize the FabricDataSubmitter.

        Parameters
        ----------
        api_key : Optional[str]
            API key for authentication. Default is None.
        base_url : str
            Base URL for the FABRIC API. Default is
            "https://api.openmind.org/api/core/fabric/submit".
        write_to_local_file : bool
            If True, enables local file logging of submitted data.
            Default is False.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.write_to_local_file = write_to_local_file
        self.filename_base = "dump/fabric"
        self.filename_current = self.update_filename()
        self.max_file_size_bytes = 1024 * 1024
        self.executor = ThreadPoolExecutor(max_workers=1)

    def update_filename(self):
        """
        Update the filename for local logging with a timestamp.
        """
        unix_ts = time.time()
        logging.info(f"fabric time: {unix_ts}")
        unix_ts = str(unix_ts).replace(".", "_")
        filename = f"{self.filename_base}_{unix_ts}Z.jsonl"
        return filename

    def write_dict_to_file(self, data: dict):
        """
        Writes a dictionary to a file in JSON lines format. If the file exceeds max_file_size_bytes,
        creates a new file with a timestamp.

        Parameters
        ----------
        data : dict
            The dictionary to write to the file.
        """
        if not isinstance(data, dict):
            raise ValueError("Provided data must be a dictionary.")

        if (
            os.path.exists(self.filename_current)
            and os.path.getsize(self.filename_current) > self.max_file_size_bytes
        ):
            self.filename_current = self.update_filename()
            logging.info(f"new file name: {self.filename_current}")

        with open(self.filename_current, "a", encoding="utf-8") as f:
            json_line = json.dumps(data)
            f.write(json_line + "\n")
            f.flush()

    def _share_data_worker(self, data: FabricData):
        """
        Worker function to share data from the machine.
        This function runs in a separate thread to avoid blocking the main thread.

        Parameters
        ----------
        data : FabricData
            The data to be shared.
        """
        logging.info(f"_share_data_worker: {data}")
        try:
            json_dict = data.to_dict()
        except Exception as e:
            logging.error(f"Error converting to dict: {str(e)}")
            return

        if self.write_to_local_file:
            self.write_dict_to_file(json_dict)
            logging.info(f"FDS wrote to {self.filename_current}")

        if self.api_key is None or self.api_key == "":
            logging.error("API key missing. Cannot share data to FABRIC.")
            return

        try:
            request = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=json_dict,
                timeout=10,
            )

            if request.status_code == 201:
                logging.debug(f"Data shared: {request.json()}")
            else:
                logging.error(
                    f"Failed to share data: {request.status_code} - {request.text}"
                )
        except Exception as e:
            logging.error(f"Error sharing data: {str(e)}")

    def share_data(self, data: FabricData):
        """
        Share mapping data.
        This function submits mapping data collected by a machine to a thread pool executor
        to run in a separate thread.

        Parameters
        ----------
        data : FabricData
            A mapping data payload to submit.
        """
        logging.debug(f"share data: {data}")
        self.executor.submit(self._share_data_worker, data)
