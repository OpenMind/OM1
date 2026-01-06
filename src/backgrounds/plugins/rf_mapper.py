import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional

from bleak import BleakScanner
from bleak.backends.scanner import AdvertisementData
from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.fabric_map_provider import (
    FabricData,
    FabricDataSubmitter,
    RFData,
    RFDataRaw,
)
from providers.gps_provider import GpsProvider
from providers.odom_provider import OdomProvider
from providers.rtk_provider import RtkProvider


class RFmapperConfig(BackgroundConfig):
    """
    Configuration for RFmapper Background.

    Parameters
    ----------
    name : str
        Name of the RF mapper.
    api_key : Optional[str]
        API key for Fabric.
    URID : Optional[str]
        Unique Robot ID.
    unitree_ethernet : Optional[str]
        Unitree Ethernet channel.
    """

    name: str = Field(default="RFmapper", description="Name of the RF mapper")
    api_key: Optional[str] = Field(default=None, description="API key")
    URID: Optional[str] = Field(default=None, description="Unique Robot ID")
    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Ethernet channel"
    )


class RFmapper(Background[RFmapperConfig]):
    """
    Assemble location and BLE data.
    """

    def __init__(self, config: RFmapperConfig):
        """
        Initialize the RFmapper with configuration.

        Parameters
        ----------
        config : RFmapperConfig
            Configuration object for the background.
        """
        super().__init__(config)

        logging.info(f"Mapper config: {config}")

        self.name = self.config.name
        self.api_key = self.config.api_key
        self.URID = self.config.URID
        self.unitree_ethernet = self.config.unitree_ethernet

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._scan_task)
        self.running = False

        self.scan_results: List[RFData] = []
        self.scan_idx = 0
        self.scan_last_sent = 0

        self.payload_idx = 0

        self.odom_rockchip_ts = 0.0
        self.odom_subscriber_ts = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw_0_360 = 0.0
        self.odom_yaw_m180_p180 = 0.0

        self.gps_unix_ts = 0.0
        self.gps_lat = 0.0
        self.gps_lon = 0.0
        self.gps_alt = 0.0
        self.gps_qua = 0
        self.yaw_mag_0_360 = 0.0
        self.ble_scan: List[RFDataRaw] = []

        self.rtk_unix_ts = 0.0
        self.rtk_lat = 0.0
        self.rtk_lon = 0.0
        self.rtk_alt = 0.0
        self.rtk_qua = 0

        self.gps = GpsProvider()
        self.gps_on = self.gps.running
        logging.info(f"Mapper Gps Provider: {self.gps}")

        self.rtk = RtkProvider()
        self.rtk_on = self.rtk.running
        logging.info(f"Mapper Rtk Provider: {self.rtk}")

        self.odom = OdomProvider()
        logging.info(f"Mapper Odom Provider: {self.odom}")

        self.fds = FabricDataSubmitter(api_key=self.api_key, write_to_local_file=True)

        self.seen_devices: Dict[str, RFData] = {}

        self.seen_names: List[str] = []

        self.start()

    async def scan(self) -> List[RFData]:
        """
        Perform a BLE scan and collect RF data.

        Returns
        -------
        List[RFData]
            List of RFData objects from the scan.
        """

        def detection_callback(device, advdata: AdvertisementData):
            """
            Callback for handling detected BLE devices.

            Parameters
            ----------
            device : BLEDevice
                The detected BLE device.
            advdata : AdvertisementData
                The advertisement data associated with the device.
            """
            addr = device.address

            local_name = None
            if device.name:
                local_name = device.name
            elif advdata.local_name:
                local_name = advdata.local_name

            mfgkey = ""
            mfgval = ""
            service_uuid = ""

            if advdata.manufacturer_data:
                for key, value in advdata.manufacturer_data.items():
                    mfgkey = hex(key).upper()
                    mfgval = value.hex().upper()
                    break

            if advdata.service_uuids:
                service_uuid = advdata.service_uuids[0]

            # we want to update everything EXCEPT we do not want to overwrite a resolved name
            # and we do not want to replace a long mfgval with a short one
            # we also want to update the TX power if we receive those data
            rssi = advdata.rssi
            tx_power = advdata.tx_power

            if addr in self.seen_devices:
                self.seen_devices[addr].rssi = rssi
                self.seen_devices[addr].unix_ts = time.time()
                if tx_power and self.seen_devices[addr].tx_power is None:
                    self.seen_devices[addr].tx_power = tx_power
                    logging.info(
                        f"Updated BLE tx_power: {self.seen_devices[addr].tx_power}"
                    )
                if local_name and self.seen_devices[addr].name is None:
                    self.seen_devices[addr].name = local_name
                    logging.info(f"Updated BLE name: {self.seen_devices[addr].name}")
                if len(mfgval) > len(self.seen_devices[addr].mfgval):
                    self.seen_devices[addr].mfgval = mfgval
                    logging.info(
                        f"Updated BLE mfgval: {self.seen_devices[addr].mfgval}"
                    )
            else:

                # this is a new device
                self.seen_devices[addr] = RFData(
                    unix_ts=time.time(),
                    address=addr,
                    name=local_name if local_name else None,
                    rssi=rssi,
                    tx_power=tx_power if tx_power else None,
                    service_uuid=service_uuid,
                    mfgkey=mfgkey,
                    mfgval=mfgval,
                )

        scanner = BleakScanner(detection_callback)

        await scanner.start()
        # scan for 5 seconds, then return a new payload
        await asyncio.sleep(5.0)
        await scanner.stop()

        logging.debug(f"ready to sort: {self.seen_devices.values()}")

        # Get the top 20 devices with the strongest RSSI
        sorted_devices = sorted(
            self.seen_devices.values(), key=lambda d: d.rssi, reverse=True
        )

        final_list: List[RFData] = []
        for i, device in enumerate(sorted_devices):
            # return all the strong ones, or, the ones with a local name
            # some will be both, but that's good
            if i < 20 or device.name:
                final_list.append(device)
        logging.debug(f"Scan...{final_list}")

        self.scan_idx += 1

        return final_list

    def _scan_task(self):
        """
        Background task to perform BLE scanning.
        """
        asyncio.set_event_loop(self.loop)
        logging.info("Starting RF scan thread...")
        self.running = True
        while self.running:
            self.scan_results = self.loop.run_until_complete(self.scan())
            logging.info(f"RF scan index: {self.scan_idx}")
            logging.info(f"RF scan last sent: {self.scan_last_sent}")
            time.sleep(0.5)

    def start(self):
        """
        Start the background process.
        """
        self.thread.start()

    def stop(self):
        """
        Stop the background process.
        """
        self.running = False
        time.sleep(1)
        self.thread.join()

    def run(self) -> None:
        """
        Run the background process.
        This method will run indefinitely, simulating a long-running task.
        """
        try:
            while self.running:

                logging.info(f"Sending to fabric: payload {self.payload_idx}")

                # add scan results if they are new
                logging.info(f"RF scan index: {self.scan_idx}")
                logging.info(f"RF scan last sent: {self.scan_last_sent}")
                fresh_scan_results = []
                if self.scan_results and self.scan_idx > self.scan_last_sent:
                    fresh_scan_results = self.scan_results
                    self.scan_last_sent = self.scan_idx
                    self.scan_results = []
                    logging.info(f"RF scan sending new payload: {self.scan_last_sent}")

                # basic gps data and occasional scan results
                try:
                    g = self.gps.data
                    logging.debug(f"GPS data: {g}")
                    if g:
                        if g["gps_unix_ts"]:
                            self.gps_unix_ts = g["gps_unix_ts"]
                            self.gps_lat = g["gps_lat"]
                            self.gps_lon = g["gps_lon"]
                            self.gps_alt = g["gps_alt"]
                            self.yaw_mag_0_360 = g["yaw_mag_0_360"]
                            self.gps_qua = g["gps_qua"]

                        if g["ble_scan"] is not None:
                            self.ble_scan = g["ble_scan"]
                            logging.debug(f"RF scan results {self.ble_scan}")
                        else:
                            logging.warn("No nRF52 scan results")

                except Exception as e:
                    logging.error(f"Error parsing GPS: {e}")

                try:
                    o = self.odom.position
                    logging.debug(f"Odom data: {o}")
                    if o:
                        self.odom_x = o["odom_x"]
                        self.odom_y = o["odom_y"]
                        self.odom_rockchip_ts = o["odom_rockchip_ts"]
                        self.odom_subscriber_ts = o["odom_subscriber_ts"]
                        self.odom_yaw_0_360 = o["odom_yaw_0_360"]
                        self.odom_yaw_m180_p180 = o["odom_yaw_m180_p180"]
                except Exception as e:
                    logging.error(f"Error parsing Odom: {e}")

                if hasattr(self.rtk, "running"):
                    try:
                        r = self.rtk.data
                        logging.debug(f"RTK data: {r}")
                        if r:
                            self.rtk_unix_ts = r["rtk_unix_ts"]
                            self.rtk_lat = r["rtk_lat"]
                            self.rtk_lon = r["rtk_lon"]
                            self.rtk_alt = r["rtk_alt"]
                            self.rtk_qua = r["rtk_qua"]
                    except Exception as e:
                        logging.error(f"Error parsing RTK: {e}")

                try:
                    self.fds.share_data(
                        FabricData(
                            machine_id=self.URID if self.URID else "Unknown",
                            payload_idx=self.payload_idx,
                            gps_unix_ts=self.gps_unix_ts,
                            gps_lat=self.gps_lat,
                            gps_lon=self.gps_lon,
                            gps_alt=self.gps_alt,
                            gps_qua=self.gps_qua,
                            rtk_unix_ts=self.rtk_unix_ts,
                            rtk_lat=self.rtk_lat,
                            rtk_lon=self.rtk_lon,
                            rtk_alt=self.rtk_alt,
                            rtk_qua=self.rtk_qua,
                            mag=self.yaw_mag_0_360,
                            unix_ts=time.time(),
                            odom_rockchip_ts=self.odom_rockchip_ts,
                            odom_subscriber_ts=self.odom_subscriber_ts,
                            odom_x=self.odom_x,
                            odom_y=self.odom_y,
                            odom_yaw_0_360=self.odom_yaw_0_360,
                            odom_yaw_m180_p180=self.odom_yaw_m180_p180,
                            rf_data=fresh_scan_results,
                            rf_data_raw=self.ble_scan,
                        )
                    )
                    self.payload_idx += 1
                except Exception as e:
                    logging.error(f"Error sharing to Fabric: {e}")

                time.sleep(1)  # we should send a payload every second

        except KeyboardInterrupt:
            logging.info("Stopping RF scanner...")
            self.stop()
