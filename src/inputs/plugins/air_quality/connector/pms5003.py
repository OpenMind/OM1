import asyncio

import serial

from inputs.plugins.air_quality.connector.base import (
    AirQualityConnector,
    AirQualityData,
)


class PMS5003Connector(AirQualityConnector):
    """
    Air quality connector for PMS5003/PMS7003 particulate matter sensor.

    Reads PM1.0, PM2.5, PM10 via UART/Serial.
    Sensor sends 32-byte frames at 9600 baud.

    Wiring:
        VCC  → 5V
        GND  → GND
        TX   → RX (e.g. /dev/ttyUSB0 or /dev/ttyAMA0)

    Datasheet: https://www.aqmd.gov/docs/default-source/aq-spec/resources-page/plantower-pms5003-manual_v2-3.pdf
    """

    BAUD_RATE = 9600
    FRAME_LENGTH = 32
    FRAME_START = (0x42, 0x4D)

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must contain:
            - port (str): serial port, e.g. '/dev/ttyUSB0'
            - location (str, optional): location label, default 'Robot'
        """
        super().__init__(config)
        self.port: str = config.get("port", "/dev/ttyUSB0")
        self.location: str = config.get("location", "Robot")
        self._serial: serial.Serial | None = None

    async def connect(self) -> bool:
        """Open the serial port and connect to the PMS5003/7003 sensor."""
        try:
            self._serial = serial.Serial(
                self.port, baudrate=self.BAUD_RATE, timeout=2.0
            )
            self.logger.info(f"PMS5003Connector: connected on {self.port}")
            return True
        except serial.SerialException as e:
            self.logger.error(f"PMS5003Connector: failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Close the serial port if open."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            self.logger.info("PMS5003Connector: disconnected")

    async def read(self) -> AirQualityData | None:
        """
        Read one frame from PMS5003 sensor.

        Returns
        -------
        AirQualityData or None
            Parsed particulate matter data, or None if read failed.
        """
        if self._serial is None or not self._serial.is_open:
            self.logger.error("PMS5003Connector: not connected")
            return None

        try:
            # Run blocking serial read in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            frame = await loop.run_in_executor(None, self._read_frame)
            if frame is None:
                return None
            return self._parse(frame)

        except Exception as e:
            self.logger.error(f"PMS5003Connector: read error: {e}")
            return None

    def _read_frame(self) -> bytes | None:
        """
        Read and validate one 32-byte PMS5003 frame (blocking).

        Returns
        -------
        bytes or None
            Raw 32-byte frame, or None if invalid/timeout.
        """
        # Sync to frame start bytes 0x42 0x4D
        if self._serial is None:
            return None
        while True:
            byte = self._serial.read(1)
            if not byte:
                return None
            if byte[0] == self.FRAME_START[0]:
                next_byte = self._serial.read(1)
                if next_byte and next_byte[0] == self.FRAME_START[1]:
                    break

        rest = self._serial.read(self.FRAME_LENGTH - 2)
        if len(rest) != self.FRAME_LENGTH - 2:
            return None

        frame = bytes(self.FRAME_START) + rest

        # Validate checksum — sum of bytes 0..29 == bytes 30..31 (big-endian)
        checksum = sum(frame[:30]) & 0xFFFF
        expected = (frame[30] << 8) | frame[31]
        if checksum != expected:
            self.logger.warning("PMS5003Connector: checksum mismatch")
            return None

        return frame

    def _parse(self, frame: bytes) -> AirQualityData:
        """
        Parse validated 32-byte PMS5003 frame.

        Frame layout (big-endian, each field 2 bytes):
            [4:6]   PM1.0 standard
            [6:8]   PM2.5 standard
            [8:10]  PM10  standard

        Parameters
        ----------
        frame : bytes
            Validated 32-byte frame.

        Returns
        -------
        AirQualityData
        """

        def word(offset: int) -> int:
            return (frame[offset] << 8) | frame[offset + 1]

        pm25 = float(word(6))
        pm10 = float(word(8))

        # Estimate AQI from PM2.5 using US EPA breakpoints
        aqi = self._pm25_to_aqi(pm25)

        return AirQualityData(
            aqi=aqi,
            pm25=pm25,
            pm10=pm10,
            location=self.location,
            source="pms5003",
        )

    @staticmethod
    def _pm25_to_aqi(pm25: float) -> int:
        """
        Convert PM2.5 concentration to AQI using US EPA formula.

        Parameters
        ----------
        pm25 : float
            PM2.5 in µg/m³.

        Returns
        -------
        int
            Estimated AQI value.
        """
        # (C_low, C_high, AQI_low, AQI_high)
        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        for c_low, c_high, aqi_low, aqi_high in breakpoints:
            if c_low <= pm25 <= c_high:
                aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (
                    pm25 - c_low
                ) + aqi_low
                return round(aqi)
        return 500
