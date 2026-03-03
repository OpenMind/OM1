import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AirQualityData:
    """
    Standardized air quality data structure.
    All connectors must return data in this format.

    Parameters
    ----------
    aqi : Optional[int]
        Air Quality Index (0-500+). None if not available.
    pm25 : Optional[float]
        PM2.5 concentration in µg/m³.
    pm10 : Optional[float]
        PM10 concentration in µg/m³.
    co : Optional[float]
        Carbon monoxide in ppm.
    no2 : Optional[float]
        Nitrogen dioxide in µg/m³.
    so2 : Optional[float]
        Sulfur dioxide in µg/m³.
    o3 : Optional[float]
        Ozone in µg/m³.
    temperature : Optional[float]
        Temperature in Celsius.
    humidity : Optional[float]
        Relative humidity in %.
    location : str
        Human-readable location name.
    source : str
        Data source identifier (e.g. 'pms5003', 'bme680', 'aqicn').
    """

    aqi: Optional[int] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    co: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    o3: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    location: str = "Unknown"
    source: str = "Unknown"


# AQI scale berdasarkan standard US EPA
AQI_LEVELS = [
    (50, "GOOD", "Air quality is satisfactory."),
    (100, "MODERATE", "Acceptable; some pollutants may concern sensitive groups."),
    (
        150,
        "UNHEALTHY FOR SENSITIVE GROUPS",
        "Sensitive groups may experience health effects.",
    ),
    (200, "UNHEALTHY", "Everyone may begin to experience health effects."),
    (300, "VERY UNHEALTHY", "Health alert: everyone may experience serious effects."),
    (
        float("inf"),
        "HAZARDOUS",
        "Health warning: emergency conditions for entire population.",
    ),
]


def get_aqi_level(aqi: int) -> tuple[str, str]:
    """
    Get AQI level label and description based on AQI value.

    Parameters
    ----------
    aqi : int
        AQI value.

    Returns
    -------
    tuple[str, str]
        (label, description)
    """
    for threshold, label, description in AQI_LEVELS:
        if aqi <= threshold:
            return label, description
    return "HAZARDOUS", AQI_LEVELS[-1][2]  # pragma: no cover


class AirQualityConnector(ABC):
    """
    Abstract base class for all air quality connectors.

    Every connector (hardware or API) must implement this interface
    so that AirQualityInput stays sensor-agnostic.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def connect(self) -> bool:
        """
        Initialize connection to sensor or API.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def read(self) -> Optional[AirQualityData]:
        """
        Read air quality data.

        Returns
        -------
        Optional[AirQualityData]
            Standardized data, or None if read failed.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Clean up and close connection.
        """
        pass
