import asyncio

import aiohttp

from inputs.plugins.air_quality.connector.base import (
    AirQualityConnector,
    AirQualityData,
)


class AqicnConnector(AirQualityConnector):
    """
    Air quality connector using AQICN (World Air Quality Index) API.

    Fetches real-time AQI and pollutant data from waqi.info.
    Use this connector when no physical sensor is available.

    API docs: https://aqicn.org/json-api/doc/
    Free token: https://aqicn.org/data-platform/token/
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must contain:
            - api_key (str): AQICN token
            - latitude (float): location latitude
            - longitude (float): location longitude
        """
        super().__init__(config)
        self.api_key: str = config.get("api_key", "")
        self.latitude: float = config.get("latitude", -6.2088)
        self.longitude: float = config.get("longitude", 106.8456)

    async def connect(self) -> bool:
        """Validate API key and confirm connector is ready."""
        if not self.api_key:
            self.logger.warning("AqicnConnector: no API key provided")
            return False
        return True

    async def disconnect(self) -> None:
        """No-op: stateless HTTP connector requires no teardown."""
        pass  # Stateless HTTP — nothing to close

    async def read(self) -> AirQualityData | None:
        """
        Fetch air quality data from AQICN API.

        Returns
        -------
        AirQualityData or None
            Parsed data, or None if request failed.
        """
        if not self.api_key:
            return None

        url = f"https://api.waqi.info/feed/geo:{self.latitude};{self.longitude}/"
        params = {"token": self.api_key}

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"AqicnConnector: HTTP {response.status}: {await response.text()}"
                        )
                        return None

                    payload = await response.json()

            if payload.get("status") != "ok":
                self.logger.error(f"AqicnConnector: API error: {payload.get('data')}")
                return None

            return self._parse(payload)

        except asyncio.TimeoutError:
            self.logger.error("AqicnConnector: request timed out")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"AqicnConnector: network error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"AqicnConnector: unexpected error: {e}")
            return None

    def _parse(self, payload: dict) -> AirQualityData:
        """
        Parse AQICN API response into AirQualityData.

        Parameters
        ----------
        payload : dict
            Raw API response with status == 'ok'.

        Returns
        -------
        AirQualityData
        """
        data = payload.get("data", {})
        iaqi = data.get("iaqi", {})

        def get_val(key: str) -> float | None:
            entry = iaqi.get(key, {})
            return entry.get("v") if entry else None

        aqi_raw = data.get("aqi", "-")
        aqi = int(aqi_raw) if aqi_raw not in ("-", None) else None

        location = data.get("city", {}).get("name", "Unknown")

        return AirQualityData(
            aqi=aqi,
            pm25=get_val("pm25"),
            pm10=get_val("pm10"),
            co=get_val("co"),
            no2=get_val("no2"),
            so2=get_val("so2"),
            o3=get_val("o3"),
            temperature=get_val("t"),
            humidity=get_val("h"),
            location=location,
            source="aqicn",
        )
