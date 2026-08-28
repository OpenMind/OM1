import logging

from backgrounds.base import Background, BackgroundConfig
from providers.room_type_location_provider import RoomTypeLocationProvider


class RoomTypeLocation(Background):
    """
    Background task that starts the RoomTypeLocationProvider.

    This task monitors the IOProvider's `room_type` dynamic variable and, when a
    valid room type appears, posts it to the map locations API
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        """
        Initialize the Room Type Location background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration for the background task.
            Expected (but optional) fields on config:
              - base_url: str
              - timeout: int
              - refresh_interval: int
              - map_name: str
        """
        super().__init__(config)

        base_url = getattr(
            self.config,
            "base_url",
            "http://localhost:5000/maps/locations/add/slam",
        )
        timeout = getattr(self.config, "timeout", 5)
        refresh_interval = getattr(self.config, "refresh_interval", 5)
        map_name = getattr(self.config, "map_name", "map")

        self.room_type_location_provider = RoomTypeLocationProvider(
            base_url=base_url,
            timeout=timeout,
            refresh_interval=refresh_interval,
            map_name=map_name,
        )
        self.room_type_location_provider.start()

        logging.info(
            "RoomTypeLocation background initialized "
            f"(base_url: {base_url}, map_name: {map_name}, "
            f"refresh: {refresh_interval}s)"
        )
