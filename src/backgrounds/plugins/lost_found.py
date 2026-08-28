import logging

from backgrounds.base import Background, BackgroundConfig
from providers.lostfound_ingest_provider import LostAndFoundIngestProvider


class LostAndFoundIngest(Background):
    """
    Background task that ingests detections published by YoloDetectRTSPProvider
    and writes frames + crops + sightings into SQLite.

    It polls IOProvider for the latest detection packet under a known key.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        cfg_path = getattr(self.config, "cfg_path", None)
        poll_interval = getattr(self.config, "poll_interval", 0.3)

        self.ingest_provider = LostAndFoundIngestProvider(
            cfg_path=cfg_path,
            poll_interval=poll_interval,
        )
        self.ingest_provider.start()

        logging.info(
            "LostAndFoundIngest background initialized "
            f"(cfg_path: {cfg_path}, poll_interval: {poll_interval}s)"
        )

    async def stop(self) -> None:
        try:
            self.ingest_provider.stop()
        except Exception:
            logging.exception("Error stopping LostAndFoundIngestProvider")
