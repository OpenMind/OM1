import logging

from backgrounds.base import Background, BackgroundConfig
from providers.episodic_memory_provider import EpisodicMemoryProvider


class EpisodicMemoryWriterConfig(BackgroundConfig):
    """Configuration for EpisodicMemoryWriter. No extra fields needed."""

    pass


class EpisodicMemoryWriter(Background[EpisodicMemoryWriterConfig]):
    """
    Background task that periodically flushes episodic memories to disk.

    BackgroundOrchestrator wraps run() in its own while-not-stop loop,
    so run() must NOT contain its own while loop.
    Pattern: flush once -> sleep(interval) -> return.
    """

    FLUSH_INTERVAL_SECONDS: float = 30.0

    def __init__(self, config: EpisodicMemoryWriterConfig):
        super().__init__(config)
        self.provider = EpisodicMemoryProvider()
        logging.info("EpisodicMemoryWriter initialized")

    def run(self) -> None:
        """Flush pending episodes to disk, then sleep until next interval."""
        try:
            self.provider.save_to_disk()
            logging.debug("EpisodicMemoryWriter: flush completed")
        except Exception as e:
            logging.error(f"EpisodicMemoryWriter: flush failed: {e}")

        self.sleep(self.FLUSH_INTERVAL_SECONDS)

    def stop(self) -> None:
        """Final flush when the background task is stopped."""
        logging.info("EpisodicMemoryWriter: stopping, performing final flush")
        try:
            self.provider.save_to_disk()
        except Exception as e:
            logging.error(f"EpisodicMemoryWriter: final flush failed: {e}")
