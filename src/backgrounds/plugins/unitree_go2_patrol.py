import asyncio
import logging
import time

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.unitree_go2_patrol_provider import UnitreeGo2PatrolProvider


class UnitreeGo2PatrolConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 Patrol Background.
    """

    patrol_base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the patrol control API",
    )
    face_presence_base_url: str = Field(
        default="http://localhost:6793",
        description="Base URL for the face presence API",
    )
    patrol_image_report_base_url: str = Field(
        default="https://api.openmind.com",
        description="URL for reporting patrol data to OpenMind API",
    )
    api_key: str = Field(
        default="",
        description="API key for OpenMind patrol upload endpoint",
    )
    unknown_capture_threshold: int = Field(
        default=2,
        description="Threshold for the duration of detecting unknown faces before triggering an alert",
    )
    upload_cooldown_seconds: int = Field(
        default=5,
        description="Minimum seconds to wait before uploading the same track_id again",
    )
    force_resume_seconds: int = Field(
        default=15,
        description="Seconds to wait before force resuming patrol if unknown captures persist",
    )
    safe_force_resume_seconds: int = Field(
        default=5,
        description="Minimum seconds to wait before force resuming patrol to avoid rapid pause/resume cycles",
    )


class UnitreeGo2Patrol(Background[UnitreeGo2PatrolConfig]):
    """
    Background task for patrolling with Unitree Go2 robot.
    """

    def __init__(self, config: UnitreeGo2PatrolConfig):
        """
        Initialize Patrol background task with configuration.

        Parameters
        ----------
        config : UnitreeGo2PatrolConfig
            Configuration for the Unitree Go2 Patrol background task, including patrol parameters and options.
        """
        super().__init__(config)

        self.patrol_provider = UnitreeGo2PatrolProvider(
            api_key=config.api_key,
            patrol_base_url=config.patrol_base_url,
            face_presence_base_url=config.face_presence_base_url,
            patrol_image_report_base_url=config.patrol_image_report_base_url,
        )

        self.elevenlabs_provider: ElevenLabsTTSProvider = ElevenLabsTTSProvider()

        self.loop = asyncio.new_event_loop()
        self.uploaded_track_ids = set()
        self.is_paused = False
        self.last_pause_time = 0
        self.last_force_resume_time = 0

        logging.info("Initialized Unitree Go2 Patrol Background Task")

    async def start_patrol(self) -> None:
        """
        Start the patrol behavior.
        """
        await self.patrol_provider.start_patrol()

    async def stop_patrol(self) -> None:
        """
        Stop the patrol behavior.
        """
        await self.patrol_provider.stop_patrol()

    async def pause_patrol(self) -> None:
        """
        Pause the patrol behavior.
        """
        await self.patrol_provider.pause_patrol()

    async def resume_patrol(self) -> None:
        """
        Resume the patrol behavior.
        """
        await self.patrol_provider.resume_patrol()

    def run(self) -> None:
        """
        Main loop for the patrol background task. This method will be called by the background manager to execute the patrol behavior.
        """
        try:
            report = self.loop.run_until_complete(self.patrol_provider.get_report())
            frame_base64 = report.get("frame_b64", "")
            unknown_captures = report.get("unknown_captures", [])
            track_ids = [capture.get("track_id") for capture in unknown_captures]

            if (
                len(unknown_captures) > 0
                and not self.is_paused
                and not any(track_id in self.uploaded_track_ids for track_id in track_ids)
                and time.time() - self.last_force_resume_time > self.config.safe_force_resume_seconds
            ):
                logging.info(f"Detected {len(unknown_captures)} unknown captures")
                try:
                    self.loop.run_until_complete(self.patrol_provider.pause_patrol())
                    self.is_paused = True
                    self.last_pause_time = time.time()
                    logging.info("Patrol paused to handle unknown captures")
                except Exception:
                    logging.exception("Failed to pause patrol")

            if (
                self.is_paused
                and len(track_ids) > 0
                and any(
                    detection.get("unknown_duration", 0) > self.config.unknown_capture_threshold
                    for detection in unknown_captures
                )
            ):
                try:
                    description = (
                        f"Detected unknown person with track IDs: {track_ids} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    self.loop.run_until_complete(self.patrol_provider.upload_patrol_image(frame_base64, description))
                    self.elevenlabs_provider.add_pending_message(
                        "Alert: Unknown person detected during patrol. Image has been uploaded for review."
                    )
                    logging.info(f"Uploaded patrol data for track IDs: {track_ids}")
                except Exception:
                    logging.exception("Failed to upload patrol data")

                self.uploaded_track_ids.update(track_ids)

                try:
                    self.loop.run_until_complete(self.patrol_provider.resume_patrol())
                    self.is_paused = False
                    logging.info("Patrol resumed after handling unknown captures")
                except Exception:
                    logging.exception("Failed to resume patrol")

            if self.is_paused and time.time() - self.last_pause_time > self.config.force_resume_seconds:
                try:
                    self.loop.run_until_complete(self.patrol_provider.resume_patrol())
                    self.is_paused = False
                    self.last_force_resume_time = time.time()
                    logging.info("Force resumed patrol after prolonged pause")
                except Exception:
                    logging.exception("Failed to force resume patrol")

        except Exception as e:
            logging.error(f"Error getting patrol report: {e}")

        self.sleep(1)

    def stop(self) -> None:
        """
        Stop the patrol background task and clean up resources.
        """
        logging.info("Stopping Unitree Go2 Patrol Background Task")
        if self.loop and not self.loop.is_closed():
            self.loop.close()
