import asyncio
import logging

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
        default="http://127.0.0.1:6793",
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
        default=3,
        description="Threshold for the duration of detecting unknown faces before triggering an alert",
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

            # Check if we need to pause due to multiple unknown captures
            if len(unknown_captures) > 1 and not self.is_paused:
                logging.warning(f"Multiple unknown captures detected ({len(unknown_captures)}), pausing patrol")
                self.loop.run_until_complete(self.patrol_provider.pause_patrol())
                self.is_paused = True

            # Process qualified captures (exceeding threshold)
            qualified_captures = [
                capture
                for capture in unknown_captures
                if capture.get("unknown_duration", 0) >= self.config.unknown_capture_threshold
            ]

            # Upload images for new qualified captures
            newly_uploaded = False
            for capture in qualified_captures:
                track_id = capture.get("track_id")
                if track_id and track_id not in self.uploaded_track_ids:
                    if frame_base64:
                        logging.warning(
                            f"Uploading capture for track_id {track_id} "
                            f"(duration: {capture.get('unknown_duration', 0):.1f}s)"
                        )
                        description = (
                            f"Unknown person detected (track_id: {track_id}, "
                            f"duration: {capture.get('unknown_duration', 0):.1f}s, "
                            f"bbox: {capture.get('bbox', [])}, "
                            f"area: {capture.get('area', 0)})"
                        )
                        upload_result = self.loop.run_until_complete(
                            self.patrol_provider.upload_patrol_image(frame_base64, description)
                        )
                        if upload_result:
                            logging.info(f"Successfully uploaded patrol image for track_id {track_id}")
                            self.uploaded_track_ids.add(track_id)
                        newly_uploaded = True

            # Resume patrol if paused and all qualified captures have been uploaded
            if self.is_paused and newly_uploaded:
                all_uploaded = all(
                    capture.get("track_id") in self.uploaded_track_ids
                    for capture in qualified_captures
                    if capture.get("track_id")
                )
                if all_uploaded:
                    self.elevenlabs_provider.add_pending_message(
                        "Alert: Unknown person detected. Please check the patrol report for details."
                    )
                    logging.info("All qualified captures uploaded, resuming patrol")
                    self.loop.run_until_complete(self.patrol_provider.resume_patrol())
                    self.is_paused = False

            if len(unknown_captures) == 0 and self.uploaded_track_ids:
                self.uploaded_track_ids.clear()

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
