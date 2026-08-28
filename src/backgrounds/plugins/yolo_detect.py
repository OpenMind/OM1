import logging

from backgrounds.base import Background, BackgroundConfig
from providers.yolo_detect_rtsp_provider import YoloDetectRTSPProvider


class YoloDetectRTSP(Background):
    """
    Background task that runs YOLO object detection on an RTSP stream.

    Publishes the latest frame + detections into IOProvider dynamic variables.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        cfg_path = getattr(self.config, "cfg_path", None)
        rtsp_url = getattr(self.config, "rtsp_url", "rtsp://localhost:8554/top_camera")
        decode_format = getattr(self.config, "decode_format", "H264")
        fps = getattr(self.config, "fps", 10)
        ingest_stride = getattr(self.config, "ingest_stride", 10)
        queue_max = getattr(self.config, "queue_max", 10)

        self.detect_provider = YoloDetectRTSPProvider(
            cfg_path=cfg_path,
            rtsp_url=rtsp_url,
            decode_format=decode_format,
            fps=fps,
            ingest_stride=ingest_stride,
            queue_max=queue_max,
        )
        self.detect_provider.start()

        logging.info(
            "YoloDetectRTSP background initialized "
            f"(cfg_path: {cfg_path}, rtsp_url: {rtsp_url}, "
            f"stride: {ingest_stride}, queue_max: {queue_max})"
        )

    async def stop(self) -> None:
        try:
            self.detect_provider.stop()
        except Exception:
            logging.exception("Error stopping YoloDetectRTSPProvider")
