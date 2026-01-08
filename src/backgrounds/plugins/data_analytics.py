import json
import logging
import threading
import time
from typing import Optional, Tuple

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.data_analytics_provider import (
    DataAnalyticsProvider,
    SensorReading,
)
from providers.io_provider import IOProvider


class DataAnalyticsConfig(BackgroundConfig):
    """Configuration for Data Analytics Background."""

    enable_ml: bool = Field(
        default=True, description="Whether to enable ML-based anomaly detection"
    )
    window_size: int = Field(
        default=100, description="Size of the sliding window for data analysis"
    )
    contamination: float = Field(
        default=0.1, description="Expected proportion of anomalies in the data"
    )
    metrics_output_file: Optional[str] = Field(
        default=None, description="Optional file path to write metrics JSON"
    )
    metrics_update_interval: float = Field(
        default=5.0, description="Interval in seconds between metrics updates"
    )
    enable_api: bool = Field(
        default=True, description="Whether to enable API endpoint for metrics"
    )
    api_port: int = Field(default=8080, description="Port for metrics API endpoint")


class DataAnalytics(Background[DataAnalyticsConfig]):
    """Data Analytics Background Process."""

    def __init__(self, config: DataAnalyticsConfig):
        super().__init__(config)

        self.analytics_provider = DataAnalyticsProvider(
            window_size=self.config.window_size,
            contamination=self.config.contamination,
            enable_ml=self.config.enable_ml,
        )

        self.io_provider = IOProvider()
        self.api_server = None
        self.api_thread = None
        self.metrics_thread = None
        self._running = False

        logging.info(
            f"DataAnalytics background initialized "
            f"(enable_ml={self.config.enable_ml}, "
            f"window_size={self.config.window_size})"
        )

    def run(self) -> None:
        self._running = True

        if self.config.enable_api:
            self._start_api_server()

        self._start_metrics_loop()

        while self._running:
            time.sleep(1)

    def _start_api_server(self):
        try:
            import uvicorn
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse

            app = FastAPI(title="OM1 Data Analytics API")

            @app.get("/metrics")
            def get_metrics():
                metrics = self.analytics_provider.get_metrics()
                return JSONResponse(content=metrics.to_dict())

            @app.get("/readings")
            def get_readings(count: int = 10):
                readings = self.analytics_provider.get_recent_readings(count)
                return JSONResponse(content=[r.to_dict() for r in readings])

            @app.get("/anomalies")
            def get_anomalies(count: int = 10):
                anomalies = self.analytics_provider.get_anomalies(count)
                return JSONResponse(content=[r.to_dict() for r in anomalies])

            @app.post("/reset")
            def reset_metrics():
                self.analytics_provider.reset_metrics()
                return JSONResponse(content={"status": "reset"})

            def run_server():
                uvicorn.run(
                    app,
                    host="0.0.0.0",
                    port=self.config.api_port,
                    log_level="info",
                )

            self.api_thread = threading.Thread(target=run_server, daemon=True)
            self.api_thread.start()
            logging.info(
                f"Data Analytics API server started on port {self.config.api_port}"
            )

        except ImportError:
            logging.warning(
                "FastAPI/uvicorn not available. API server disabled. "
                "Install with: pip install fastapi uvicorn"
            )
        except Exception as e:
            logging.error(f"Error starting API server: {e}")

    def _start_metrics_loop(self):
        def update_metrics():
            while self._running:
                try:
                    metrics = self.analytics_provider.get_metrics()

                    if self.config.metrics_output_file:
                        with open(self.config.metrics_output_file, "w") as f:
                            json.dump(metrics.to_dict(), f, indent=2)

                    logging.info(
                        f"Analytics Metrics: "
                        f"readings={metrics.total_readings}, "
                        f"anomalies={metrics.anomalies_detected}, "
                        f"avg_temp={metrics.avg_temperature:.2f}°C, "
                        f"quality={metrics.data_quality_score:.2f}"
                    )

                except Exception as e:
                    logging.error(f"Error updating metrics: {e}")

                time.sleep(self.config.metrics_update_interval)

        self.metrics_thread = threading.Thread(target=update_metrics, daemon=True)
        self.metrics_thread.start()
        logging.info("Metrics update loop started")

    def process_sensor_data(
        self,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        air_quality: Optional[float] = None,
    ) -> Tuple[SensorReading, bool]:
        reading = SensorReading(
            timestamp=time.time(),
            temperature=temperature,
            humidity=humidity,
            air_quality=air_quality,
        )

        return self.analytics_provider.add_reading(reading)

    def stop(self):
        self._running = False
        logging.info("Data Analytics background process stopped")
