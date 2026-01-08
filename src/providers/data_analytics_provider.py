import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .singleton import singleton


@dataclass
class SensorReading:
    """Container for a single sensor reading."""

    timestamp: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    air_quality: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.temperature if self.temperature is not None else np.nan,
                self.humidity if self.humidity is not None else np.nan,
                self.air_quality if self.air_quality is not None else np.nan,
            ]
        )


@dataclass
class AnalyticsMetrics:
    """Container for analytics metrics."""

    total_readings: int = 0
    anomalies_detected: int = 0
    avg_temperature: Optional[float] = None
    avg_humidity: Optional[float] = None
    avg_air_quality: Optional[float] = None
    min_temperature: Optional[float] = None
    max_temperature: Optional[float] = None
    data_quality_score: float = 1.0
    processing_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@singleton
class DataAnalyticsProvider:
    """Data Analytics Provider for preprocessing, anomaly detection, and metrics."""

    def __init__(
        self,
        window_size: int = 100,
        contamination: float = 0.1,
        enable_ml: bool = True,
    ):
        self.window_size = window_size
        self.contamination = contamination
        self.enable_ml = enable_ml
        self.data_buffer: deque = deque(maxlen=window_size)
        self.anomalies: List[SensorReading] = []
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self._model_trained = False
        self.metrics = AnalyticsMetrics()
        self.processing_times: deque = deque(maxlen=100)
        self._lock = threading.Lock()

        logging.info(
            f"DataAnalyticsProvider initialized (window_size={window_size}, "
            f"contamination={contamination}, enable_ml={enable_ml})"
        )

    def preprocess_data(self, reading: SensorReading) -> SensorReading:
        start_time = time.time()

        processed = SensorReading(
            timestamp=reading.timestamp,
            temperature=reading.temperature,
            humidity=reading.humidity,
            air_quality=reading.air_quality,
        )

        if len(self.data_buffer) >= 10:
            with self._lock:
                temps = [
                    r.temperature for r in self.data_buffer if r.temperature is not None
                ]
                humids = [
                    r.humidity for r in self.data_buffer if r.humidity is not None
                ]
                aqs = [
                    r.air_quality for r in self.data_buffer if r.air_quality is not None
                ]

                if temps and processed.temperature is not None:
                    q1, q3 = np.percentile(temps, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    if (
                        processed.temperature < lower_bound
                        or processed.temperature > upper_bound
                    ):
                        logging.warning(
                            f"Temperature outlier detected: {processed.temperature}, "
                            f"bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
                        )
                        processed.temperature = max(
                            lower_bound, min(upper_bound, processed.temperature)
                        )

                if humids and processed.humidity is not None:
                    q1, q3 = np.percentile(humids, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    if (
                        processed.humidity < lower_bound
                        or processed.humidity > upper_bound
                    ):
                        logging.warning(
                            f"Humidity outlier detected: {processed.humidity}, "
                            f"bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
                        )
                        processed.humidity = max(
                            lower_bound, min(upper_bound, processed.humidity)
                        )

                if aqs and processed.air_quality is not None:
                    q1, q3 = np.percentile(aqs, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    if (
                        processed.air_quality < lower_bound
                        or processed.air_quality > upper_bound
                    ):
                        logging.warning(
                            f"Air quality outlier detected: {processed.air_quality}, "
                            f"bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
                        )
                        processed.air_quality = max(
                            lower_bound, min(upper_bound, processed.air_quality)
                        )

        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)

        return processed

    def add_reading(self, reading: SensorReading) -> Tuple[SensorReading, bool]:
        processed = self.preprocess_data(reading)

        with self._lock:
            self.data_buffer.append(processed)

        self._update_metrics(processed)

        is_anomaly = False
        if self.enable_ml:
            is_anomaly = self._detect_anomaly(processed)

        if is_anomaly:
            with self._lock:
                self.anomalies.append(processed)
                self.metrics.anomalies_detected += 1
            logging.warning(
                f"Anomaly detected: temp={processed.temperature}, "
                f"humidity={processed.humidity}, aq={processed.air_quality}"
            )

        return processed, is_anomaly

    def _detect_anomaly(self, reading: SensorReading) -> bool:
        if not self.enable_ml:
            return False

        if len(self.data_buffer) < 20:
            return False

        if not self._model_trained or len(self.data_buffer) % 50 == 0:
            self._train_anomaly_detector()

        if self.anomaly_detector is None or self.scaler is None:
            return False

        try:
            data_array = reading.to_array().reshape(1, -1)

            if np.isnan(data_array).any():
                return False

            data_scaled = self.scaler.transform(data_array)
            prediction = self.anomaly_detector.predict(data_scaled)[0]

            return prediction == -1

        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            return False

    def _train_anomaly_detector(self):
        if len(self.data_buffer) < 20:
            return

        try:
            with self._lock:
                data_list = [r.to_array() for r in self.data_buffer]
                data_array = np.array(data_list)

                valid_mask = ~np.isnan(data_array).all(axis=1)
                if valid_mask.sum() < 10:
                    return

                data_array = data_array[valid_mask]

                for col_idx in range(data_array.shape[1]):
                    col = data_array[:, col_idx]
                    nan_mask = np.isnan(col)
                    if nan_mask.any():
                        last_valid = None
                        for i in range(len(col)):
                            if not np.isnan(col[i]):
                                last_valid = col[i]
                            elif last_valid is not None:
                                col[i] = last_valid

                self.scaler = StandardScaler()
                data_scaled = self.scaler.fit_transform(data_array)

                self.anomaly_detector = IsolationForest(
                    contamination=self.contamination, random_state=42  # type: ignore
                )
                self.anomaly_detector.fit(data_scaled)

                self._model_trained = True
                logging.info(
                    f"Anomaly detection model trained on {len(data_array)} samples"
                )

        except Exception as e:
            logging.error(f"Error training anomaly detector: {e}")
            self._model_trained = False

    def _update_metrics(self, reading: SensorReading):
        with self._lock:
            self.metrics.total_readings += 1

            temps = [
                r.temperature for r in self.data_buffer if r.temperature is not None
            ]
            humids = [r.humidity for r in self.data_buffer if r.humidity is not None]
            aqs = [r.air_quality for r in self.data_buffer if r.air_quality is not None]

            if temps:
                self.metrics.avg_temperature = float(np.mean(temps))
                self.metrics.min_temperature = float(np.min(temps))
                self.metrics.max_temperature = float(np.max(temps))

            if humids:
                self.metrics.avg_humidity = float(np.mean(humids))

            if aqs:
                self.metrics.avg_air_quality = float(np.mean(aqs))

            total_values = sum(
                [
                    1
                    for r in self.data_buffer
                    if r.temperature is not None
                    or r.humidity is not None
                    or r.air_quality is not None
                ]
            )
            expected_values = len(self.data_buffer) * 3
            if expected_values > 0:
                completeness = total_values / expected_values
                if len(temps) > 5:
                    temp_std = np.std(temps)
                    consistency = 1.0 / (1.0 + temp_std / 10.0)
                else:
                    consistency = 1.0
                self.metrics.data_quality_score = float(
                    (completeness + consistency) / 2.0
                )

            if self.processing_times:
                self.metrics.processing_latency_ms = float(
                    np.mean(list(self.processing_times))
                )

    def get_metrics(self) -> AnalyticsMetrics:
        with self._lock:
            return AnalyticsMetrics(
                total_readings=self.metrics.total_readings,
                anomalies_detected=self.metrics.anomalies_detected,
                avg_temperature=self.metrics.avg_temperature,
                avg_humidity=self.metrics.avg_humidity,
                avg_air_quality=self.metrics.avg_air_quality,
                min_temperature=self.metrics.min_temperature,
                max_temperature=self.metrics.max_temperature,
                data_quality_score=self.metrics.data_quality_score,
                processing_latency_ms=self.metrics.processing_latency_ms,
            )

    def get_recent_readings(self, count: int = 10) -> List[SensorReading]:
        with self._lock:
            return list(self.data_buffer)[-count:]

    def get_anomalies(self, count: int = 10) -> List[SensorReading]:
        with self._lock:
            return list(self.anomalies)[-count:]

    def reset_metrics(self):
        with self._lock:
            self.data_buffer.clear()
            self.anomalies.clear()
            self.metrics = AnalyticsMetrics()
            self.processing_times.clear()
            self._model_trained = False
            self.anomaly_detector = None
            self.scaler = None
            logging.info("DataAnalyticsProvider metrics reset")
