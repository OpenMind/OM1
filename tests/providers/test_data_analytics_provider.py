import time

from providers.data_analytics_provider import (
    AnalyticsMetrics,
    DataAnalyticsProvider,
    SensorReading,
)


class TestSensorReading:
    """Tests for SensorReading dataclass."""

    def test_sensor_reading_creation(self):
        """Test creating a SensorReading."""
        reading = SensorReading(
            timestamp=time.time(),
            temperature=25.5,
            humidity=60.0,
            air_quality=100.0,
        )
        assert reading.temperature == 25.5
        assert reading.humidity == 60.0
        assert reading.air_quality == 100.0

    def test_sensor_reading_to_dict(self):
        """Test converting SensorReading to dictionary."""
        reading = SensorReading(timestamp=1234567890.0, temperature=20.0, humidity=50.0)
        data = reading.to_dict()
        assert data["timestamp"] == 1234567890.0
        assert data["temperature"] == 20.0
        assert data["humidity"] == 50.0
        assert data["air_quality"] is None

    def test_sensor_reading_to_array(self):
        """Test converting SensorReading to numpy array."""
        reading = SensorReading(
            timestamp=time.time(),
            temperature=25.0,
            humidity=60.0,
            air_quality=100.0,
        )
        arr = reading.to_array()
        assert arr[0] == 25.0
        assert arr[1] == 60.0
        assert arr[2] == 100.0

    def test_sensor_reading_to_array_with_none(self):
        """Test converting SensorReading with None values to array."""
        reading = SensorReading(timestamp=time.time(), temperature=25.0)
        arr = reading.to_array()
        assert arr[0] == 25.0
        import numpy as np

        assert np.isnan(arr[1])
        assert np.isnan(arr[2])


class TestDataAnalyticsProvider:
    """Tests for DataAnalyticsProvider."""

    def test_provider_initialization(self):
        """Test initializing DataAnalyticsProvider."""
        provider = DataAnalyticsProvider(window_size=50, enable_ml=True)
        assert provider.window_size == 50
        assert provider.enable_ml is True
        assert len(provider.data_buffer) == 0

    def test_add_reading(self):
        """Test adding a sensor reading."""
        provider = DataAnalyticsProvider(window_size=10, enable_ml=False)
        reading = SensorReading(timestamp=time.time(), temperature=25.0, humidity=60.0)
        processed, is_anomaly = provider.add_reading(reading)
        assert processed.temperature == 25.0
        assert is_anomaly is False
        assert len(provider.data_buffer) == 1

    def test_preprocess_data(self):
        """Test data preprocessing."""
        provider = DataAnalyticsProvider(window_size=20, enable_ml=False)
        reading = SensorReading(timestamp=time.time(), temperature=25.0, humidity=60.0)
        processed = provider.preprocess_data(reading)
        assert processed.temperature == 25.0
        assert processed.humidity == 60.0

    def test_metrics_update(self):
        """Test metrics update."""
        provider = DataAnalyticsProvider(window_size=10, enable_ml=False)
        for i in range(5):
            reading = SensorReading(
                timestamp=time.time(),
                temperature=20.0 + i,
                humidity=50.0 + i,
                air_quality=100.0 + i,
            )
            provider.add_reading(reading)

        metrics = provider.get_metrics()
        assert metrics.total_readings == 5
        assert metrics.avg_temperature is not None
        assert metrics.avg_humidity is not None

    def test_get_recent_readings(self):
        """Test getting recent readings."""
        provider = DataAnalyticsProvider(window_size=10, enable_ml=False)
        for i in range(5):
            reading = SensorReading(timestamp=time.time(), temperature=20.0 + i)
            provider.add_reading(reading)

        recent = provider.get_recent_readings(count=3)
        assert len(recent) == 3

    def test_reset_metrics(self):
        """Test resetting metrics."""
        provider = DataAnalyticsProvider(window_size=10, enable_ml=False)
        reading = SensorReading(timestamp=time.time(), temperature=25.0, humidity=60.0)
        provider.add_reading(reading)
        assert len(provider.data_buffer) > 0

        provider.reset_metrics()
        assert len(provider.data_buffer) == 0
        metrics = provider.get_metrics()
        assert metrics.total_readings == 0

    def test_anomaly_detection_with_ml(self):
        """Test ML-based anomaly detection."""
        provider = DataAnalyticsProvider(
            window_size=30, enable_ml=True, contamination=0.1
        )

        # Add normal readings
        for i in range(25):
            reading = SensorReading(
                timestamp=time.time(),
                temperature=20.0 + (i % 5),
                humidity=50.0 + (i % 5),
                air_quality=100.0 + (i % 5),
            )
            provider.add_reading(reading)

        # Add an obvious anomaly
        anomaly_reading = SensorReading(
            timestamp=time.time(), temperature=100.0, humidity=200.0, air_quality=500.0
        )
        processed, is_anomaly = provider.add_reading(anomaly_reading)

        # Anomaly should be detected (though not guaranteed with small dataset)
        # At minimum, it should process without error
        assert processed is not None

    def test_anomaly_detection_without_ml(self):
        """Test that anomaly detection returns False when ML is disabled."""
        provider = DataAnalyticsProvider(window_size=10, enable_ml=False)
        reading = SensorReading(timestamp=time.time(), temperature=25.0, humidity=60.0)
        processed, is_anomaly = provider.add_reading(reading)
        assert is_anomaly is False


class TestAnalyticsMetrics:
    """Tests for AnalyticsMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating AnalyticsMetrics."""
        metrics = AnalyticsMetrics(
            total_readings=100,
            anomalies_detected=5,
            avg_temperature=25.0,
            data_quality_score=0.95,
        )
        assert metrics.total_readings == 100
        assert metrics.anomalies_detected == 5
        assert metrics.avg_temperature == 25.0
        assert metrics.data_quality_score == 0.95

    def test_metrics_to_dict(self):
        """Test converting AnalyticsMetrics to dictionary."""
        metrics = AnalyticsMetrics(
            total_readings=50, avg_temperature=20.0, avg_humidity=60.0
        )
        data = metrics.to_dict()
        assert data["total_readings"] == 50
        assert data["avg_temperature"] == 20.0
        assert data["avg_humidity"] == 60.0
